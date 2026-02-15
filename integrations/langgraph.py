"""
LangGraph integration for Synapse V2.

Provides memory components for LangGraph's state-based agent architecture.
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

from synapse import Synapse

# LangGraph imports with graceful fallback
try:
    from langgraph.checkpoint import BaseCheckpointSaver, CheckpointTuple
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Create mock base classes for testing
    class BaseCheckpointSaver:
        def __init__(self):
            pass
            
    class CheckpointTuple:
        def __init__(self, config: dict, checkpoint: dict, metadata: dict = None):
            self.config = config
            self.checkpoint = checkpoint
            self.metadata = metadata or {}
    
    LANGGRAPH_AVAILABLE = False


class SynapseCheckpointer:
    """LangGraph-compatible checkpointer using Synapse for state persistence.

    Important: this is NOT a drop-in replacement for LangGraph's checkpoint savers.
    We do not subclass LangGraph's `BaseCheckpointSaver`, so you cannot pass this
    directly to `compile(checkpointer=...)`. Use it as a small, compatible helper
    for storing and loading checkpoints in a LangGraph-like shape.

    Checkpoints are stored as Synapse memories whose content begins with a unique
    tag: `synapse_checkpoint_{thread_id}`. Retrieval uses this tag for exact-ish
    lookup instead of relying on BM25 matching of a JSON blob.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the checkpointer.
        
        Args:
            data_dir: Directory for persistent storage (None for in-memory)
        """
        self.synapse = Synapse(data_dir or ":memory:")

    @staticmethod
    def _tag(thread_id: str) -> str:
        return f"synapse_checkpoint_{thread_id}"

    @staticmethod
    def _extract_json(content: str, tag: str) -> Optional[dict]:
        if not content:
            return None
        if not content.startswith(tag):
            return None
        # Expected format: "{tag}\n{json}"
        parts = content.split("\n", 1)
        if len(parts) != 2:
            return None
        try:
            return json.loads(parts[1])
        except json.JSONDecodeError:
            return None
        
    def get(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Load checkpoint from Synapse storage."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        tag = self._tag(thread_id)
        # Avoid BM25-based search for exact lookups; scan and match the unique tag.
        memories = self.synapse.recall("", limit=10000)
        candidates = [
            m
            for m in memories
            if isinstance(getattr(m, "content", None), str) and m.content.startswith(tag)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda m: getattr(m, "created_at", 0), reverse=True)

        checkpoint_data = self._extract_json(candidates[0].content, tag)
        if not checkpoint_data:
            return None

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint_data.get("checkpoint", {}),
            metadata=checkpoint_data.get("metadata", {}),
        )
    
    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint to Synapse storage."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required in config.configurable")
            
        checkpoint_data = {
            "checkpoint": checkpoint,
            "metadata": {
                "thread_id": thread_id,
                "saved_at": time.time(),
                "config": config
            }
        }

        tag = self._tag(thread_id)

        # Remove any existing checkpoint for this thread_id to keep get() deterministic.
        existing = self.synapse.recall("", limit=10000)
        for mem in existing:
            if isinstance(getattr(mem, "content", None), str) and mem.content.startswith(tag):
                self.synapse.forget(mem.id)

        content = f"{tag}\n{json.dumps(checkpoint_data, separators=(',', ':'), sort_keys=True)}"
        self.synapse.remember(
            content,
            memory_type="event",
            metadata={
                "checkpoint_thread_id": thread_id,
                "checkpoint_type": "langgraph_state",
                "synapse_checkpoint_tag": tag,
            },
        )
    
    def list(self, config: Optional[Dict[str, Any]] = None) -> List[CheckpointTuple]:
        """List all checkpoints, optionally filtered by config."""
        thread_id_filter = None
        if config:
            thread_id_filter = config.get("configurable", {}).get("thread_id")

        # Use the tag prefix so recall isn't dependent on JSON tokenization.
        memories = self.synapse.recall("", limit=10000)
        
        checkpoints = []
        for memory in memories:
            content = getattr(memory, "content", "")
            if not isinstance(content, str):
                continue
            if not content.startswith("synapse_checkpoint_"):
                continue

            tag = content.split("\n", 1)[0]
            if thread_id_filter and tag != self._tag(thread_id_filter):
                continue

            checkpoint_data = self._extract_json(content, tag)
            if not checkpoint_data:
                continue

            stored_config = checkpoint_data.get("metadata", {}).get("config", {}) or {}
            checkpoints.append(
                CheckpointTuple(
                    config=stored_config,
                    checkpoint=checkpoint_data.get("checkpoint", {}),
                    metadata=checkpoint_data.get("metadata", {}),
                )
            )
                
        return checkpoints


class SynapseMemoryStore:
    """Long-term memory store for LangGraph agents.
    
    Use within LangGraph nodes to remember/recall across conversations.
    Provides both direct methods and LangGraph node helpers.
    
    Example usage:
        ```python
        from langgraph.graph import StateGraph
        from integrations.langgraph import SynapseMemoryStore

        memory = SynapseMemoryStore(data_dir="./agent_memory")

        # Define graph
        graph = StateGraph(AgentState)
        graph.add_node("recall", memory.as_recall_node())
        graph.add_node("respond", respond_fn)
        graph.add_node("remember", memory.as_remember_node())

        graph.add_edge("recall", "respond")
        graph.add_edge("respond", "remember")
        ```
    """
    
    def __init__(self, data_dir: Optional[str] = None, extract: bool = False):
        """Initialize the memory store.
        
        Args:
            data_dir: Directory for persistent storage (None for in-memory)
            extract: Whether to use fact extraction when remembering
        """
        self.synapse = Synapse(data_dir or ":memory:")
        self.extract = extract
        
    def remember(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store a memory and return its ID.
        
        Args:
            content: Content to remember
            metadata: Additional metadata to store
            
        Returns:
            Memory ID
        """
        memory = self.synapse.remember(
            content,
            memory_type="fact",
            metadata=metadata or {},
            extract=self.extract
        )
        return memory.id
    
    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant memories.
        
        Args:
            query: Query string for memory retrieval
            k: Number of memories to return
            
        Returns:
            List of memory dictionaries with content, id, score, etc.
        """
        memories = self.synapse.recall(query, limit=k)
        return [
            {
                "id": mem.id,
                "content": mem.content,
                "score": mem.effective_strength,
                "memory_type": mem.memory_type,
                "created_at": mem.created_at,
                "metadata": mem.metadata
            }
            for mem in memories
        ]
    
    def forget(self, memory_id: int) -> bool:
        """Remove a memory by ID.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if memory was removed, False if not found
        """
        return self.synapse.forget(memory_id)
    
    def as_remember_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Return a function usable as a LangGraph node that saves state to memory.
        
        The returned function expects state with 'content' and optionally 'metadata' keys.
        It adds a 'memory_id' to the state after saving.
        
        Returns:
            Function that can be used as a LangGraph node
        """
        def remember_node(state: Dict[str, Any]) -> Dict[str, Any]:
            content = state.get("content", "")
            metadata = state.get("metadata", {})
            
            if not content:
                return state
                
            # Add agent context to metadata
            metadata.update({
                "agent_memory": True,
                "stored_at": time.time(),
                "session_id": state.get("session_id", "default")
            })
            
            memory_id = self.remember(content, metadata)
            
            # Return state with memory_id added
            return {**state, "memory_id": memory_id}
            
        return remember_node
    
    def as_recall_node(self, query_key: str = "query", memories_key: str = "memories") -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Return a function usable as a LangGraph node that loads relevant memories into state.
        
        Args:
            query_key: Key in state containing the query string
            memories_key: Key to store recalled memories in state
            
        Returns:
            Function that can be used as a LangGraph node
        """
        def recall_node(state: Dict[str, Any]) -> Dict[str, Any]:
            query = state.get(query_key, "")
            k = state.get("recall_limit", 5)
            
            if not query:
                # No query - return state with empty memories
                return {**state, memories_key: []}
                
            memories = self.recall(query, k=k)
            
            # Return state with memories added
            return {**state, memories_key: memories}
            
        return recall_node
    
    def as_memory_aware_node(self, 
                            node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                            auto_recall: bool = True,
                            auto_remember: bool = True) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Wrap a node function to automatically handle memory recall/storage.
        
        Args:
            node_fn: Original node function to wrap
            auto_recall: If True, automatically recall memories before calling node_fn
            auto_remember: If True, automatically remember the response after calling node_fn
            
        Returns:
            Wrapped function with automatic memory handling
        """
        def memory_aware_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Auto-recall memories if enabled
            if auto_recall:
                query = state.get("query", state.get("input", ""))
                if query:
                    memories = self.recall(query, k=state.get("recall_limit", 5))
                    state = {**state, "memories": memories}
            
            # Call the original node function
            result_state = node_fn(state)
            
            # Auto-remember response if enabled
            if auto_remember:
                response = result_state.get("response", result_state.get("output", ""))
                if response:
                    metadata = {
                        "node_function": node_fn.__name__,
                        "session_id": state.get("session_id", "default"),
                        "auto_stored": True
                    }
                    memory_id = self.remember(response, metadata)
                    result_state = {**result_state, "memory_id": memory_id}
            
            return result_state
            
        return memory_aware_node


# Utility functions for common LangGraph patterns
def create_memory_enhanced_graph(base_graph_fn: Callable, memory_store: SynapseMemoryStore) -> Any:
    """Create a memory-enhanced version of a LangGraph graph.
    
    Args:
        base_graph_fn: Function that returns a base StateGraph
        memory_store: SynapseMemoryStore instance to use
        
    Returns:
        Enhanced graph with memory nodes added
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not available. Install with: pip install langgraph")
    
    graph = base_graph_fn()
    
    # Add memory nodes
    graph.add_node("recall_memories", memory_store.as_recall_node())
    graph.add_node("store_memory", memory_store.as_remember_node())
    
    return graph


def create_persistent_agent(state_schema: type, 
                          memory_data_dir: Optional[str] = None,
                          checkpoint_data_dir: Optional[str] = None) -> Tuple[SynapseMemoryStore, SynapseCheckpointer]:
    """Create a memory store and checkpointer for a persistent agent.
    
    Args:
        state_schema: State schema class for the agent
        memory_data_dir: Directory for long-term memory storage
        checkpoint_data_dir: Directory for conversation checkpoints
        
    Returns:
        Tuple of (memory_store, checkpointer)
    """
    memory_store = SynapseMemoryStore(data_dir=memory_data_dir)
    checkpointer = SynapseCheckpointer(data_dir=checkpoint_data_dir)
    
    return memory_store, checkpointer


# Example state schema for reference
class AgentState:
    """Example state schema for LangGraph agents using Synapse memory."""
    
    def __init__(self):
        self.query: str = ""
        self.response: str = ""
        self.memories: List[Dict[str, Any]] = []
        self.memory_id: Optional[int] = None
        self.session_id: str = "default"
        self.metadata: Dict[str, Any] = {}
