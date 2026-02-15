"""
LangChain integration for Synapse V2.

Provides drop-in replacements for LangChain's memory and vector store interfaces.
"""

import json
import time
from typing import Any, Dict, List, Optional, Type, Union

from synapse import Synapse

# LangChain imports with graceful fallback
try:
    from langchain.memory import BaseMemory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.embeddings import Embeddings
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create mock base classes for testing
    class BaseMemory:
        def __init__(self):
            pass
            
    class BaseChatMessageHistory:
        def __init__(self):
            pass
            
    class VectorStore:
        def __init__(self):
            pass
            
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
            
    class BaseMessage:
        def __init__(self, content: str):
            self.content = content
            
    class HumanMessage(BaseMessage):
        pass
        
    class AIMessage(BaseMessage):
        pass
        
    class Embeddings:
        pass
    
    LANGCHAIN_AVAILABLE = False


class SynapseMemory(BaseMemory):
    """LangChain memory backend using Synapse."""
    
    memory_key: str = "history"
    input_key: str = "input"  
    output_key: str = "output"
    return_messages: bool = False
    k: int = 5  # number of memories to recall
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output", 
        return_messages: bool = False,
        k: int = 5
    ):
        """Initialize SynapseMemory.
        
        Args:
            data_dir: Directory for persistent storage (None for in-memory)
            memory_key: Key to store memory in chain variables
            input_key: Key for input in chain context
            output_key: Key for output in chain context
            return_messages: Whether to return Message objects or strings
            k: Number of memories to recall for context
        """
        self.synapse = Synapse(data_dir or ":memory:")
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.k = k
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables (keys this memory provider will populate)."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from storage."""
        # Get input text for context
        input_text = inputs.get(self.input_key, "")
        
        # Recall relevant memories
        memories = self.synapse.recall(input_text, limit=self.k)
        
        if self.return_messages:
            # Return as message-like objects (for compatibility)
            memory_content = [{"role": "system", "content": mem.content} for mem in memories]
        else:
            # Return as formatted string
            if memories:
                memory_content = "\n".join([f"Memory: {mem.content}" for mem in memories])
            else:
                memory_content = ""
        
        return {self.memory_key: memory_content}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the input/output interaction to memory."""
        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")
        
        # Save both input and output as separate memories
        if input_text:
            self.synapse.remember(
                f"User input: {input_text}",
                memory_type="event",
                metadata={"interaction_type": "input"}
            )
        
        if output_text:
            self.synapse.remember(
                f"Assistant response: {output_text}",
                memory_type="event", 
                metadata={"interaction_type": "output"}
            )
    
    def clear(self) -> None:
        """Clear memory by creating a fresh Synapse instance."""
        data_dir = self.synapse.path if self.synapse.path != ":memory:" else None
        self.synapse.close()
        self.synapse = Synapse(data_dir or ":memory:")


class SynapseChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in Synapse."""
    
    def __init__(self, session_id: str = "default", data_dir: Optional[str] = None):
        """Initialize chat history.
        
        Args:
            session_id: Unique identifier for this chat session
            data_dir: Directory for persistent storage (None for in-memory)
        """
        self.session_id = session_id
        self.synapse = Synapse(data_dir or ":memory:")
        
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store."""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            role = "system"
            
        self.synapse.remember(
            message.content,
            memory_type="event",
            metadata={
                "session_id": self.session_id,
                "message_role": role,
                "timestamp": time.time()
            }
        )
    
    def add_user_message(self, message: str) -> None:
        """Convenience method to add a user message."""
        self.synapse.remember(
            message,
            memory_type="event", 
            metadata={
                "session_id": self.session_id,
                "message_role": "user",
                "timestamp": time.time()
            }
        )
    
    def add_ai_message(self, message: str) -> None:
        """Convenience method to add an AI message."""
        self.synapse.remember(
            message,
            memory_type="event",
            metadata={
                "session_id": self.session_id, 
                "message_role": "ai",
                "timestamp": time.time()
            }
        )
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages for this session, ordered by time."""
        # Recall all messages for this session
        all_memories = self.synapse.recall("", limit=1000)
        
        # Filter by session_id and sort by creation time
        session_memories = [
            mem for mem in all_memories
            if mem.metadata.get("session_id") == self.session_id
        ]
        session_memories.sort(key=lambda m: m.created_at)
        
        # Convert to message objects
        messages = []
        for memory in session_memories:
            role = memory.metadata.get("message_role", "system")
            if role == "user":
                messages.append(HumanMessage(content=memory.content))
            elif role == "ai":
                messages.append(AIMessage(content=memory.content))
        
        return messages
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        # For simplicity, we create a fresh instance
        # In a production system, you'd want to selectively delete
        data_dir = self.synapse.path if self.synapse.path != ":memory:" else None
        self.synapse.close()
        self.synapse = Synapse(data_dir or ":memory:")


class SynapseVectorStore(VectorStore):
    """Use Synapse as a LangChain vector store (BM25 + concept graph)."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory for persistent storage (None for in-memory)
        """
        self.synapse = Synapse(data_dir or ":memory:")
        
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add texts to the store and return their IDs."""
        metadatas = metadatas or [{}] * len(texts)
        ids = []
        
        for text, metadata in zip(texts, metadatas):
            memory = self.synapse.remember(
                text,
                memory_type="fact",
                metadata=metadata
            )
            ids.append(str(memory.id))
            
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to query."""
        memories = self.synapse.recall(query, limit=k)
        
        documents = []
        for memory in memories:
            doc = Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.id,
                    "score": memory.effective_strength,
                    "memory_type": memory.memory_type,
                    **memory.metadata
                }
            )
            documents.append(doc)
            
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Return documents with similarity scores."""
        memories = self.synapse.recall(query, limit=k)
        
        results = []
        for memory in memories:
            doc = Document(
                page_content=memory.content,
                metadata={
                    "memory_id": memory.id,
                    "memory_type": memory.memory_type,
                    **memory.metadata
                }
            )
            results.append((doc, memory.effective_strength))
            
        return results
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,  # Ignored - Synapse handles this
        metadatas: Optional[List[dict]] = None,
        data_dir: Optional[str] = None,
        **kwargs: Any
    ) -> "SynapseVectorStore":
        """Create a SynapseVectorStore from a list of texts."""
        instance = cls(data_dir=data_dir)
        instance.add_texts(texts, metadatas, **kwargs)
        return instance
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs."""
        if not ids:
            return False
            
        for id_str in ids:
            try:
                memory_id = int(id_str)
                self.synapse.forget(memory_id)
            except (ValueError, TypeError):
                continue
                
        return True
    
    def get(self, ids: List[str], **kwargs: Any) -> List[Document]:
        """Get documents by IDs."""
        documents = []
        for id_str in ids:
            try:
                memory_id = int(id_str)
                # Get memory by ID (need to implement this in Synapse if not available)
                memories = self.synapse.recall("", limit=1000)
                memory = next((m for m in memories if m.id == memory_id), None)
                if memory:
                    doc = Document(
                        page_content=memory.content,
                        metadata={
                            "memory_id": memory.id,
                            "memory_type": memory.memory_type,
                            **memory.metadata
                        }
                    )
                    documents.append(doc)
            except (ValueError, TypeError):
                continue
                
        return documents


# Convenience functions
def create_synapse_memory(**kwargs) -> SynapseMemory:
    """Create a SynapseMemory instance with convenient defaults."""
    return SynapseMemory(**kwargs)


def create_synapse_chat_history(session_id: str = "default", **kwargs) -> SynapseChatMessageHistory:
    """Create a SynapseChatMessageHistory instance."""
    return SynapseChatMessageHistory(session_id=session_id, **kwargs)


def create_synapse_vectorstore(**kwargs) -> SynapseVectorStore:
    """Create a SynapseVectorStore instance."""
    return SynapseVectorStore(**kwargs)