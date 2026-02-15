"""
Synapse Framework Integrations

This package provides drop-in replacements for popular AI framework components
that use Synapse as the memory backend.
"""

# LangChain integrations
try:
    from .langchain import (
        SynapseMemory,
        SynapseChatMessageHistory,
        SynapseVectorStore,
        create_synapse_memory,
        create_synapse_chat_history,
        create_synapse_vectorstore,
        LANGCHAIN_AVAILABLE
    )
    
    # Only export if LangChain is available
    if LANGCHAIN_AVAILABLE:
        __all_langchain__ = [
            "SynapseMemory",
            "SynapseChatMessageHistory", 
            "SynapseVectorStore",
            "create_synapse_memory",
            "create_synapse_chat_history",
            "create_synapse_vectorstore"
        ]
    else:
        __all_langchain__ = []
        
except ImportError:
    __all_langchain__ = []
    LANGCHAIN_AVAILABLE = False


# LangGraph integrations
try:
    from .langgraph import (
        SynapseCheckpointer,
        SynapseMemoryStore,
        create_memory_enhanced_graph,
        create_persistent_agent,
        AgentState,
        LANGGRAPH_AVAILABLE
    )
    
    # Only export if LangGraph is available
    if LANGGRAPH_AVAILABLE:
        __all_langgraph__ = [
            "SynapseCheckpointer",
            "SynapseMemoryStore", 
            "create_memory_enhanced_graph",
            "create_persistent_agent",
            "AgentState"
        ]
    else:
        __all_langgraph__ = []
        
except ImportError:
    __all_langgraph__ = []
    LANGGRAPH_AVAILABLE = False


# Combine all exports
__all__ = __all_langchain__ + __all_langgraph__ + [
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE"
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Raghuram Parvataneni"
__description__ = "Framework integrations for Synapse AI Memory"


def list_available_integrations():
    """List all available integrations based on installed dependencies."""
    integrations = {}
    
    if LANGCHAIN_AVAILABLE:
        integrations["langchain"] = {
            "status": "available",
            "components": __all_langchain__,
            "description": "LangChain memory and vector store components"
        }
    else:
        integrations["langchain"] = {
            "status": "unavailable", 
            "reason": "langchain not installed",
            "install": "pip install langchain langchain-core"
        }
    
    if LANGGRAPH_AVAILABLE:
        integrations["langgraph"] = {
            "status": "available",
            "components": __all_langgraph__,
            "description": "LangGraph checkpointer and memory components"
        }
    else:
        integrations["langgraph"] = {
            "status": "unavailable",
            "reason": "langgraph not installed", 
            "install": "pip install langgraph"
        }
    
    return integrations


# Convenience imports for when both are available
if LANGCHAIN_AVAILABLE and LANGGRAPH_AVAILABLE:
    def create_full_agent_memory(data_dir: str = None):
        """Create a complete memory system for AI agents.
        
        Returns a dictionary with all memory components configured.
        """
        return {
            "langchain_memory": SynapseMemory(data_dir=data_dir),
            "chat_history": SynapseChatMessageHistory(data_dir=data_dir),
            "vector_store": SynapseVectorStore(data_dir=data_dir),
            "memory_store": SynapseMemoryStore(data_dir=data_dir),
            "checkpointer": SynapseCheckpointer(data_dir=data_dir)
        }
    
    __all__.append("create_full_agent_memory")


# Module-level convenience function
def get_integration_status():
    """Get a quick status of integration availability."""
    return {
        "langchain": LANGCHAIN_AVAILABLE,
        "langgraph": LANGGRAPH_AVAILABLE,
        "all_available": LANGCHAIN_AVAILABLE and LANGGRAPH_AVAILABLE
    }