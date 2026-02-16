"""LangChain integration for Synapse AI Memory — privacy-first memory for LangChain chains and agents."""

from .memory import SynapseMemory
from .chat_history import SynapseChatMessageHistory
from .retriever import SynapseRetriever

try:
    import langchain  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

__all__ = ["SynapseMemory", "SynapseChatMessageHistory", "SynapseRetriever", "LANGCHAIN_AVAILABLE"]
