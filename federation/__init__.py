"""Federation layer — Git-like memory sync between Synapse agents."""

from .node import SynapseNode
from .server import SynapseServer
from .client import SyncClient
from .sync import SyncEngine
from .memory_object import FederatedMemory, FederatedEdge
from .vector_clock import VectorClock

__all__ = [
    "SynapseNode",
    "SynapseServer",
    "SyncClient",
    "SyncEngine",
    "FederatedMemory",
    "FederatedEdge",
    "VectorClock",
]
