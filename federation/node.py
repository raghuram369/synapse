"""SynapseNode — the high-level API for a federated memory peer."""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set

from federation.store import FederatedStore
from federation.sync import SyncEngine
from federation.server import SynapseServer
from federation.client import SyncClient
from federation.discovery import PeerDiscovery
from federation.memory_object import FederatedMemory, FederatedEdge
from federation.vector_clock import VectorClock


class SynapseNode:
    """
    A federated memory peer — the main entry point.
    
    Usage:
        node = SynapseNode(node_id="coding-agent")
        node.remember("Python uses GIL for thread safety", namespaces=["public"])
        node.listen(port=9470)
        node.push("http://peer:9470")
        node.pull("http://peer:9470")
    """

    def __init__(
        self,
        node_id: str,
        path: str = ":memory:",
        synapse: Any = None,  # Optional Synapse v2 instance for interop
        auth_token: Optional[str] = None,
    ):
        self.node_id = node_id
        self.store = FederatedStore(node_id, path)
        self.engine = SyncEngine(self.store)
        self.client = SyncClient(self.store)
        self.auth_token = auth_token
        self._server: Optional[SynapseServer] = None
        self._discovery: Optional[PeerDiscovery] = None
        self._synapse = synapse  # v2 interop

    def add_peer(self, url: str, token: Optional[str] = None):
        """Add a known peer, optionally with an auth token."""
        self.store.peers[url] = {"node_id": "unknown"}
        if token:
            self.client.set_peer_token(url, token)

    # ── Memory operations ─────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        metadata: Dict[str, Any] | None = None,
        namespaces: List[str] | None = None,
    ) -> FederatedMemory:
        """Store a memory locally."""
        vclock = VectorClock().increment(self.node_id)
        mem = FederatedMemory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            origin_node=self.node_id,
            namespaces=namespaces or [],
            vclock=vclock,
        )
        self.store.add_memory(mem)
        return mem

    def link(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
        weight: float = 1.0,
    ) -> FederatedEdge:
        """Create a link between two memories."""
        vclock = VectorClock().increment(self.node_id)
        edge = FederatedEdge(
            source_hash=source_hash,
            target_hash=target_hash,
            edge_type=edge_type,
            weight=weight,
            origin_node=self.node_id,
            vclock=vclock,
        )
        self.store.add_edge(edge)
        return edge

    def memories(self, namespace: str | None = None) -> List[FederatedMemory]:
        """List memories, optionally filtered by namespace."""
        if namespace:
            return self.store.get_memories_by_namespace(namespace)
        return list(self.store.memories.values())

    def forget(self, content_hash: str) -> bool:
        """Remove a memory."""
        return self.store.remove_memory(content_hash)

    # ── Namespace management ──────────────────────────────────

    def share(self, namespace: str):
        """Mark a namespace as shared with peers."""
        self.store.share(namespace)

    def unshare(self, namespace: str):
        self.store.unshare(namespace)

    def subscribe(self, peer_url: str, namespace: str):
        """Subscribe to a namespace from a specific peer."""
        self.store.subscribe(peer_url, namespace)

    # ── Network operations ────────────────────────────────────

    def listen(self, port: int = 9470, host: str = "0.0.0.0"):
        """Start the HTTP server for incoming sync requests."""
        self._server = SynapseServer(self.store, host, port, auth_token=self.auth_token)
        self._server.start()

    def stop(self):
        """Stop the server and discovery."""
        if self._server:
            self._server.stop()
            self._server = None
        if self._discovery:
            self._discovery.stop()
            self._discovery = None

    def push(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """Push memories to a remote peer."""
        return self.client.push(peer_url, namespaces)

    def pull(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """Pull memories from a remote peer."""
        return self.client.pull(peer_url, namespaces)

    def sync(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """Bidirectional sync with a remote peer."""
        return self.client.sync(peer_url, namespaces)

    def peers(self) -> Dict[str, Dict]:
        """Return known peers (from discovery + explicit)."""
        result = dict(self.store.peers)
        if self._discovery:
            result.update(self._discovery.peers)
        return result

    # ── Discovery ─────────────────────────────────────────────

    def start_discovery(self, port: int = 9470):
        """Start LAN peer discovery."""
        def on_found(url, info):
            self.store.peers[url] = info

        self._discovery = PeerDiscovery(
            self.node_id, port=port, on_peer_found=on_found
        )
        self._discovery.start()

    # ── Persistence ───────────────────────────────────────────

    def save(self):
        self.store.save()

    # ── Status ────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "memory_count": len(self.store.memories),
            "edge_count": len(self.store.edges),
            "root_hash": self.store.memory_tree.root,
            "shared_namespaces": sorted(self.store.shared_namespaces),
            "peers": len(self.peers()),
            "server_running": self._server is not None,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def __repr__(self):
        return f"SynapseNode({self.node_id!r}, memories={len(self.store.memories)})"
