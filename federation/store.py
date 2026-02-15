"""Federated memory store — manages content-addressed memories with Merkle tracking."""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from federation.content_hash import memory_hash
from federation.merkle import MerkleTree
from federation.memory_object import FederatedMemory, FederatedEdge
from federation.vector_clock import VectorClock


class FederatedStore:
    """
    Content-addressed memory store with Merkle tree tracking.
    This wraps the concept of a local memory database that can sync.
    """

    def __init__(self, node_id: str, path: str = ":memory:"):
        self.node_id = node_id
        self.path = path

        # Content-addressed stores: hash -> object
        self.memories: Dict[str, FederatedMemory] = {}
        self.edges: Dict[str, FederatedEdge] = {}

        # Merkle trees for efficient delta detection
        self.memory_tree = MerkleTree()
        self.edge_tree = MerkleTree()

        # Namespace index: namespace -> set of memory hashes
        self._ns_index: Dict[str, Set[str]] = {}

        # Shared namespaces (what this node exposes)
        self._shared_namespaces: Set[str] = set()

        # Subscriptions: peer_url -> set of namespaces
        self._subscriptions: Dict[str, Set[str]] = {}

        # Known peers
        self.peers: Dict[str, Dict[str, Any]] = {}  # url -> info

        # Load from disk if path provided
        if path != ":memory:":
            self._load()

    # ── Memory operations ─────────────────────────────────────

    def add_memory(self, memory: FederatedMemory) -> str:
        """Add a federated memory. Returns its hash."""
        if memory.hash in self.memories:
            # Already have it — merge vector clocks
            existing = self.memories[memory.hash]
            existing.vclock.merge(memory.vclock)
            # Merge namespaces
            ns_set = set(existing.namespaces) | set(memory.namespaces)
            existing.namespaces = sorted(ns_set)
            return memory.hash

        self.memories[memory.hash] = memory
        self.memory_tree.add(memory.hash)

        # Update namespace index
        for ns in memory.namespaces:
            self._ns_index.setdefault(ns, set()).add(memory.hash)

        return memory.hash

    def add_edge(self, edge: FederatedEdge) -> str:
        """Add a federated edge. Returns its hash."""
        if edge.hash in self.edges:
            existing = self.edges[edge.hash]
            existing.vclock.merge(edge.vclock)
            return edge.hash

        self.edges[edge.hash] = edge
        self.edge_tree.add(edge.hash)
        return edge.hash

    def remove_memory(self, content_hash: str) -> bool:
        if content_hash in self.memories:
            mem = self.memories.pop(content_hash)
            self.memory_tree.remove(content_hash)
            for ns in mem.namespaces:
                if ns in self._ns_index:
                    self._ns_index[ns].discard(content_hash)
            return True
        return False

    def get_memory(self, content_hash: str) -> Optional[FederatedMemory]:
        return self.memories.get(content_hash)

    def get_memories_by_namespace(self, namespace: str) -> List[FederatedMemory]:
        """Get all memories in a namespace."""
        hashes = self._ns_index.get(namespace, set())
        return [self.memories[h] for h in hashes if h in self.memories]

    def all_hashes(self) -> Set[str]:
        return set(self.memories.keys())

    def all_edge_hashes(self) -> Set[str]:
        return set(self.edges.keys())

    # ── Namespace management ──────────────────────────────────

    def share(self, namespace: str):
        """Mark a namespace as shared (will be offered to peers)."""
        self._shared_namespaces.add(namespace)

    def unshare(self, namespace: str):
        self._shared_namespaces.discard(namespace)

    @property
    def shared_namespaces(self) -> Set[str]:
        return set(self._shared_namespaces)

    def subscribe(self, peer_url: str, namespace: str):
        """Subscribe to a namespace from a peer."""
        self._subscriptions.setdefault(peer_url, set()).add(namespace)

    def get_subscriptions(self, peer_url: str) -> Set[str]:
        return self._subscriptions.get(peer_url, set())

    # ── Filtering for sync ────────────────────────────────────

    def get_exportable_hashes(self, namespaces: Set[str] | None = None) -> Set[str]:
        """Get memory hashes that match the given namespaces (or all shared)."""
        if namespaces is None:
            namespaces = self._shared_namespaces

        if not namespaces:
            # No namespace filter — return all
            return self.all_hashes()

        result = set()
        for ns in namespaces:
            result.update(self._ns_index.get(ns, set()))
        return result

    def get_exportable_memories(self, hashes: Set[str]) -> List[Dict[str, Any]]:
        """Serialize memories by hash for wire transfer."""
        return [self.memories[h].to_dict() for h in hashes if h in self.memories]

    def get_exportable_edges(self, memory_hashes: Set[str]) -> List[Dict[str, Any]]:
        """Get edges where both endpoints are in the given memory set."""
        result = []
        for edge in self.edges.values():
            if edge.source_hash in memory_hashes and edge.target_hash in memory_hashes:
                result.append(edge.to_dict())
        return result

    # ── Persistence ───────────────────────────────────────────

    def save(self):
        if self.path == ":memory:":
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "node_id": self.node_id,
            "memories": {h: m.to_dict() for h, m in self.memories.items()},
            "edges": {h: e.to_dict() for h, e in self.edges.items()},
            "shared_namespaces": sorted(self._shared_namespaces),
            "subscriptions": {url: sorted(ns) for url, ns in self._subscriptions.items()},
            "peers": self.peers,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self.path)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        for d in data.get("memories", {}).values():
            mem = FederatedMemory.from_dict(d)
            self.memories[mem.hash] = mem
            self.memory_tree.add(mem.hash)
            for ns in mem.namespaces:
                self._ns_index.setdefault(ns, set()).add(mem.hash)

        for d in data.get("edges", {}).values():
            edge = FederatedEdge.from_dict(d)
            self.edges[edge.hash] = edge
            self.edge_tree.add(edge.hash)

        self._shared_namespaces = set(data.get("shared_namespaces", []))
        self._subscriptions = {
            url: set(ns) for url, ns in data.get("subscriptions", {}).items()
        }
        self.peers = data.get("peers", {})
