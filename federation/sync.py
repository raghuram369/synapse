"""Sync protocol — delta exchange between two federated stores."""

from __future__ import annotations
import io
import sys
import os
from typing import Any, Dict, List, Set, Tuple

from federation.store import FederatedStore
from federation.memory_object import FederatedMemory, FederatedEdge
from federation.vector_clock import VectorClock

# portable module is now co-located in the main package


class SyncEngine:
    """
    Handles the sync protocol between two stores.
    
    Protocol flow (push):
      1. Pusher sends: {root_hash, bucket_hashes, shared_namespaces}
      2. Receiver replies: {differing_buckets}
      3. Pusher sends items in differing buckets
      4. Receiver replies with which items it's missing
      5. Pusher sends full memory objects for missing items
      6. Receiver ingests and returns ACK
    
    For in-process sync (tests), we short-circuit the HTTP layer.
    """

    def __init__(self, store: FederatedStore):
        self.store = store

    # ── Phase 1: Exchange tree state ──────────────────────────

    def get_tree_state(self, namespaces: Set[str] | None = None) -> Dict[str, Any]:
        """Return Merkle tree state for the given namespaces."""
        exportable = self.store.get_exportable_hashes(namespaces)

        # Build a temporary tree for just the exportable subset
        from federation.merkle import MerkleTree
        tree = MerkleTree()
        tree.add_many(exportable)

        return {
            "root": tree.root,
            "bucket_hashes": tree.bucket_hashes,
            "count": len(exportable),
            "node_id": self.store.node_id,
            "shared_namespaces": sorted(self.store.shared_namespaces),
        }

    # ── Phase 2: Compute delta ────────────────────────────────

    def compute_missing(
        self,
        remote_tree_state: Dict[str, Any],
        namespaces: Set[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Given remote's tree state, figure out what we're missing.
        Returns {differing_buckets, request_hashes_in_buckets}.
        """
        exportable = self.store.get_exportable_hashes(namespaces)

        from federation.merkle import MerkleTree
        local_tree = MerkleTree()
        local_tree.add_many(exportable)

        # Same root? Nothing to do
        if local_tree.root == remote_tree_state["root"]:
            return {"status": "in_sync", "missing": []}

        # Find differing buckets
        remote_buckets = {
            int(k): v for k, v in remote_tree_state["bucket_hashes"].items()
        }
        diff_buckets = local_tree.diff_buckets(remote_buckets)

        return {
            "status": "needs_sync",
            "differing_buckets": diff_buckets,
        }

    def get_items_in_buckets(
        self, bucket_ids: List[int], namespaces: Set[str] | None = None
    ) -> Set[str]:
        """Return hashes in the specified buckets (filtered by namespace)."""
        exportable = self.store.get_exportable_hashes(namespaces)

        from federation.merkle import MerkleTree
        tree = MerkleTree()
        tree.add_many(exportable)
        return tree.items_in_buckets(bucket_ids)

    def get_missing_hashes(
        self, remote_hashes: Set[str], namespaces: Set[str] | None = None
    ) -> Set[str]:
        """Return hashes from remote that we don't have locally."""
        local = self.store.get_exportable_hashes(namespaces)
        return remote_hashes - local

    # ── Phase 3: Transfer objects ─────────────────────────────

    def export_memories(self, hashes: Set[str]) -> List[Dict[str, Any]]:
        """Export full memory objects for the given hashes."""
        return self.store.get_exportable_memories(hashes)

    def export_edges(self, memory_hashes: Set[str]) -> List[Dict[str, Any]]:
        """Export edges whose both endpoints are in memory_hashes."""
        return self.store.get_exportable_edges(memory_hashes)

    def export_synapse_binary(self, hashes: Set[str]) -> bytes:
        """Export memories and edges as a .synapse binary blob using Phase 2 format."""
        import tempfile
        try:
            from portable import SynapseWriter
        except ImportError:
            # Fallback: return None so callers use JSON
            return None

        memories = self.store.get_exportable_memories(hashes)
        all_hashes = self.store.all_hashes()
        edges = self.store.get_exportable_edges(all_hashes)

        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            tmp_path = f.name

        try:
            writer = SynapseWriter(tmp_path, source_agent=self.store.node_id)
            for m in memories:
                writer.add_memory(m)
            for e in edges:
                writer.add_edge(e)
            writer.write()

            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    def import_synapse_binary(self, data: bytes) -> Dict[str, int]:
        """Import memories and edges from a .synapse binary blob."""
        import tempfile
        try:
            from portable import SynapseReader
        except ImportError:
            return {"memories": 0, "edges": 0}

        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            f.write(data)
            tmp_path = f.name

        try:
            reader = SynapseReader(tmp_path)
            mem_count = 0
            for record in reader.iter_memories():
                mem = FederatedMemory.from_dict(record)
                if mem.hash not in self.store.memories:
                    self.store.add_memory(mem)
                    mem_count += 1
                else:
                    self.store.memories[mem.hash].vclock.merge(mem.vclock)

            edge_count = 0
            for record in reader.iter_edges():
                # Skip V2-format edges that use integer IDs instead of content hashes
                if "source_hash" not in record:
                    continue
                try:
                    edge = FederatedEdge.from_dict(record)
                except (KeyError, ValueError):
                    continue
                if edge.hash not in self.store.edges:
                    self.store.add_edge(edge)
                    edge_count += 1
                else:
                    self.store.edges[edge.hash].vclock.merge(edge.vclock)

            return {"memories": mem_count, "edges": edge_count}
        finally:
            os.unlink(tmp_path)

    def import_memories(self, memory_dicts: List[Dict[str, Any]]) -> int:
        """Import memories from wire format. Returns count imported."""
        count = 0
        for d in memory_dicts:
            mem = FederatedMemory.from_dict(d)
            if mem.hash not in self.store.memories:
                self.store.add_memory(mem)
                count += 1
            else:
                # Merge vclock
                self.store.memories[mem.hash].vclock.merge(mem.vclock)
        return count

    def import_edges(self, edge_dicts: List[Dict[str, Any]]) -> int:
        """Import edges from wire format. Returns count imported."""
        count = 0
        for d in edge_dicts:
            edge = FederatedEdge.from_dict(d)
            if edge.hash not in self.store.edges:
                self.store.add_edge(edge)
                count += 1
            else:
                self.store.edges[edge.hash].vclock.merge(edge.vclock)
        return count

    # ── Full sync (in-process, no network) ────────────────────

    def sync_with(
        self,
        other: "SyncEngine",
        namespaces: Set[str] | None = None,
        bidirectional: bool = True,
    ) -> Dict[str, int]:
        """
        Full sync with another engine (in-process, for tests/local use).
        Returns stats: {pushed, pulled}.
        """
        stats = {"pushed": 0, "pulled": 0}

        # Pull: get what other has that we don't
        stats["pulled"] = self._pull_from(other, namespaces)

        # Push: give other what we have that they don't
        if bidirectional:
            stats["pushed"] = other._pull_from(self, namespaces)

        return stats

    def _pull_from(
        self, remote: "SyncEngine", namespaces: Set[str] | None = None
    ) -> int:
        """Pull missing items from remote. Returns count of new memories."""
        # Phase 1: Get remote tree state
        remote_state = remote.get_tree_state(namespaces)

        # Phase 2: Compute what we're missing
        delta = self.compute_missing(remote_state, namespaces)
        if delta["status"] == "in_sync":
            return 0

        # Phase 2b: Get hashes in differing buckets from remote
        diff_buckets = delta["differing_buckets"]
        remote_hashes = remote.get_items_in_buckets(diff_buckets, namespaces)

        # Phase 2c: Figure out which of those we're missing
        missing = remote_hashes - self.store.all_hashes()
        if not missing:
            return 0

        # Phase 3: Get full objects
        memory_dicts = remote.export_memories(missing)
        count = self.import_memories(memory_dicts)

        # Also sync edges for the transferred memories
        all_relevant = self.store.all_hashes()
        edge_dicts = remote.export_edges(all_relevant)
        self.import_edges(edge_dicts)

        return count


def resolve_conflict(
    local: FederatedMemory, remote: FederatedMemory
) -> FederatedMemory:
    """
    Resolve conflict between two versions of a memory.
    
    Strategy:
    1. If one happened-before the other (vector clock), take the later one
    2. If concurrent, last-writer-wins by created_at timestamp
    3. Merge vector clocks in all cases
    """
    merged_vclock = local.vclock.copy().merge(remote.vclock)

    if local.vclock < remote.vclock:
        # Remote is newer
        winner = remote
    elif remote.vclock < local.vclock:
        # Local is newer
        winner = local
    else:
        # Concurrent — last-writer-wins
        winner = remote if remote.created_at >= local.created_at else local

    winner.vclock = merged_vclock
    # Merge supersedes lists
    winner.supersedes = sorted(set(local.supersedes + remote.supersedes))
    return winner
