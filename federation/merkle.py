"""Merkle tree for efficient delta detection between memory stores."""

import hashlib
from typing import Dict, List, Set, Tuple


class MerkleTree:
    """
    A sorted-hash Merkle tree over a set of content-hash IDs.
    
    Rather than a full binary tree, we use a simple fan-out scheme:
    - Leaves are individual memory hashes
    - Leaves are sorted and split into 256 buckets by first byte
    - Each bucket gets a hash of its sorted members
    - Root = hash of all 256 bucket hashes
    
    This gives O(1) root comparison and O(n/256) per-bucket delta.
    """

    __slots__ = ("_items", "_buckets", "_bucket_hashes", "_root")

    def __init__(self):
        self._items: Set[str] = set()
        self._buckets: Dict[int, List[str]] = {}
        self._bucket_hashes: Dict[int, str] = {}
        self._root: str | None = None

    # ── Mutation ──────────────────────────────────────────────

    def add(self, content_hash: str):
        """Add a content hash. Invalidates cached root."""
        if content_hash not in self._items:
            self._items.add(content_hash)
            self._root = None  # invalidate

    def remove(self, content_hash: str):
        if content_hash in self._items:
            self._items.discard(content_hash)
            self._root = None

    def add_many(self, hashes):
        before = len(self._items)
        self._items.update(hashes)
        if len(self._items) != before:
            self._root = None

    # ── Tree computation ─────────────────────────────────────

    def _rebuild(self):
        """Recompute buckets and root."""
        self._buckets = {}
        for h in self._items:
            # Use first 2 hex chars if valid, otherwise hash to get a bucket
            try:
                bucket = int(h[:2], 16)
            except ValueError:
                bucket = hash(h) % 256
            self._buckets.setdefault(bucket, []).append(h)
        
        # Sort each bucket for determinism
        self._bucket_hashes = {}
        for b in range(256):
            members = self._buckets.get(b, [])
            members.sort()
            payload = "\n".join(members).encode("utf-8")
            self._bucket_hashes[b] = hashlib.sha256(payload).hexdigest()
        
        # Root = hash of all bucket hashes in order
        all_bucket_data = "".join(self._bucket_hashes[b] for b in range(256))
        self._root = hashlib.sha256(all_bucket_data.encode("utf-8")).hexdigest()

    @property
    def root(self) -> str:
        if self._root is None:
            self._rebuild()
        return self._root

    @property
    def bucket_hashes(self) -> Dict[int, str]:
        if self._root is None:
            self._rebuild()
        return dict(self._bucket_hashes)

    def items(self) -> Set[str]:
        return set(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, h: str) -> bool:
        return h in self._items

    # ── Delta computation ─────────────────────────────────────

    def diff_buckets(self, other_bucket_hashes: Dict[int, str]) -> List[int]:
        """Return bucket indices that differ between self and other."""
        if self._root is None:
            self._rebuild()
        differing = []
        for b in range(256):
            if self._bucket_hashes.get(b) != other_bucket_hashes.get(b):
                differing.append(b)
        return differing

    def items_in_buckets(self, bucket_ids: List[int]) -> Set[str]:
        """Return all hashes in the given buckets."""
        if self._root is None:
            self._rebuild()
        result = set()
        for b in bucket_ids:
            result.update(self._buckets.get(b, []))
        return result

    def missing_from(self, other_items: Set[str]) -> Set[str]:
        """Return items in self that are NOT in other_items."""
        return self._items - other_items

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "bucket_hashes": {str(k): v for k, v in self.bucket_hashes.items()},
            "count": len(self._items),
        }
