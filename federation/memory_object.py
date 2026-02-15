"""Federated memory object — the unit of sync between nodes."""

from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional

from federation.content_hash import memory_hash, edge_hash
from federation.vector_clock import VectorClock


class FederatedMemory:
    """
    A memory that can be synced between nodes.
    Content-addressed (hash is its identity), with vector clock for causality.
    """

    __slots__ = (
        "hash", "content", "memory_type", "metadata",
        "created_at", "origin_node", "namespaces",
        "vclock", "supersedes",
    )

    def __init__(
        self,
        content: str,
        memory_type: str = "fact",
        metadata: Dict[str, Any] | None = None,
        created_at: float | None = None,
        origin_node: str = "",
        namespaces: List[str] | None = None,
        vclock: VectorClock | None = None,
        supersedes: List[str] | None = None,
    ):
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()
        self.origin_node = origin_node
        self.namespaces = namespaces or []
        self.vclock = vclock or VectorClock()
        self.supersedes = supersedes or []  # list of hashes this supersedes
        self.hash = memory_hash(content, memory_type, self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "content": self.content,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "origin_node": self.origin_node,
            "namespaces": self.namespaces,
            "vclock": self.vclock.to_dict(),
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FederatedMemory":
        fm = cls(
            content=d["content"],
            memory_type=d["memory_type"],
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", 0.0),
            origin_node=d.get("origin_node", ""),
            namespaces=d.get("namespaces", []),
            vclock=VectorClock.from_dict(d.get("vclock", {})),
            supersedes=d.get("supersedes", []),
        )
        # Verify hash matches
        expected = memory_hash(fm.content, fm.memory_type, fm.metadata)
        if d.get("hash") and d["hash"] != expected:
            raise ValueError(f"Hash mismatch: expected {expected}, got {d['hash']}")
        return fm

    def __repr__(self):
        return f"FederatedMemory({self.hash[:12]}..., {self.content[:40]!r})"


class FederatedEdge:
    """An edge between two federated memories, also content-addressed."""

    __slots__ = ("hash", "source_hash", "target_hash", "edge_type", "weight",
                 "created_at", "origin_node", "vclock")

    def __init__(
        self,
        source_hash: str,
        target_hash: str,
        edge_type: str,
        weight: float = 1.0,
        created_at: float | None = None,
        origin_node: str = "",
        vclock: VectorClock | None = None,
    ):
        self.source_hash = source_hash
        self.target_hash = target_hash
        self.edge_type = edge_type
        self.weight = weight
        self.created_at = created_at or time.time()
        self.origin_node = origin_node
        self.vclock = vclock or VectorClock()
        self.hash = edge_hash(source_hash, target_hash, edge_type, weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "created_at": self.created_at,
            "origin_node": self.origin_node,
            "vclock": self.vclock.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FederatedEdge":
        return cls(
            source_hash=d["source_hash"],
            target_hash=d["target_hash"],
            edge_type=d["edge_type"],
            weight=d.get("weight", 1.0),
            created_at=d.get("created_at", 0.0),
            origin_node=d.get("origin_node", ""),
            vclock=VectorClock.from_dict(d.get("vclock", {})),
        )
