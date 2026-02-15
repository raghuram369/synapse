"""Content-addressed hashing for memories — like git object IDs."""

import hashlib
import json
from typing import Any, Dict


def memory_hash(content: str, memory_type: str, metadata: Dict[str, Any] | None = None) -> str:
    """
    Deterministic content hash for a memory.
    Like a git blob hash: hash(type + content + sorted metadata).
    Returns hex SHA-256.
    """
    canonical = json.dumps({
        "content": content,
        "memory_type": memory_type,
        "metadata": _normalize(metadata or {}),
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def edge_hash(source_hash: str, target_hash: str, edge_type: str, weight: float) -> str:
    """Deterministic hash for an edge between two memories."""
    canonical = json.dumps({
        "source": source_hash,
        "target": target_hash,
        "edge_type": edge_type,
        "weight": round(weight, 6),
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize(obj: Any) -> Any:
    """Recursively sort dicts for deterministic serialization."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj
