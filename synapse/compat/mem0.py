"""Mem0 compatibility layer.

This provides a small surface-area shim so code written for Mem0 can be
ported to Synapse AI Memory with minimal changes.

API mirrored:
  - from synapse.compat import Mem0Client
  - m = Mem0Client()  # wraps Synapse AI Memory internally
  - m.add(text, user_id=None, metadata=None) -> {"id": ...}
  - m.search(query, user_id=None, limit=10) -> [{id, memory, score, metadata, created_at}]
  - m.get_all(user_id=None) -> [...]
  - m.get(memory_id) -> {...} | None
  - m.update(memory_id, text) -> {...} | None
  - m.delete(memory_id) -> bool
  - m.delete_all(user_id=None) -> int
  - m.history(memory_id) -> version history (oldest -> newest)
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Dict, List, Optional, Union

from synapse import Synapse


MemoryId = Union[int, str]


def _coerce_id(memory_id: MemoryId) -> int:
    if isinstance(memory_id, int):
        return memory_id
    if isinstance(memory_id, str):
        s = memory_id.strip()
        # Accept "123" and also tolerate "mem_123" style prefixes if present.
        if s.isdigit():
            return int(s)
        for i, ch in enumerate(s):
            if ch.isdigit():
                tail = s[i:]
                if tail.isdigit():
                    return int(tail)
        raise ValueError(f"Invalid memory_id: {memory_id!r}")
    raise TypeError(f"Invalid memory_id type: {type(memory_id)!r}")


def _user_tag(user_id: Optional[str]) -> Optional[str]:
    if user_id is None:
        return None
    uid = str(user_id).strip()
    if not uid:
        return None
    return f"user:{uid}"


def _ensure_tags(meta: Dict[str, Any], tag: Optional[str]) -> Dict[str, Any]:
    if tag is None:
        return meta
    tags = meta.get("tags")
    if tags is None:
        meta["tags"] = [tag]
        return meta
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        # Don't guess; just overwrite to a sane format.
        meta["tags"] = [tag]
        return meta
    if tag not in tags:
        tags.append(tag)
    meta["tags"] = tags
    return meta


def _mem_to_dict(memory: Any, *, score: float = 0.0) -> Dict[str, Any]:
    # Synapse AI Memory `Memory` is a dataclass-like object with these fields.
    return {
        "id": str(memory.id) if memory.id is not None else None,
        "memory": memory.content,
        "score": float(score),
        "metadata": copy.deepcopy(getattr(memory, "metadata", {}) or {}),
        "created_at": float(getattr(memory, "created_at", 0.0) or 0.0),
    }


class Mem0Client:
    """Mem0-like client that stores and searches via Synapse AI Memory."""

    def __init__(self, *, path: str = ":memory:", synapse: Optional[Synapse] = None):
        self._synapse = synapse or Synapse(path)

    @property
    def synapse(self) -> Synapse:
        return self._synapse

    def close(self) -> None:
        self._synapse.close()

    def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = copy.deepcopy(metadata or {})
        meta = _ensure_tags(meta, _user_tag(user_id))
        mem = self._synapse.remember(text, metadata=meta, deduplicate=False)
        return {"id": str(mem.id)}

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        tag = _user_tag(user_id)
        # Recall can't filter on metadata/tags; over-fetch then filter.
        fetch = max(limit * 10, 50)
        results = self._synapse.recall(query, limit=fetch, explain=True)

        out: List[Dict[str, Any]] = []
        q = (query or "").strip()
        q_lower = q.lower()
        q_tokens = re.findall(r"[a-z0-9]+", q_lower)
        for mem in results:
            meta = getattr(mem, "metadata", {}) or {}
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            if tag is not None and tag not in (tags or []):
                continue
            sb = getattr(mem, "score_breakdown", None)
            score = getattr(sb, "bm25_score", 0.0) if sb is not None else 0.0
            # Mem0's search is primarily lexical. Synapse may surface semantic-only
            # matches; filter those out for deterministic compat behavior.
            if q_tokens:
                content = (getattr(mem, "content", "") or "").lower()
                # Require all query tokens to appear in the content to avoid
                # partial/stem matches like "note" ~ "notebook".
                if not all(t in content for t in q_tokens):
                    continue
                if sb is not None and float(score) <= 0.0:
                    continue
                if sb is None and q_lower not in content:
                    continue
            out.append(_mem_to_dict(mem, score=score))
            if len(out) >= limit:
                break
        return out

    def get_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        tag = _user_tag(user_id)
        memories = []
        for mid, mdata in self._synapse.store.memories.items():
            if mdata.get("consolidated", False):
                continue
            meta = json.loads(mdata.get("metadata", "{}"))
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            if tag is not None and tag not in (tags or []):
                continue
            memories.append(self._synapse._memory_data_to_object(mdata))

        memories.sort(key=lambda m: m.created_at)
        return [_mem_to_dict(m, score=0.0) for m in memories]

    def get(self, memory_id: MemoryId) -> Optional[Dict[str, Any]]:
        mid = _coerce_id(memory_id)
        mdata = self._synapse.store.memories.get(mid)
        if not mdata or mdata.get("consolidated", False):
            return None
        mem = self._synapse._memory_data_to_object(mdata)
        return _mem_to_dict(mem, score=0.0)

    def update(self, memory_id: MemoryId, text: str) -> Optional[Dict[str, Any]]:
        try:
            mid = _coerce_id(memory_id)
        except (TypeError, ValueError):
            return None
        mdata = self._synapse.store.memories.get(mid)
        if not mdata or mdata.get("consolidated", False):
            return None

        now = time.time()
        meta = json.loads(mdata.get("metadata", "{}"))

        # Keep a simple, Mem0-style version history within metadata.
        hist = meta.get("_mem0_history")
        if not isinstance(hist, list):
            hist = []
        hist.append(
            {
                "memory": mdata.get("content", ""),
                "metadata": copy.deepcopy(meta),
                "updated_at": float(now),
            }
        )
        meta["_mem0_history"] = hist
        meta["_mem0_updated_at"] = float(now)

        # Update storage.
        self._synapse.store.update_memory(
            mid, {"content": text, "metadata": json.dumps(meta), "last_accessed": now}
        )
        mdata["content"] = text
        mdata["metadata"] = json.dumps(meta)
        mdata["last_accessed"] = now

        # Re-index. (Synapse doesn't currently expose an official update API.)
        self._synapse.inverted_index.remove_document(mid)
        self._synapse.inverted_index.add_document(mid, text)

        self._synapse.embedding_index.remove(mid)
        if self._synapse._use_embeddings is None:
            self._synapse._use_embeddings = self._synapse.embedding_index.is_available()
        if self._synapse._use_embeddings:
            self._synapse.embedding_index.add(mid, text)

        self._synapse.concept_graph.unlink_memory(mid)
        from entity_graph import extract_concepts as _extract_concepts

        for concept_name, category in _extract_concepts(text):
            self._synapse.concept_graph.link_memory_concept(mid, concept_name, category)

        mem = self._synapse._memory_data_to_object(mdata)
        return _mem_to_dict(mem, score=0.0)

    def delete(self, memory_id: MemoryId) -> bool:
        mid = _coerce_id(memory_id)
        return bool(self._synapse.forget(mid))

    def delete_all(self, user_id: Optional[str] = None) -> int:
        tag = _user_tag(user_id)
        to_delete: List[int] = []
        for mid, mdata in self._synapse.store.memories.items():
            if mdata.get("consolidated", False):
                continue
            if tag is None:
                to_delete.append(mid)
                continue
            meta = json.loads(mdata.get("metadata", "{}"))
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            if tag in (tags or []):
                to_delete.append(mid)

        deleted = 0
        for mid in list(to_delete):
            if self._synapse.forget(mid):
                deleted += 1
        return deleted

    def history(self, memory_id: MemoryId) -> List[Dict[str, Any]]:
        mid = _coerce_id(memory_id)
        mdata = self._synapse.store.memories.get(mid)
        if not mdata or mdata.get("consolidated", False):
            return []

        meta = json.loads(mdata.get("metadata", "{}"))
        hist = meta.get("_mem0_history")
        out: List[Dict[str, Any]] = []
        if isinstance(hist, list):
            for i, entry in enumerate(hist, start=1):
                out.append(
                    {
                        "id": str(mid),
                        "version": i,
                        "current": False,
                        "memory": entry.get("memory", ""),
                        "metadata": entry.get("metadata", {}),
                        "created_at": float(entry.get("updated_at", 0.0) or 0.0),
                    }
                )

        # Append current version as the last entry.
        mem = self._synapse._memory_data_to_object(mdata)
        out.append(
            {
                "id": str(mid),
                "version": len(out) + 1,
                "current": True,
                "memory": mem.content,
                "metadata": copy.deepcopy(mem.metadata),
                "created_at": float(getattr(mem, "created_at", 0.0) or 0.0),
            }
        )
        return out
