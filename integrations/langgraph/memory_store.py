"""
SynapseStore — LangGraph cross-thread memory store backed by Synapse.

Enables agents to share memories across threads and conversations.
Privacy-first: all data stays local. Federation for multi-agent systems.

Requirements:
    pip install langgraph synapse-ai-memory
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from synapse import Synapse


class SynapseStore:
    """Cross-thread memory store for LangGraph using Synapse.

    Implements LangGraph's BaseStore interface for shared memory across
    threads. Memories are namespaced and semantically searchable.

    Example::

        from synapse import Synapse
        from synapse.integrations.langgraph import SynapseStore

        syn = Synapse("./shared_memory")
        store = SynapseStore(synapse=syn)

        # Store shared context
        store.put(("user", "preferences"), "diet", {"value": "vegetarian"})

        # Retrieve across any thread
        item = store.get(("user", "preferences"), "diet")

        # Semantic search across all stored items
        results = store.search(("user",), query="food preferences")
    """

    _base_resolved = False

    def __new__(cls, *args, **kwargs):
        if not cls._base_resolved:
            try:
                from langgraph.store.base import BaseStore
                cls.__bases__ = (BaseStore,) + tuple(
                    b for b in cls.__bases__ if b is not object
                )
                cls._base_resolved = True
            except ImportError:
                pass
        return super().__new__(cls)

    def __init__(
        self,
        synapse: Optional[Synapse] = None,
        path: str = ":memory:",
    ):
        self.synapse = synapse or Synapse(path)
        try:
            super().__init__()
        except TypeError:
            pass

    def _namespace_key(self, namespace: Tuple[str, ...]) -> str:
        """Convert namespace tuple to a string key."""
        return "/".join(namespace)

    def _full_key(self, namespace: Tuple[str, ...], key: str) -> str:
        return f"{self._namespace_key(namespace)}/{key}"

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any],
            index: Optional[List[str]] = None) -> None:
        """Store or update an item in the given namespace."""
        full_key = self._full_key(namespace, key)
        ns_str = self._namespace_key(namespace)

        # Check if this key already exists (update case)
        existing = self._find_by_key(namespace, key)
        if existing:
            self.synapse.forget(existing.id)

        content = json.dumps({"key": key, "value": value, "namespace": ns_str}, default=str)

        metadata = {
            "source": "langgraph_store",
            "namespace": ns_str,
            "item_key": key,
            "full_key": full_key,
            "updated_at": time.time(),
        }
        if index:
            metadata["index_fields"] = json.dumps(index)

        self.synapse.remember(
            content,
            memory_type="fact",
            metadata=metadata,
            deduplicate=False,
        )

    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
        """Get a single item by namespace and key."""
        mem = self._find_by_key(namespace, key)
        if not mem:
            return None

        try:
            from langgraph.store.base import Item
            data = json.loads(mem.content)
            return Item(
                value=data.get("value", {}),
                key=key,
                namespace=namespace,
                created_at=mem.created_at,
                updated_at=(mem.metadata or {}).get("updated_at", mem.created_at),
            )
        except ImportError:
            data = json.loads(mem.content)
            return data.get("value")

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete an item."""
        mem = self._find_by_key(namespace, key)
        if mem:
            self.synapse.forget(mem.id)

    def list_namespaces(self, *, prefix: Optional[Tuple[str, ...]] = None,
                        suffix: Optional[Tuple[str, ...]] = None,
                        max_depth: Optional[int] = None,
                        limit: int = 100, offset: int = 0) -> List[Tuple[str, ...]]:
        """List all namespaces, optionally filtered by prefix."""
        all_memories = self.synapse.recall("", limit=10000)
        namespaces = set()

        for mem in all_memories:
            meta = mem.metadata or {}
            if meta.get("source") == "langgraph_store":
                ns = meta.get("namespace", "")
                ns_tuple = tuple(ns.split("/")) if ns else ()
                
                if prefix and not ns_tuple[:len(prefix)] == prefix:
                    continue
                if suffix and not ns_tuple[-len(suffix):] == suffix:
                    continue
                if max_depth and len(ns_tuple) > max_depth:
                    ns_tuple = ns_tuple[:max_depth]

                namespaces.add(ns_tuple)

        result = sorted(namespaces)
        return result[offset:offset + limit]

    def search(self, namespace_prefix: Tuple[str, ...], *,
               query: Optional[str] = None, filter: Optional[Dict[str, Any]] = None,
               limit: int = 10, offset: int = 0) -> List[Any]:
        """Semantic search across items in a namespace.

        This is Synapse's superpower — semantic search across stored items,
        not just key-value lookups.
        """
        ns_prefix = self._namespace_key(namespace_prefix)

        if query:
            memories = self.synapse.recall(query, limit=limit * 3)
        else:
            memories = self.synapse.recall("", limit=limit * 3)

        # Filter to namespace and source
        results = []
        for mem in memories:
            meta = mem.metadata or {}
            if meta.get("source") != "langgraph_store":
                continue
            ns = meta.get("namespace", "")
            if not ns.startswith(ns_prefix):
                continue

            # Apply metadata filter
            if filter:
                data = json.loads(mem.content)
                value = data.get("value", {})
                if not all(value.get(k) == v for k, v in filter.items()):
                    continue

            try:
                from langgraph.store.base import Item
                data = json.loads(mem.content)
                results.append(Item(
                    value=data.get("value", {}),
                    key=meta.get("item_key", ""),
                    namespace=tuple(ns.split("/")),
                    created_at=mem.created_at,
                    updated_at=meta.get("updated_at", mem.created_at),
                ))
            except ImportError:
                data = json.loads(mem.content)
                results.append(data.get("value", {}))

        return results[offset:offset + limit]

    def _find_by_key(self, namespace: Tuple[str, ...], key: str):
        """Find a memory by namespace and key."""
        full_key = self._full_key(namespace, key)
        all_memories = self.synapse.recall("", limit=10000)
        for mem in all_memories:
            meta = mem.metadata or {}
            if (meta.get("source") == "langgraph_store" and
                    meta.get("full_key") == full_key):
                return mem
        return None

    # Batch operations
    def batch(self, ops: Sequence[Any]) -> List[Any]:
        """Execute a batch of operations."""
        results = []
        for op in ops:
            if hasattr(op, 'namespace') and hasattr(op, 'key'):
                # GetOp
                results.append(self.get(op.namespace, op.key))
            else:
                results.append(None)
        return results
