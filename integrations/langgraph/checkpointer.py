"""
SynapseCheckpointer — LangGraph checkpoint persistence backed by Synapse.

Saves and restores graph state between invocations, enabling
long-running agents with persistent state. All data stays local.

Requirements:
    pip install langgraph synapse-ai-memory
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from synapse import Synapse


class SynapseCheckpointer:
    """LangGraph-compatible checkpointer using Synapse for persistence.

    Stores graph state snapshots as Synapse memories with structured metadata,
    enabling state recovery across process restarts.

    Example::

        from synapse import Synapse
        from synapse.integrations.langgraph import SynapseCheckpointer

        syn = Synapse("./agent_state")
        checkpointer = SynapseCheckpointer(synapse=syn)

        # Use with LangGraph
        graph = builder.compile(checkpointer=checkpointer)
        graph.invoke({"input": "hello"}, config={"configurable": {"thread_id": "t1"}})
    """

    _base_resolved = False

    def __new__(cls, *args, **kwargs):
        if not cls._base_resolved:
            try:
                from langgraph.checkpoint.base import BaseCheckpointSaver
                cls.__bases__ = (BaseCheckpointSaver,) + tuple(
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
        # Index: thread_id -> list of (checkpoint_id, memory_id)
        self._index: Dict[str, list] = {}
        self._rebuild_index()

        try:
            super().__init__()
        except TypeError:
            pass

    def _rebuild_index(self):
        """Rebuild checkpoint index from stored memories."""
        all_memories = self.synapse.recall("", limit=10000)
        for mem in all_memories:
            meta = mem.metadata or {}
            if meta.get("source") == "langgraph_checkpoint":
                thread_id = meta.get("thread_id", "default")
                checkpoint_id = meta.get("checkpoint_id", "")
                self._index.setdefault(thread_id, []).append(
                    (checkpoint_id, mem.id, mem.created_at)
                )

        # Sort each thread's checkpoints by time
        for thread_id in self._index:
            self._index[thread_id].sort(key=lambda x: x[2])

    def _make_checkpoint_id(self, thread_id: str) -> str:
        """Generate a monotonic checkpoint ID."""
        count = len(self._index.get(thread_id, []))
        return f"{thread_id}:{count}:{time.time()}"

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None, new_versions: Optional[Any] = None) -> Dict[str, Any]:
        """Save a checkpoint."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        checkpoint_id = self._make_checkpoint_id(thread_id)

        # Serialize checkpoint as JSON content
        content = json.dumps(checkpoint, default=str)

        mem = self.synapse.remember(
            content,
            memory_type="fact",
            metadata={
                "source": "langgraph_checkpoint",
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "checkpoint_metadata": json.dumps(metadata or {}, default=str),
            },
            deduplicate=False,  # Each checkpoint is unique
        )

        self._index.setdefault(thread_id, []).append(
            (checkpoint_id, mem.id, mem.created_at)
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: Dict[str, Any], writes: Sequence[Tuple[str, Any]],
                   task_id: str) -> None:
        """Save intermediate writes (pending sends)."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get("checkpoint_id", "")

        content = json.dumps({"writes": writes, "task_id": task_id}, default=str)
        self.synapse.remember(
            content,
            memory_type="fact",
            metadata={
                "source": "langgraph_writes",
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
            },
            deduplicate=False,
        )

    def get_tuple(self, config: Dict[str, Any]) -> Optional[Any]:
        """Get the latest checkpoint for a thread."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get("checkpoint_id")

        entries = self._index.get(thread_id, [])
        if not entries:
            return None

        # Find specific or latest checkpoint
        if checkpoint_id:
            target = next((e for e in entries if e[0] == checkpoint_id), None)
        else:
            target = entries[-1]  # Latest

        if not target:
            return None

        cp_id, mem_id, _ = target

        # Load the memory
        memories = self.synapse.recall("", limit=10000)
        mem = next((m for m in memories if m.id == mem_id), None)
        if not mem:
            return None

        checkpoint = json.loads(mem.content)
        meta = mem.metadata or {}
        checkpoint_metadata = json.loads(meta.get("checkpoint_metadata", "{}"))

        try:
            from langgraph.checkpoint.base import CheckpointTuple
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": meta.get("checkpoint_ns", ""),
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=checkpoint_metadata,
                parent_config=None,
                pending_writes=[],
            )
        except ImportError:
            return {
                "config": config,
                "checkpoint": checkpoint,
                "metadata": checkpoint_metadata,
            }

    def list(self, config: Optional[Dict[str, Any]] = None,
             *, filter: Optional[Dict[str, Any]] = None,
             before: Optional[Dict[str, Any]] = None,
             limit: Optional[int] = None) -> Iterator[Any]:
        """List checkpoints for a thread."""
        if config:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "default")
            entries = self._index.get(thread_id, [])
        else:
            entries = []
            for thread_entries in self._index.values():
                entries.extend(thread_entries)

        # Sort newest first
        entries.sort(key=lambda x: x[2], reverse=True)

        if limit:
            entries = entries[:limit]

        for cp_id, mem_id, created_at in entries:
            result = self.get_tuple({
                "configurable": {"thread_id": thread_id if config else "default", "checkpoint_id": cp_id}
            })
            if result:
                yield result
