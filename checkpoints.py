"""Checkpoint manager for Synapse memory snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contradictions import ContradictionDetector


_NON_SLUG_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass
class Checkpoint:
    """A named snapshot of the full memory state."""

    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    checksum: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    snapshot_path: str = ""


class CheckpointManager:
    """Manage named checkpoint creation, inspection, restore, and diff."""

    _SNAPSHOT_BASENAME = "state.snapshot"
    _LOG_BASENAME = "state.log"
    _METADATA_BASENAME = "metadata.json"

    def __init__(self, synapse, checkpoint_dir: str = None):
        self.synapse = synapse
        if checkpoint_dir is None:
            base_path = getattr(synapse, "_store", None)
            if base_path is None:
                raise ValueError("synapse must provide _store with base_path")
            root = getattr(base_path, "base_path", None)
            if root in (None, "", ":memory:"):
                checkpoint_dir = ""
            else:
                checkpoint_dir = os.path.join(root, "checkpoints")

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._auto_checkpoint_every: Optional[int] = None
        self._auto_checkpoint_next: Optional[int] = None

    @property
    def supported(self) -> bool:
        return not getattr(self.synapse, "path", None) == ":memory:"

    def _slug(self, name: str) -> str:
        safe = _NON_SLUG_CHARS.sub("_", (name or "").strip())
        safe = safe.strip("._-")
        if not safe:
            safe = f"checkpoint_{int(time.time())}"
        return safe

    def _checkpoint_dir_for_name(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, self._slug(name))

    def _metadata_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, self._METADATA_BASENAME)

    def _snapshot_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, self._SNAPSHOT_BASENAME)

    def _log_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, self._LOG_BASENAME)

    @staticmethod
    def _normalize_payload(value: Any):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {str(key): CheckpointManager._normalize_payload(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return [CheckpointManager._normalize_payload(v) for v in value]
        if isinstance(value, list):
            return [CheckpointManager._normalize_payload(v) for v in value]
        return value

    def _serialize_for_checksum(self, payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _active_memory_objects(self):
        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get('consolidated', False):
                continue
            yield self.synapse._memory_data_to_object(memory_data)

    def _snapshot_data(self) -> Dict[str, Any]:
        detector = ContradictionDetector()
        active_memories = list(self._active_memory_objects())

        # Rebuild snapshots directly from current store state to avoid any index bias.
        memory_map = {
            str(memory_id): self._normalize_payload(memory_data)
            for memory_id, memory_data in self.synapse.store.memories.items()
        }
        edge_map = {
            str(edge_id): self._normalize_payload(edge_data)
            for edge_id, edge_data in self.synapse.store.edges.items()
        }
        episode_map = {
            str(episode_id): self._normalize_payload(episode_data)
            for episode_id, episode_data in self.synapse.store.episodes.items()
        }
        card_map = {
            str(card_id): self._normalize_payload(card_data)
            for card_id, card_data in getattr(self.synapse.store, "cards", {}).items()
        }

        concepts = set()
        for memory in active_memories:
            for concept_name in self.synapse.concept_graph.get_memory_concepts(memory.id):
                concepts.add(concept_name)

        beliefs = self.synapse.beliefs() or {}
        belief_map = {
            key: {
                "fact": key,
                "value": version.value,
                "memory_id": version.memory_id,
                "valid_from": version.valid_from,
                "valid_to": version.valid_to,
                "reason": version.reason,
                "confidence": version.confidence,
            }
            for key, version in beliefs.items()
        }

        contradictions = [
            {
                "memory_id_a": int(contradiction.memory_id_a),
                "memory_id_b": int(contradiction.memory_id_b),
                "kind": str(contradiction.kind),
                "description": str(contradiction.description),
                "confidence": float(contradiction.confidence),
            }
            for contradiction in detector.scan_memories(active_memories)
        ]

        active_memory_count = len([m for m in memory_map.values() if not m.get("consolidated", False)])
        concept_count = len(concepts)
        stats = {
            "memory_count": len(memory_map),
            "active_memory_count": active_memory_count,
            "concept_count": concept_count,
            "edge_count": len(edge_map),
            "belief_count": len(belief_map),
            "contradiction_count": len(contradictions),
            "card_count": len(card_map),
            "next_memory_id": self.synapse.store.next_memory_id,
            "next_episode_id": self.synapse.store.next_episode_id,
        }

        return {
            "memories": memory_map,
            "edges": edge_map,
            "episodes": episode_map,
            "cards": card_map,
            "beliefs": belief_map,
            "contradictions": contradictions,
            "concepts": sorted(concepts),
            "stats": stats,
        }

    def _collect_concepts_from_state(self, state: Dict[str, Any]) -> List[str]:
        concepts = state.get("concepts", [])
        if isinstance(concepts, set):
            return sorted(concepts)
        if isinstance(concepts, list):
            return sorted(str(item) for item in concepts)
        return []

    def _load_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _find_checkpoint(self, name: str) -> Optional[str]:
        if not self.checkpoint_dir:
            return None

        for entry in sorted(os.listdir(self.checkpoint_dir)):
            candidate = os.path.join(self.checkpoint_dir, entry)
            if not os.path.isdir(candidate):
                continue
            metadata = self._load_metadata(self._metadata_path(candidate))
            if not isinstance(metadata, dict):
                continue
            checkpoint_name = metadata.get("name", metadata.get("checkpoint_name", ""))
            if checkpoint_name == name:
                return candidate
        return None

    def _read_checkpoint_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        checkpoint_dir = self._find_checkpoint(name)
        if checkpoint_dir is None:
            return None
        return self._load_metadata(self._metadata_path(checkpoint_dir))

    def create(self, name: str, description: str = "") -> Checkpoint:
        """Create a named checkpoint of current state."""
        if not self.supported:
            raise RuntimeError("Checkpoints require a persistent Synapse store")
        if not self.checkpoint_dir:
            raise RuntimeError("Checkpoint storage directory is not configured")
        if self._find_checkpoint(name):
            raise ValueError(f"checkpoint already exists: {name}")

        snapshot_dir = self._checkpoint_dir_for_name(name)
        if os.path.exists(snapshot_dir):
            raise ValueError(f"checkpoint path collision: {name}")
        os.makedirs(snapshot_dir, exist_ok=False)

        source_snapshot = self.synapse.store.snapshot_path
        source_log = self.synapse.store.log_path
        snapshot_copy = self._snapshot_path(snapshot_dir)
        log_copy = self._log_path(snapshot_dir)

        if os.path.exists(source_snapshot):
            shutil.copy2(source_snapshot, snapshot_copy)
        else:
            open(snapshot_copy, "w", encoding="utf-8").close()
        if os.path.exists(source_log):
            shutil.copy2(source_log, log_copy)
        else:
            open(log_copy, "w", encoding="utf-8").close()

        state = self._snapshot_data()
        created_at = time.time()
        checkpoint = Checkpoint(
            name=name,
            description=description,
            created_at=created_at,
            stats=state["stats"].copy(),
            snapshot_path=snapshot_copy,
        )
        checksum_payload = {
            "name": checkpoint.name,
            "description": checkpoint.description,
            "created_at": checkpoint.created_at,
            "state": state,
        }
        checkpoint.checksum = self._serialize_for_checksum(self._normalize_payload(checksum_payload))
        checkpoint_data = {
            "checkpoint_name": checkpoint.name,
            "checkpoint_slug": os.path.basename(snapshot_dir),
            "checkpoint_description": checkpoint.description,
            "created_at": checkpoint.created_at,
            "checksum": checkpoint.checksum,
            "snapshot_path": checkpoint.snapshot_path,
            "log_path": log_copy,
            "state": state,
            "stats": checkpoint.stats,
            "synapse_path": getattr(self.synapse, "path", ""),
        }
        with open(self._metadata_path(snapshot_dir), "w", encoding="utf-8") as fp:
            json.dump(checkpoint_data, fp, ensure_ascii=False, indent=2, sort_keys=True)

        return checkpoint

    def list(self) -> List[Checkpoint]:
        """List all checkpoints, newest first."""
        if not self.checkpoint_dir:
            return []

        checkpoints: List[Checkpoint] = []
        for entry in sorted(os.listdir(self.checkpoint_dir)):
            checkpoint_dir = os.path.join(self.checkpoint_dir, entry)
            if not os.path.isdir(checkpoint_dir):
                continue
            metadata = self._load_metadata(self._metadata_path(checkpoint_dir))
            if not isinstance(metadata, dict):
                continue
            created_at = float(metadata.get("created_at", 0.0))
            checkpoint = Checkpoint(
                name=metadata.get("checkpoint_name", entry),
                description=str(metadata.get("checkpoint_description", "")),
                created_at=created_at,
                checksum=str(metadata.get("checksum", "")),
                stats=self._normalize_payload(metadata.get("stats", {})),
                snapshot_path=metadata.get("snapshot_path", self._snapshot_path(checkpoint_dir)),
            )
            checkpoints.append(checkpoint)

        checkpoints.sort(key=lambda item: item.created_at, reverse=True)
        return checkpoints

    def restore(self, name: str) -> Dict[str, Any]:
        """Restore state from a checkpoint."""
        if not self.supported:
            raise RuntimeError("Checkpoints require a persistent Synapse store")
        metadata = self._read_checkpoint_metadata(name)
        if metadata is None:
            raise ValueError(f"checkpoint not found: {name}")

        checkpoint_dir = self._find_checkpoint(name)
        if checkpoint_dir is None:
            raise ValueError(f"checkpoint not found: {name}")
        source_snapshot = self._snapshot_path(checkpoint_dir)
        source_log = self._log_path(checkpoint_dir)

        if not os.path.exists(source_snapshot):
            raise FileNotFoundError(f"checkpoint snapshot missing: {source_snapshot}")
        if not os.path.exists(source_log):
            open(source_log, "w", encoding="utf-8").close()

        shutil.copy2(source_snapshot, self.synapse.store.snapshot_path)
        shutil.copy2(source_log, self.synapse.store.log_path)

        previous_count = self.synapse.count()
        self.synapse._reload_from_checkpoint_state()

        restored_state = self._normalize_payload(metadata.get("state", {}))
        stats = restored_state.get("stats", {})
        restored = {
            "checkpoint": name,
            "created_at": float(metadata.get("created_at", 0.0)),
            "checksum": str(metadata.get("checksum", "")),
            "memories_restored": len(restored_state.get("memories", {})),
            "concepts_restored": len(restored_state.get("concepts", [])),
            "edges_restored": len(restored_state.get("edges", {})),
            "beliefs_restored": len(restored_state.get("beliefs", {})),
            "cards_restored": len(restored_state.get("cards", {})),
            "memories_before": previous_count,
            "memories_after": stats.get("active_memory_count", len(restored_state.get("memories", {}))),
        }
        return restored

    def delete(self, name: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_dir = self._find_checkpoint(name)
        if checkpoint_dir is None:
            return False
        shutil.rmtree(checkpoint_dir)
        return True

    def diff(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """Compare two checkpoints by name."""
        state_a = self._read_checkpoint_metadata(name_a)
        if state_a is None:
            raise ValueError(f"checkpoint not found: {name_a}")
        state_b = self._read_checkpoint_metadata(name_b)
        if state_b is None:
            raise ValueError(f"checkpoint not found: {name_b}")

        snapshot_a = self._normalize_payload(state_a.get("state", {}))
        snapshot_b = self._normalize_payload(state_b.get("state", {}))

        memories_a = snapshot_a.get("memories", {})
        memories_b = snapshot_b.get("memories", {})
        beliefs_a = snapshot_a.get("beliefs", {})
        beliefs_b = snapshot_b.get("beliefs", {})
        concepts_a = set(self._collect_concepts_from_state(snapshot_a))
        concepts_b = set(self._collect_concepts_from_state(snapshot_b))

        added_ids = sorted(set(memories_b.keys()) - set(memories_a.keys()))
        removed_ids = sorted(set(memories_a.keys()) - set(memories_b.keys()))

        belief_changed: List[Dict[str, Any]] = []
        for fact in set(beliefs_a.keys()) & set(beliefs_b.keys()):
            old_val = beliefs_a[fact].get("value")
            new_val = beliefs_b[fact].get("value")
            if old_val != new_val:
                belief_changed.append({
                    "fact": fact,
                    "old_value": old_val,
                    "new_value": new_val,
                })

        contradictions_a = snapshot_a.get("contradictions", [])
        contradictions_b = snapshot_b.get("contradictions", [])
        contradiction_map_a = {
            (
                int(item.get("memory_id_a")),
                int(item.get("memory_id_b")),
                str(item.get("kind")),
            ): item
            for item in contradictions_a
        }
        contradiction_map_b = {
            (
                int(item.get("memory_id_a")),
                int(item.get("memory_id_b")),
                str(item.get("kind")),
            ): item
            for item in contradictions_b
        }

        introduced = []
        for key in sorted(contradiction_map_b.keys()):
            if key not in contradiction_map_a:
                introduced.append(contradiction_map_b[key])

        resolved = []
        for key in sorted(contradiction_map_a.keys()):
            if key not in contradiction_map_b:
                resolved.append(contradiction_map_a[key])

        stats_a = self._normalize_payload(snapshot_a.get("stats", {}))
        stats_b = self._normalize_payload(snapshot_b.get("stats", {}))
        delta_keys = set(stats_a.keys()) | set(stats_b.keys())
        stats_diff = {
            key: stats_b.get(key, 0) - stats_a.get(key, 0)
            for key in sorted(delta_keys)
        }

        return {
            "checkpoint_a": state_a.get("checkpoint_name", name_a),
            "checkpoint_b": state_b.get("checkpoint_name", name_b),
            "memories_added": added_ids,
            "memories_removed": removed_ids,
            "beliefs_changed": belief_changed,
            "contradictions_introduced": introduced,
            "contradictions_resolved": resolved,
            "concepts_added": sorted(concepts_b - concepts_a),
            "concepts_removed": sorted(concepts_a - concepts_b),
            "stats_diff": stats_diff,
        }

    def auto_checkpoint(self, every_n_memories: int = 100):
        """Set automatic checkpoint interval in memory-count steps."""
        every_n_memories = int(every_n_memories)
        if every_n_memories <= 0:
            raise ValueError("every_n_memories must be greater than zero")
        if not self.supported:
            return

        self._auto_checkpoint_every = every_n_memories
        self._auto_checkpoint_next = self.synapse.count() + every_n_memories

    def checkpoint_before_sleep(self) -> Optional[Checkpoint]:
        """Create a checkpoint if auto threshold is reached."""
        if self._auto_checkpoint_every is None:
            return None
        if not self.supported:
            return None
        if self._auto_checkpoint_next is None:
            self._auto_checkpoint_next = self.synapse.count() + self._auto_checkpoint_every

        current_count = self.synapse.count()
        if current_count < self._auto_checkpoint_next:
            return None

        name = f"auto-checkpoint-{int(time.time())}-{current_count}"
        description = (
            f"Automatic checkpoint created before sleep at {current_count} memories"
        )
        checkpoint = self.create(name=name, description=description)

        while self._auto_checkpoint_next is not None and current_count >= self._auto_checkpoint_next:
            self._auto_checkpoint_next += self._auto_checkpoint_every
        return checkpoint
