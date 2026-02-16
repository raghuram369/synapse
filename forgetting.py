"""Forgetting and retention policies for Synapse memories."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set

from entity_graph import extract_concepts

if TYPE_CHECKING:
    from synapse import Synapse


SECONDS_PER_DAY = 86400.0


class ForgettingPolicy:
    def __init__(self, synapse):
        self.synapse = synapse

    def apply_ttl(self, default_ttl_days: int = 365):
        """Delete memories older than TTL.

        Memories may override with metadata['ttl_days'].
        """
        now = time.time()
        default_ttl = self._coerce_positive_float(default_ttl_days, 365.0)
        to_delete = []

        for memory_id, memory_data in list(self.synapse.store.memories.items()):
            created_at = float(memory_data.get("created_at", 0.0) or 0.0)
            metadata = self._get_metadata(memory_data)
            ttl_days = metadata.get("ttl_days")
            ttl = self._coerce_positive_float(ttl_days, default_ttl)

            if ttl is None:
                continue
            if now - created_at > ttl * SECONDS_PER_DAY:
                to_delete.append(memory_id)

        deleted = self._delete_memory_ids(to_delete)

        return {
            "action": "apply_ttl",
            "default_ttl_days": default_ttl,
            "matched_count": len(set(to_delete)),
            "deleted_count": len(deleted),
            "deleted_ids": deleted,
        }

    def forget_topic(self, topic: str):
        """Forget all memories related to a topic/concept.

        Cascades to indexes, triples, beliefs, and contradictions through the
        Synapse forget() path.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")

        topic_norm = topic.strip().lower()
        related_ids = set()

        # Fast path: concept graph lookup by extracted concept names.
        for concept_name, _ in extract_concepts(topic_norm):
            concept_node = self.synapse.concept_graph.concepts.get(concept_name)
            if concept_node is None:
                continue
            related_ids.update(concept_node.memory_ids)

        # Fallback: direct content containment.
        for memory_id, memory_data in self.synapse.store.memories.items():
            content = (memory_data.get("content") or "").lower()
            if topic_norm in content:
                related_ids.add(memory_id)

        matched = sorted(related_ids)
        deleted = self._delete_memory_ids(matched)

        return {
            "action": "forget_topic",
            "topic": topic,
            "matched_count": len(matched),
            "deleted_count": len(deleted),
            "deleted_ids": deleted,
        }

    def redact(self, memory_id: int, fields: List[str] = None):
        """Redact specific content fields while preserving metadata and links."""
        if not isinstance(memory_id, int) or memory_id <= 0:
            return {"memory_id": memory_id, "redacted": False, "reason": "invalid_memory_id"}

        memory_data = self.synapse.store.memories.get(memory_id)
        if memory_data is None:
            return {"memory_id": memory_id, "redacted": False, "reason": "not_found"}

        if fields is None:
            fields = ["content"]
        if not isinstance(fields, list) or not fields:
            return {"memory_id": memory_id, "redacted": False, "reason": "invalid_fields"}

        cleaned_fields: Set[str] = {
            str(field).strip() for field in fields if isinstance(field, str) and str(field).strip()
        }
        if not cleaned_fields:
            return {"memory_id": memory_id, "redacted": False, "reason": "no_valid_fields"}

        updates: Dict[str, Any] = {}
        redacted_fields: List[str] = []
        for field in cleaned_fields:
            if field == "content" and memory_data.get("content") != "[REDACTED]":
                updates[field] = "[REDACTED]"
                redacted_fields.append(field)
            elif field in memory_data and field != "content":
                value = memory_data.get(field)
                if isinstance(value, str) and value != "[REDACTED]":
                    updates[field] = "[REDACTED]"
                    redacted_fields.append(field)

        if not updates:
            return {
                "memory_id": memory_id,
                "redacted": False,
                "reason": "nothing_to_redact",
            }

        self.synapse.store.update_memory(memory_id, updates)
        return {
            "memory_id": memory_id,
            "redacted": True,
            "redacted_fields": sorted(redacted_fields),
        }

    def gdpr_delete(self, user_id: str = None, concept: str = None):
        """Full deletion for GDPR-like erasure."""
        if user_id is None and concept is None:
            raise ValueError("user_id or concept must be provided")

        user_matches = set()
        concept_matches = set()

        if user_id is not None:
            user_target = f"user:{str(user_id).strip()}"
            if not user_target:
                user_target = None

        if concept is not None:
            if not isinstance(concept, str) or not concept.strip():
                raise ValueError("concept must be a non-empty string")
            concept_norm = concept.strip().lower()

        for memory_id, memory_data in self.synapse.store.memories.items():
            metadata = self._get_metadata(memory_data)
            tags = self._as_list(metadata.get("tags"))
            if user_id is not None and user_target in tags:
                user_matches.add(memory_id)

            if concept is not None:
                content = (memory_data.get("content") or "").lower()
                if concept_norm in content:
                    concept_matches.add(memory_id)

        if concept is not None:
            target_ids = user_matches | concept_matches if user_id is not None else concept_matches
        else:
            target_ids = user_matches

        deleted = self._delete_memory_ids(sorted(target_ids))
        return {
            "action": "gdpr_delete",
            "requested_user_id": user_id,
            "requested_concept": concept,
            "matched_count": len(target_ids),
            "deleted_count": len(deleted),
            "deleted_ids": deleted,
        }

    def retention_rules(self, rules: List[Dict]):
        """Apply retention rules.

        Supported rule examples:
          - {'tag': 'temporary', 'ttl_days': 7}
          - {'min_access': 0, 'older_than_days': 90, 'action': 'archive'}
        """
        if not isinstance(rules, list):
            raise ValueError("rules must be a list")

        reports: List[Dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue

            action = str(rule.get("action", "delete")).strip().lower()
            ttl_days = rule.get("ttl_days")
            older_than_days = rule.get("older_than_days")

            candidate_ids = set()
            for memory_id, memory_data in self._iter_candidate_memories(rule):
                if ttl_days is not None:
                    metadata = self._get_metadata(memory_data)
                    ttl = self._coerce_positive_float(ttl_days, None)
                    if ttl is None:
                        continue
                    created_at = float(memory_data.get("created_at", 0.0) or 0.0)
                    if (time.time() - created_at) <= ttl * SECONDS_PER_DAY:
                        continue

                if older_than_days is not None:
                    older_than = self._coerce_positive_float(older_than_days, None)
                    if older_than is None:
                        continue
                    created_at = float(memory_data.get("created_at", 0.0) or 0.0)
                    if (time.time() - created_at) <= older_than * SECONDS_PER_DAY:
                        continue

                candidate_ids.add(memory_id)

            matched_ids = sorted(candidate_ids)

            if action == "archive":
                archived = self._archive_memory_ids(matched_ids, rule)
                reports.append({
                    "action": "archive",
                    "matched_count": len(matched_ids),
                    "archived_count": len(archived),
                    "archived_ids": archived,
                    "rule": rule,
                })
            else:
                deleted = self._delete_memory_ids(matched_ids)
                reports.append({
                    "action": "delete",
                    "matched_count": len(matched_ids),
                    "deleted_count": len(deleted),
                    "deleted_ids": deleted,
                    "rule": rule,
                })

        return reports

    def _iter_candidate_memories(self, rule: Dict) -> List[tuple[int, Dict[str, Any]]]:
        tag = rule.get("tag")
        min_access = self._coerce_int(rule.get("min_access"))
        memory_type = rule.get("memory_type")
        concept = rule.get("concept")

        candidates: List[tuple[int, Dict[str, Any]]] = []
        for memory_id, memory_data in list(self.synapse.store.memories.items()):
            if tag is not None:
                metadata = self._get_metadata(memory_data)
                tags = self._as_list(metadata.get("tags"))
                if str(tag) not in tags:
                    continue

            if min_access is not None and int(memory_data.get("access_count", 0)) > min_access:
                continue

            if concept is not None:
                content = (memory_data.get("content") or "").lower()
                if str(concept).lower() not in content:
                    continue

            if memory_type is not None and str(memory_data.get("memory_type", "")) != str(memory_type):
                continue

            candidates.append((memory_id, memory_data))
        return candidates

    def _archive_memory_ids(self, memory_ids: List[int], rule: Optional[Dict] = None) -> List[int]:
        if not memory_ids:
            return []

        archived: List[int] = []
        now = time.time()
        for memory_id in memory_ids:
            memory_data = self.synapse.store.memories.get(memory_id)
            if memory_data is None:
                continue

            metadata = self._get_metadata(memory_data)
            metadata["archived"] = True
            metadata["archived_at"] = now
            metadata["archive_rule"] = rule or {}
            self.synapse.store.update_memory(memory_id, {"metadata": json.dumps(metadata)})
            archived.append(memory_id)
        return archived

    def _delete_memory_ids(self, memory_ids: Iterable[int]) -> List[int]:
        deleted: List[int] = []
        for memory_id in sorted(set(int(mid) for mid in memory_ids if isinstance(mid, int) and mid > 0)):
            if self.synapse.forget(memory_id):
                deleted.append(memory_id)
        return deleted

    @staticmethod
    def _coerce_positive_float(value: Any, default: Optional[float]) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return default
        if value_f <= 0:
            return default
        return value_f

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            value_i = int(value)
        except (TypeError, ValueError):
            return None
        if value_i < 0:
            return None
        return value_i

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _get_metadata(memory_data: Dict[str, Any]) -> Dict[str, Any]:
        raw = memory_data.get("metadata", "{}")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            return json.loads(raw) if raw else {}
        except (TypeError, json.JSONDecodeError):
            return {}

