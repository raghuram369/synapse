"""Belief version tracking for Synapse memories.

This module tracks fact-level versions inferred from extracted triples.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from contradictions import ContradictionDetector
from triples import extract_triples


@dataclass
class BeliefVersion:
    """A single fact version in a temporal belief chain."""

    fact_key: str
    value: str
    memory_id: int
    valid_from: float
    valid_to: Optional[float]
    reason: str
    confidence: float


class BeliefTracker:
    """Track versioned beliefs derived from memory triples."""

    def __init__(self, contradiction_detector: Optional[ContradictionDetector] = None):
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self._versions: Dict[str, List[BeliefVersion]] = {}
        self._polarity_by_fact: Dict[Tuple[str, int], str] = {}

    @staticmethod
    def _norm(value: Optional[str]) -> str:
        if not value:
            return ""
        lowered = value.strip().lower()
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    @staticmethod
    def _fact_key(subject: str, predicate: str) -> str:
        return f"{subject}|{predicate}"

    def _triple_time(self, memory: Any) -> float:
        valid_from = getattr(memory, "valid_from", None)
        if valid_from is not None:
            return float(valid_from)
        return float(getattr(memory, "created_at", time.time()))

    @classmethod
    def _triple_value(cls, triple) -> str:
        return cls._norm(getattr(triple, "object", ""))

    @classmethod
    def _triple_subject(cls, triple) -> str:
        return cls._norm(getattr(triple, "subject", ""))

    @classmethod
    def _triple_predicate(cls, triple) -> str:
        return cls._norm(getattr(triple, "predicate", ""))

    def _version_signature(self, fact_key: str, memory_id: int) -> Tuple[str, int]:
        return fact_key, int(memory_id)

    def _normalize_statement(self, fact_key: str, value: str, polarity: str) -> str:
        subject, predicate = fact_key.split("|", 1)
        negation = "not " if polarity == "negative" else ""
        return f"{subject} {negation}{predicate} {value}".strip()

    def _find_current(self, fact_key: str) -> Optional[BeliefVersion]:
        versions = self._versions.get(fact_key)
        if not versions:
            return None
        open_versions = [v for v in versions if v.valid_to is None]
        if open_versions:
            return sorted(open_versions, key=lambda v: v.valid_from)[-1]
        return sorted(versions, key=lambda v: v.valid_from)[-1]

    def _current_polarity(self, fact_key: str, memory_id: int) -> Optional[str]:
        return self._polarity_by_fact.get(self._version_signature(fact_key, memory_id))

    def _is_same(self, fact_key: str, value: str, polarity: str, memory_id: int) -> bool:
        current = self._find_current(fact_key)
        if current is None:
            return False
        if current.valid_to is not None:
            return False

        current_polarity = self._current_polarity(fact_key, current.memory_id)
        if current_polarity is None:
            return False
        return current.value == value and current_polarity == polarity

    def _is_contradiction(self, fact_key: str, current: BeliefVersion, value: str, polarity: str) -> bool:
        current_polarity = self._current_polarity(fact_key, current.memory_id) or "positive"
        if current_polarity != polarity and current.value == value:
            return True

        old_text = self._normalize_statement(fact_key, current.value, current_polarity)
        new_text = self._normalize_statement(fact_key, value, polarity)

        if self.contradiction_detector.detect_polarity(old_text, new_text) is not None:
            return True
        if self.contradiction_detector.detect_mutual_exclusion(old_text, new_text) is not None:
            return True
        if self.contradiction_detector.detect_numeric_conflict(old_text, new_text) is not None:
            return True
        return False

    def rebuild(self, memories: List[Any]) -> None:
        """Rebuild version history from an ordered memory list."""
        self._versions = {}
        self._polarity_by_fact = {}
        for memory in sorted(memories, key=lambda item: getattr(item, "created_at", 0.0)):
            self.on_remember(memory)

    def remove_memory(self, memory_id: int) -> int:
        """Remove all fact versions introduced by *memory_id*.

        Returns the number of removed versions.
        """
        if not isinstance(memory_id, int):
            return 0

        removed = 0
        for fact_key in list(self._versions.keys()):
            versions = self._versions.get(fact_key, [])
            survivors = [version for version in versions if version.memory_id != memory_id]
            removed += len(versions) - len(survivors)

            if survivors:
                self._versions[fact_key] = survivors
            else:
                self._versions.pop(fact_key, None)

        # Remove polarity cache entries for this memory.
        to_delete = [
            key for key in self._polarity_by_fact
            if key[1] == memory_id
        ]
        for key in to_delete:
            self._polarity_by_fact.pop(key, None)

        return removed

    def on_remember(self, memory: Any) -> List[BeliefVersion]:
        """Process a new memory and update fact timelines if needed."""
        created: List[BeliefVersion] = []
        memory_id = getattr(memory, "id", None)
        if memory_id is None:
            return created

        triples = extract_triples(getattr(memory, "content", "") or "")
        if not triples:
            return created

        now = self._triple_time(memory)
        seen_fact_keys = set()

        for triple in triples:
            subject = self._triple_subject(triple)
            predicate = self._triple_predicate(triple)
            value = self._triple_value(triple)
            polarity = getattr(triple, "polarity", "positive")

            if not subject or not predicate or not value:
                continue

            fact_key = self._fact_key(subject, predicate)
            if fact_key in seen_fact_keys:
                continue
            seen_fact_keys.add(fact_key)

            if self._is_same(fact_key, value, polarity, memory_id):
                continue

            current = self._find_current(fact_key)
            reason = "new information"

            if current is not None:
                current_polarity = self._current_polarity(fact_key, current.memory_id) or "positive"
                if value == current.value and polarity != current_polarity:
                    reason = "user correction"
                elif self._is_contradiction(fact_key, current, value, polarity):
                    reason = "contradiction resolved"

                if current.valid_to is None:
                    current.valid_to = now

            version = BeliefVersion(
                fact_key=fact_key,
                value=value,
                memory_id=memory_id,
                valid_from=now,
                valid_to=None,
                reason=reason,
                confidence=float(getattr(triple, "confidence", 1.0)),
            )

            self._versions.setdefault(fact_key, []).append(version)
            self._polarity_by_fact[self._version_signature(fact_key, memory_id)] = polarity
            created.append(version)

        return created

    def get_current(self, fact_key: str) -> Optional[BeliefVersion]:
        """Return the current valid version for a fact key."""
        return self._find_current(fact_key)

    def get_history(self, fact_key: str) -> List[BeliefVersion]:
        """Return chronological history for a fact key."""
        versions = self._versions.get(fact_key, [])
        return sorted(versions, key=lambda item: item.valid_from)

    def get_all_current(self) -> Dict[str, BeliefVersion]:
        """Snapshot of all currently-valid beliefs."""
        current: Dict[str, BeliefVersion] = {}
        for fact_key in self._versions:
            version = self._find_current(fact_key)
            if version is not None:
                current[fact_key] = version
        return current

    def get_matching_history(self, topic: str) -> List[BeliefVersion]:
        """History for all fact keys matching a query string."""
        query = self._norm(topic)
        if not query:
            return []

        versions: List[BeliefVersion] = []
        for fact_key, _history in self._versions.items():
            if query in fact_key:
                versions.extend(_history)
        return sorted(versions, key=lambda item: item.valid_from)
