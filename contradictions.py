"""Zero-LLM contradiction detection helpers for Synapse memories."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from entity_graph import extract_concepts


@dataclass
class Contradiction:
    memory_id_a: int
    memory_id_b: int
    kind: str  # 'polarity', 'mutual_exclusion', 'numeric_range', 'temporal_conflict'
    description: str
    confidence: float  # 0-1
    detected_at: float


class ContradictionDetector:
    _NEGATION_MARKERS = {
        "not", "no", "never", "cannot", "can't", "didn't", "doesn't", "don't",
        "won't", "wasn't", "weren't", "isn't", "aren't",
        "shouldn't", "couldn't", "wouldn't", "mustn't",
    }

    _AUX_MARKERS = {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "can", "could",
        "should", "would", "will", "must", "shall",
        "i", "me", "my", "you", "he", "she", "we", "they", "user",
        "person", "someone",
    }

    _STOPWORDS = {
        "the", "a", "an", "this", "that", "to", "of", "in", "on", "for",
        "with", "at", "by", "from", "it", "and", "or", "if", "then", "so",
        "my", "your", "his", "her", "its", "our", "their",
    }

    _NEGATION_MAP = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "mustn't": "must not",
    }

    def __init__(self):
        self.known_exclusive_sets: List[Set[str]] = [
            {"male", "female", "non-binary"},
            {"married", "single", "divorced", "widowed"},
            {"vegetarian", "vegan", "pescatarian", "omnivore"},
            {"cat", "dog", "bird", "fish"},
        ]
        self.registered_contradictions: List[Contradiction] = []
        self._contradiction_index: Dict[Tuple[int, int, str], int] = {}
        self._resolved_pairs: Set[Tuple[int, int, str]] = set()

    @staticmethod
    def _pair_key(id_a: int, id_b: int, kind: str) -> Tuple[int, int, str]:
        if id_a <= id_b:
            return id_a, id_b, kind
        return id_b, id_a, kind

    @classmethod
    def _normalise_text(cls, text: str) -> str:
        lowered = text.lower().strip()
        for source, repl in cls._NEGATION_MAP.items():
            lowered = lowered.replace(source, repl)
        lowered = re.sub(r"[^a-z0-9\- ]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9\-]+", text)

    @classmethod
    def _normalise_predicate_token(cls, token: str) -> str:
        if token.endswith("ing") and len(token) > 4:
            return token[:-3]
        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if token.endswith("ed") and len(token) > 3:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _normalize_polarity_signature(self, text: str) -> Tuple[Set[str], bool]:
        cleaned = self._normalise_text(text)
        tokens = self._tokenize(cleaned)

        if not tokens:
            return set(), False

        negated = False
        signature: Set[str] = set()
        for token in tokens:
            if token in self._NEGATION_MARKERS:
                negated = True
                continue
            if token in self._AUX_MARKERS or token in self._STOPWORDS:
                continue
            signature.add(self._normalise_predicate_token(token))

        return signature, negated

    @staticmethod
    def _signature_similarity(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    def _next_conflict(
        self, kind: str, description: str, memory_id_a: int, memory_id_b: int,
        confidence: float, detected_at: float,
    ) -> Contradiction:
        return Contradiction(
            memory_id_a=memory_id_a,
            memory_id_b=memory_id_b,
            kind=kind,
            description=description,
            confidence=max(0.0, min(1.0, confidence)),
            detected_at=detected_at,
        )

    def _coerce_memories(self, memories: List) -> List[Tuple[int, str, float]]:
        views: List[Tuple[int, str, float]] = []
        for memory in memories:
            if isinstance(memory, dict):
                memory_id = memory.get("id")
                content = memory.get("content", "")
                created_at = memory.get("created_at", 0.0)
            else:
                memory_id = getattr(memory, "id", None)
                content = getattr(memory, "content", "")
                created_at = getattr(memory, "created_at", 0.0)

            if not isinstance(memory_id, int) or not content:
                continue
            views.append((memory_id, str(content), float(created_at or 0.0)))
        return views

    def _register(self, contradiction: Contradiction) -> Optional[Contradiction]:
        if contradiction.memory_id_a <= 0 or contradiction.memory_id_b <= 0:
            return None
        if contradiction.memory_id_a == contradiction.memory_id_b:
            return None
        if contradiction.confidence <= 0.0:
            return None

        key = self._pair_key(
            contradiction.memory_id_a,
            contradiction.memory_id_b,
            contradiction.kind,
        )
        if key in self._resolved_pairs:
            return None

        if key in self._contradiction_index:
            idx = self._contradiction_index[key]
            existing = self.registered_contradictions[idx]
            if contradiction.confidence <= existing.confidence:
                return existing
            self.registered_contradictions[idx] = contradiction
            return contradiction

        self._contradiction_index[key] = len(self.registered_contradictions)
        self.registered_contradictions.append(contradiction)
        return contradiction

    def unresolved_contradictions(self) -> List[Contradiction]:
        return [
            c for c in self.registered_contradictions
            if self._pair_key(c.memory_id_a, c.memory_id_b, c.kind) not in self._resolved_pairs
        ]

    def resolve_contradiction(self, contradiction_id: int) -> Optional[Contradiction]:
        unresolved = self.unresolved_contradictions()
        if contradiction_id < 0 or contradiction_id >= len(unresolved):
            return None
        target = unresolved[contradiction_id]
        key = self._pair_key(target.memory_id_a, target.memory_id_b, target.kind)
        self._resolved_pairs.add(key)
        return target

    def get_conflicted_memory_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for contradiction in self.unresolved_contradictions():
            ids.add(contradiction.memory_id_a)
            ids.add(contradiction.memory_id_b)
        return ids

    @staticmethod
    def _extract_concepts(text: str) -> Set[str]:
        concepts = set()
        for concept_name, _ in extract_concepts(text):
            concepts.add(concept_name.lower())
        return concepts

    def detect_polarity(self, text_a: str, text_b: str) -> Optional[Contradiction]:
        """Detect when two statements directly contradict via negation."""
        sig_a, neg_a = self._normalize_polarity_signature(text_a)
        sig_b, neg_b = self._normalize_polarity_signature(text_b)

        if not sig_a or not sig_b:
            return None
        if neg_a == neg_b:
            return None

        overlap = self._signature_similarity(sig_a, sig_b)
        if overlap < 0.5:
            return None

        confidence = 0.55 + 0.35 * overlap
        common = ", ".join(sorted(sig_a & sig_b))
        description = f"Possible polarity contradiction on '{common}'"
        return self._next_conflict(
            "polarity",
            description,
            -1,
            -1,
            confidence,
            time.time(),
        )

    def detect_mutual_exclusion(self, text_a: str, text_b: str) -> Optional[Contradiction]:
        """Check if texts assert mutually exclusive values from known sets."""
        lower_a = self._normalise_text(text_a)
        lower_b = self._normalise_text(text_b)

        for values in self.known_exclusive_sets:
            matched_a: Set[str] = set()
            matched_b: Set[str] = set()

            for value in values:
                token = value.lower()
                pattern = rf"\b{re.escape(token)}\b"
                if re.search(pattern, lower_a):
                    matched_a.add(token)
                if re.search(pattern, lower_b):
                    matched_b.add(token)

            if not matched_a or not matched_b:
                continue
            if matched_a == matched_b:
                continue
            all_values = sorted(matched_a | matched_b)
            if len(all_values) < 2:
                continue

            description = (
                "Mutual-exclusion violation across set values: "
                f"{', '.join(all_values)}"
            )
            return self._next_conflict(
                "mutual_exclusion",
                description,
                -1,
                -1,
                0.95,
                time.time(),
            )

        return None

    @staticmethod
    def _extract_numeric_facts(text: str) -> Dict[str, float]:
        normalized = re.sub(r"[^a-z0-9\.\- ]", " ", text.lower())
        facts: Dict[str, float] = {}

        stable_attrs = {"age", "height", "weight", "salary", "temperature", "score", "rating"}
        age_units = {"year", "years", "yr", "yrs", "y"}
        weight_units = {"kg", "kgs", "kilogram", "kilograms", "lb", "lbs", "pound", "pounds", "st", "stone"}
        height_units = {"cm", "centimeter", "centimeters", "m", "meter", "meters", "ft", "feet", "inch", "inches"}

        # 1) Explicit age phrasing: "25 years old"
        for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:years?|yrs?|yr)s?\s*old\b", normalized):
            facts["age"] = float(match.group(1))

        # 2) Explicit attribute then value
        for match in re.finditer(
            r"\b(age|height|weight|salary|temperature|score|rating)\b[^a-z0-9]{0,12}(\d{1,3}(?:\.\d+)?)\b",
            normalized,
        ):
            attr = match.group(1)
            value = float(match.group(2))
            if attr in stable_attrs:
                facts[attr] = value

        # 3) Number then unit
        for match in re.finditer(
            r"(\d{1,3}(?:\.\d+)?)\s*(kg|kgs|kilogram|kilograms|lb|lbs|pound|pounds|st|stone|cm|centimeter|centimeters|m|meter|meters|ft|feet|inch|inches|years|year|yr|yrs|y)\b",
            normalized,
        ):
            value = float(match.group(1))
            unit = match.group(2)
            if unit in age_units:
                facts["age"] = value
            elif unit in weight_units:
                facts["weight"] = value
            elif unit in height_units:
                facts["height"] = value

        return facts

    def detect_numeric_conflict(self, text_a: str, text_b: str) -> Optional[Contradiction]:
        """Detect conflicting numeric values for same attribute."""
        facts_a = self._extract_numeric_facts(text_a)
        facts_b = self._extract_numeric_facts(text_b)

        for attr, value_a in facts_a.items():
            if attr not in facts_b:
                continue
            value_b = facts_b[attr]
            if math.isclose(value_a, value_b):
                continue
            diff = abs(value_a - value_b)
            base = max(abs(value_a), abs(value_b), 1.0)
            confidence = max(0.4, min(1.0, 0.5 + (diff / (base + diff))))
            return self._next_conflict(
                "numeric_range",
                f"Numeric contradiction on '{attr}': {value_a} vs {value_b}",
                -1,
                -1,
                confidence,
                time.time(),
            )

        return None

    def detect_temporal_conflict(self, text_a: str, text_b: str, time_a: float, time_b: float) -> Optional[Contradiction]:
        """Detect when facts that should be stable change unexpectedly."""
        base_conflict = (
            self.detect_polarity(text_a, text_b)
            or self.detect_mutual_exclusion(text_a, text_b)
            or self.detect_numeric_conflict(text_a, text_b)
        )
        if base_conflict is None:
            return None

        age = abs(float(time_a) - float(time_b))
        # 1.0 at same timestamp, 0.5 after ~1 day, 0.1 after 9 days.
        weight = 1.0 / (1.0 + (age / 86_400.0))
        confidence = base_conflict.confidence * weight
        if confidence < 0.1:
            return None

        description = (
            f"Temporal contradiction over {age:.1f}s; recent fact should probably win"
        )
        return self._next_conflict("temporal_conflict", description, -1, -1, confidence, time.time())

    @staticmethod
    def _should_check_pair(concepts_a: Set[str], concepts_b: Set[str]) -> bool:
        if concepts_a and concepts_b and not concepts_a.isdisjoint(concepts_b):
            return True
        if not concepts_a or not concepts_b:
            return bool((concepts_a | concepts_b) and len(concepts_a | concepts_b) >= 2)
        return False

    def _detect_pair(self, text_a: str, text_b: str, time_a: float, time_b: float,
                     memory_id_a: int, memory_id_b: int,
                     concepts_a: Set[str], concepts_b: Set[str]) -> List[Contradiction]:
        if not self._should_check_pair(concepts_a, concepts_b):
            return []

        results: List[Contradiction] = []
        now = max(time_a, time_b, time.time())

        detected = self.detect_polarity(text_a, text_b)
        if detected is not None:
            results.append(
                self._next_conflict(
                    detected.kind,
                    detected.description,
                    memory_id_a,
                    memory_id_b,
                    detected.confidence,
                    now,
                )
            )

        detected = self.detect_mutual_exclusion(text_a, text_b)
        if detected is not None:
            results.append(
                self._next_conflict(
                    detected.kind,
                    detected.description,
                    memory_id_a,
                    memory_id_b,
                    detected.confidence,
                    now,
                )
            )

        detected = self.detect_numeric_conflict(text_a, text_b)
        if detected is not None:
            results.append(
                self._next_conflict(
                    detected.kind,
                    detected.description,
                    memory_id_a,
                    memory_id_b,
                    detected.confidence,
                    now,
                )
            )

        detected = self.detect_temporal_conflict(text_a, text_b, time_a, time_b)
        if detected is not None:
            results.append(
                self._next_conflict(
                    detected.kind,
                    detected.description,
                    memory_id_a,
                    memory_id_b,
                    detected.confidence,
                    now,
                )
            )

        deduped: List[Contradiction] = []
        seen: Set[Tuple[str, int, int]] = set()
        for item in results:
            key = (item.kind, item.memory_id_a, item.memory_id_b)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def check_new_memory(self, new_text: str, existing_memories: List,
                        new_memory_id: Optional[int] = None, new_time: Optional[float] = None) -> List[Contradiction]:
        """Check a new memory against existing ones. Called on remember()."""
        if not new_text.strip():
            return []

        now = time.time() if new_time is None else new_time
        existing = self._coerce_memories(existing_memories)
        if not existing:
            return []

        normalized_new = self._normalise_text(new_text)
        concepts_new = self._extract_concepts(normalized_new)
        if not concepts_new:
            concepts_new = {
                token for token in re.findall(r"[a-z0-9]+", normalized_new)
                if token not in self._STOPWORDS
            }

        contradictions: List[Contradiction] = []
        for memory_id, content, created_at in existing:
            normalized_existing = self._normalise_text(content)
            concepts_existing = self._extract_concepts(normalized_existing)
            if not concepts_existing:
                concepts_existing = {
                    token for token in re.findall(r"[a-z0-9]+", normalized_existing)
                    if token not in self._STOPWORDS
                }

            detected_items = self._detect_pair(
                normalized_new,
                normalized_existing,
                now,
                created_at,
                -1 if new_memory_id is None else new_memory_id,
                memory_id,
                concepts_new,
                concepts_existing,
            )
            contradictions.extend(detected_items)

        deduped: List[Contradiction] = []
        seen: Set[Tuple[str, int, int]] = set()
        for contradiction in contradictions:
            key = (contradiction.kind, contradiction.memory_id_a, contradiction.memory_id_b)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(contradiction)
        return deduped

    def scan_memories(self, memories: List) -> List[Contradiction]:
        """Scan all memories pairwise (with smart pruning — only check memories with overlapping concepts)."""
        views = self._coerce_memories(memories)
        if len(views) < 2:
            return []

        indexed: Dict[int, Tuple[str, float, Set[str]]] = {}
        concept_to_ids: Dict[str, Set[int]] = {}
        for memory_id, content, created_at in views:
            normalized = self._normalise_text(content)
            concepts = self._extract_concepts(normalized)
            if not concepts:
                concepts = {
                    token for token in re.findall(r"[a-z0-9]+", normalized)
                    if token not in self._STOPWORDS
                }
            indexed[memory_id] = (content, created_at, concepts)
            for concept in concepts:
                concept_to_ids.setdefault(concept, set()).add(memory_id)

        if not concept_to_ids:
            return []

        candidate_pairs: Set[Tuple[int, int]] = set()
        for ids in concept_to_ids.values():
            ordered_ids = sorted(ids)
            if len(ordered_ids) < 2:
                continue
            for i, left in enumerate(ordered_ids):
                for right in ordered_ids[i + 1:]:
                    candidate_pairs.add((left, right))

        if not candidate_pairs:
            return []

        contradictions: List[Contradiction] = []
        for left_id, right_id in sorted(candidate_pairs):
            left_content, left_time, left_concepts = indexed[left_id]
            right_content, right_time, right_concepts = indexed[right_id]

            detected_items = self._detect_pair(
                left_content,
                right_content,
                left_time,
                right_time,
                left_id,
                right_id,
                left_concepts,
                right_concepts,
            )

            for contradiction in detected_items:
                resolved = self._register(contradiction)
                if resolved is not None:
                    contradictions.append(resolved)

        deduped: List[Contradiction] = []
        seen: Set[Tuple[str, int, int]] = set()
        for contradiction in contradictions:
            key = (contradiction.kind, contradiction.memory_id_a, contradiction.memory_id_b)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(contradiction)
        return deduped

    def add_exclusive_set(self, values: Set[str]):
        """Register custom mutual exclusion set."""
        if not values:
            return
        self.known_exclusive_sets.append({v.lower() for v in values if v})
