"""Evidence chain construction for explainable recalls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from contradictions import ContradictionDetector
from triples import Triple, extract_triples


@dataclass
class EvidenceChain:
    claim: str
    supporting_memories: List[int]
    contradicting_memories: List[int]
    confidence: float
    first_seen: float
    last_confirmed: float


class EvidenceCompiler:
    """Compile lightweight evidence chains from memories and triples."""

    def __init__(self, contradiction_detector: ContradictionDetector | None = None):
        self.contradiction_detector = contradiction_detector or ContradictionDetector()

    @staticmethod
    def _norm(value: str) -> str:
        return (value or "").strip().lower()

    @staticmethod
    def _triple_memory_time(memory: Any) -> float:
        return float(getattr(memory, "created_at", 0.0))

    @staticmethod
    def _triple_claim(triple: Triple) -> str:
        negation = "not " if triple.polarity == "negative" else ""
        return f"{triple.subject} {negation}{triple.predicate} {triple.object}".strip()

    def _extract_triples(self, memory: Any, triples_index: Any) -> List[Triple]:
        memory_id = getattr(memory, "id", None)
        if memory_id is None:
            return []

        if triples_index is not None:
            getter = getattr(triples_index, "get_triples_for_memory", None)
            if callable(getter):
                try:
                    triples = getter(memory_id)
                    if triples:
                        return list(triples)
                except Exception:
                    pass

        return extract_triples(getattr(memory, "content", "") or "")

    @staticmethod
    def _fact_key(triple: Triple) -> str:
        return f"{triple.subject}|{triple.predicate}|{triple.polarity}|{triple.object}"

    @staticmethod
    def _group_key(triple: Triple) -> str:
        return f"{triple.subject}|{triple.predicate}"

    @staticmethod
    def _parse_memory_time(memory: Any) -> float:
        return float(getattr(memory, "created_at", 0.0))

    def _triple_evidence_conflict(self, a: Triple, b: Triple) -> bool:
        if a.subject != b.subject or a.predicate != b.predicate:
            return False

        if a.polarity != b.polarity and a.object == b.object:
            return True

        a_text = self._triple_claim(a)
        b_text = self._triple_claim(b)
        if self.contradiction_detector.detect_mutual_exclusion(a_text, b_text) is not None:
            return True
        if self.contradiction_detector.detect_numeric_conflict(a_text, b_text) is not None:
            return True
        if a.polarity != b.polarity and self.contradiction_detector.detect_polarity(a_text, b_text) is not None:
            return True

        return False

    def compile(self, memories: List[Any], triples_index) -> List[EvidenceChain]:
        """Return evidence chains for all unique facts in the given memories."""
        if not memories:
            return []

        claim_to_data: Dict[str, Dict[str, Any]] = {}
        group_to_claims: Dict[str, Set[str]] = {}

        for memory in memories:
            memory_id = getattr(memory, "id", None)
            if memory_id is None:
                continue

            triples = self._extract_triples(memory, triples_index)
            seen_for_memory: Set[str] = set()

            for triple in triples:
                if not triple.subject or not triple.predicate or not triple.object:
                    continue

                claim = self._triple_claim(triple)
                key = self._fact_key(triple)
                if key in seen_for_memory:
                    continue
                seen_for_memory.add(key)

                entry = claim_to_data.setdefault(
                    claim,
                    {
                        "triple": triple,
                        "supporting": set(),
                        "times": [],
                        "confidences": [],
                        "group": self._group_key(triple),
                        "contradicting": set(),
                    },
                )

                entry["supporting"].add(memory_id)
                entry["times"].append(self._parse_memory_time(memory))
                entry["confidences"].append(float(getattr(triple, "confidence", 1.0)))

                group_to_claims.setdefault(entry["group"], set()).add(claim)

        if not claim_to_data:
            return []

        claims = sorted(claim_to_data.keys())
        for group_id, group_claims in group_to_claims.items():
            group_claim_list = sorted(group_claims)
            for idx, claim_a in enumerate(group_claim_list):
                for claim_b in group_claim_list[idx + 1:]:
                    triple_a = claim_to_data[claim_a]["triple"]
                    triple_b = claim_to_data[claim_b]["triple"]
                    if not self._triple_evidence_conflict(triple_a, triple_b):
                        continue
                    claim_to_data[claim_a]["contradicting"].update(
                        claim_to_data[claim_b]["supporting"],
                    )
                    claim_to_data[claim_b]["contradicting"].update(
                        claim_to_data[claim_a]["supporting"],
                    )

        chains: List[EvidenceChain] = []
        for claim in claims:
            data = claim_to_data[claim]
            supporting = sorted(data["supporting"])
            if not supporting:
                continue

            contradicting = sorted(data["contradicting"])
            times = data["times"]
            confidences = data["confidences"]
            support_confidence = sum(confidences) / len(confidences)

            penalty = 0.0
            if contradicting:
                penalty = min(0.4, 0.1 * len(contradicting))
            confidence = max(0.0, min(1.0, support_confidence - penalty))

            chains.append(
                EvidenceChain(
                    claim=claim,
                    supporting_memories=supporting,
                    contradicting_memories=contradicting,
                    confidence=confidence,
                    first_seen=min(times),
                    last_confirmed=max(times),
                )
            )

        chains.sort(key=lambda item: (-item.confidence, item.claim))
        return chains
