"""Temporal helpers for parsing date values and bitemporal queries."""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional, Set

from entity_graph import extract_concepts


if False:
    # pragma: no cover
    from synapse import Memory


def _to_float_timestamp(value: Optional[Any]) -> Optional[float]:
    """Convert a supported temporal value to epoch seconds.

    Returns ``None`` for unparseable values.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    lower = raw.lower().strip()

    relative = {
        "today": timedelta(days=0),
        "yesterday": timedelta(days=-1),
        "tomorrow": timedelta(days=1),
    }
    if lower in relative:
        dt = datetime.now(timezone.utc) + relative[lower]
        return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    if lower == "last week":
        dt = datetime.now(timezone.utc) - timedelta(days=7)
        return dt.timestamp()

    if lower in {"this week", "last 7 days"}:
        dt = datetime.now(timezone.utc)
        if lower == "this week":
            dt = dt - timedelta(days=dt.weekday())
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt.timestamp()

    if lower == "last month":
        dt = datetime.now(timezone.utc)
        if dt.month == 1:
            dt = dt.replace(year=dt.year - 1, month=12)
        else:
            dt = dt.replace(month=dt.month - 1)
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    if lower == "this month":
        dt = datetime.now(timezone.utc)
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    # Parse ISO-like forms
    normalized = raw.replace("Z", "+00:00")
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]:
        try:
            dt = datetime.strptime(normalized, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass

    # Parse month names like "March 2024"
    month_match = re.fullmatch(r"([a-zA-Z]+)\s+(\d{4})", raw)
    if month_match:
        month_text = month_match.group(1)
        year = int(month_match.group(2))
        month_name = month_text.lower()
        month_lookup = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        month = month_lookup.get(month_name)
        if month is None:
            return None
        return datetime(year, month, 1, tzinfo=timezone.utc).timestamp()

    # Float timestamp string
    try:
        return float(raw)
    except ValueError:
        return None


def parse_temporal(value: str) -> Optional[float]:
    """Parse various date formats: ISO dates, month names, and simple phrases."""
    return _to_float_timestamp(value)


def _memory_start(memory: Any) -> float:
    if getattr(memory, "valid_from", None) is not None:
        return float(memory.valid_from)
    if getattr(memory, "observed_at", None) is not None:
        return float(memory.observed_at)
    return float(memory.created_at)


def _memory_end(memory: Any) -> float:
    valid_to = getattr(memory, "valid_to", None)
    return float(valid_to) if valid_to is not None else float("inf")


def _memory_concepts(memory: Any) -> Set[str]:
    content = getattr(memory, "content", "")
    return {name.lower() for name, _ in extract_concepts(content)}


def memories_as_of(memories: List["Memory"], as_of: float) -> List["Memory"]:
    """Return the latest known facts as of time ``as_of``.

    For recall, "as_of" is interpreted as: consider memories that started at or
    before ``as_of`` (by valid_from/observed_at/created_at), then keep only the
    most recent version per overlapping concept cluster.
    """
    if as_of is None:
        return list(memories)
    as_of_float = float(as_of)
    eligible = [memory for memory in memories if _memory_start(memory) <= as_of_float]
    return latest_facts(eligible)


def memories_during(memories: List["Memory"], start: float, end: float) -> List["Memory"]:
    """Return memories that overlap with a time interval ``[start, end]``."""
    start_f = float(start)
    end_f = float(end)
    if start_f > end_f:
        start_f, end_f = end_f, start_f

    results = []
    for memory in memories:
        memory_start = _memory_start(memory)
        memory_end = _memory_end(memory)
        # Overlap if intervals intersect.
        if memory_start <= end_f and memory_end >= start_f:
            # Use strict bounds for open-ended validity semantics.
            if memory_end > start_f and memory_start < end_f:
                results.append(memory)
    return results


def latest_facts(memories: List["Memory"]) -> List["Memory"]:
    """Return only the most recent valid version for overlapping concept groups."""
    if not memories:
        return []

    def _sort_key(memory: "Memory") -> tuple:
        observed = getattr(memory, "observed_at", None)
        valid_from = getattr(memory, "valid_from", None)
        return (
            float(valid_from if valid_from is not None else (observed if observed is not None else memory.created_at)),
            float(memory.id) if memory.id is not None else -1.0,
        )

    ordered = sorted(memories, key=_sort_key, reverse=True)
    selected: List["Memory"] = []
    seen_concepts: List[Set[str]] = []
    seen_fact_keys: Set[tuple] = set()

    def _overlap(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    for memory in ordered:
        # Prefer triple-derived grouping when possible; it avoids over-deduping
        # distinct facts that share generic concepts (e.g. "User likes X" vs "User likes Y").
        fact_keys: Set[tuple] = set()
        try:
            from triples import extract_triples  # local import to avoid unused dependency paths
        except Exception:  # pragma: no cover
            extract_triples = None

        if extract_triples is not None:
            try:
                triples = extract_triples(getattr(memory, "content", "") or "")
            except Exception:
                triples = []
            for triple in triples:
                obj = getattr(triple, "object", "") or ""
                obj_head = obj.split()[0] if obj else ""
                subj = getattr(triple, "subject", "") or ""
                pred = getattr(triple, "predicate", "") or ""
                if subj and pred and obj_head:
                    fact_keys.add((subj, pred, obj_head))

        if fact_keys:
            if fact_keys & seen_fact_keys:
                continue
            selected.append(memory)
            seen_fact_keys |= fact_keys
            continue

        concepts = _memory_concepts(memory)
        if not concepts:
            concepts = {f"{memory.id}:__unmatched__"} if memory.id is not None else {f"{id(memory)}:__unmatched__"}

        # "Any overlap" is too aggressive (e.g. "User likes X" vs "User likes Y").
        # Use a modest similarity threshold so near-duplicates dedup but distinct facts remain.
        if any(_overlap(concepts, existing) >= 0.5 for existing in seen_concepts):
            continue

        selected.append(memory)
        seen_concepts.append(concepts)

    return selected


def temporal_chain(memories: List["Memory"], topic_concepts: Set[str]) -> List["Memory"]:
    """Build a chronological chain of memories for a topic."""
    normalized_topic = {c.lower() for c in topic_concepts}
    relevant = [
        memory for memory in memories
        if _memory_concepts(memory) & normalized_topic
    ]

    return sorted(
        relevant,
        key=lambda memory: _memory_start(memory),
    )
