"""Vector clocks for causality tracking in federated memory sync."""

from __future__ import annotations
import json
from typing import Dict


class VectorClock:
    """
    Vector clock for distributed causality tracking.
    Maps node_id -> logical timestamp (counter).
    """

    __slots__ = ("_clock",)

    def __init__(self, clock: Dict[str, int] | None = None):
        self._clock: Dict[str, int] = dict(clock) if clock else {}

    # ── Mutation ──────────────────────────────────────────────

    def increment(self, node_id: str) -> "VectorClock":
        """Tick the clock for *node_id* and return self (for chaining)."""
        self._clock[node_id] = self._clock.get(node_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Point-wise max merge. Returns self."""
        for nid, ts in other._clock.items():
            self._clock[nid] = max(self._clock.get(nid, 0), ts)
        return self

    # ── Comparison ────────────────────────────────────────────

    def __le__(self, other: "VectorClock") -> bool:
        """True if self happened-before or is concurrent with other (≤)."""
        for nid, ts in self._clock.items():
            if ts > other._clock.get(nid, 0):
                return False
        return True

    def __lt__(self, other: "VectorClock") -> bool:
        """True if self strictly happened-before other."""
        return self <= other and self != other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_keys = set(self._clock) | set(other._clock)
        return all(self._clock.get(k, 0) == other._clock.get(k, 0) for k in all_keys)

    def __hash__(self):
        return hash(tuple(sorted(self._clock.items())))

    def is_concurrent(self, other: "VectorClock") -> bool:
        """True if neither clock happened-before the other."""
        return not (self <= other) and not (other <= self)

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, int]:
        return dict(self._clock)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "VectorClock":
        return cls(d)

    def to_json(self) -> str:
        return json.dumps(self._clock, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "VectorClock":
        return cls(json.loads(s))

    def __repr__(self):
        return f"VectorClock({self._clock})"

    def copy(self) -> "VectorClock":
        return VectorClock(dict(self._clock))
