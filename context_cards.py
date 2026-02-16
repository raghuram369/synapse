"""Shareable, replayable context cards for Synapse memory bundles."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


RECORD_DATA = 0x01
RECORD_END = 0xFF
RECORD_HEADER_SIZE = 5

CARD_FILE_MAGIC = b"SCDP"
CARD_FILE_VERSION = 1


def _pack_record(record_type: int, payload: bytes) -> bytes:
    return struct.pack(">BI", record_type, len(payload)) + payload


def _iter_records(data: bytes, start: int = 0) -> Iterable[tuple[int, bytes]]:
    """Iterate through TLV records in a byte stream."""
    pos = start
    total = len(data)
    while pos + RECORD_HEADER_SIZE <= total:
        record_type, length = struct.unpack(">BI", data[pos: pos + RECORD_HEADER_SIZE])
        pos += RECORD_HEADER_SIZE
        if record_type == RECORD_END:
            return
        if pos + length > total:
            raise ValueError("Truncated TLV payload")
        payload = data[pos: pos + length]
        pos += length
        yield record_type, payload


def generate_card_id(seed: Optional[Dict[str, Any]] = None) -> str:
    """Generate a short deterministic id for a card payload."""
    payload = seed if isinstance(seed, dict) else {
        "created_at": time.time(),
        "entropy": time.time_ns(),
    }
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.blake2s(packed.encode("utf-8"), digest_size=6).hexdigest()
    return f"card-{digest}"


def _normalize_fact_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _node_signature(node: Any) -> str:
    """Compute a stable, hashable identity for a graph node in diff output."""
    if isinstance(node, dict):
        if "id" in node:
            return f"id:{node.get('id')}"
        if "name" in node:
            return f"name:{node.get('name')}"
        try:
            payload = json.dumps(node, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            payload = repr(node)
        return f"json:{payload}"
    if isinstance(node, (list, tuple)):
        try:
            payload = json.dumps(node, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            payload = repr(node)
        return f"list:{payload}"
    return f"value:{node!r}"


class ContextCard:
    """A deterministic, shareable memory bundle."""

    def __init__(self, pack, card_id: str = None):
        self.query = pack.query
        self.memories = list(pack.memories)
        self.graph_slice = dict(pack.graph_slice)
        self.evidence = list(pack.evidence)
        self.contradictions: List[Any] = []
        self.summary = pack.to_compact()
        self.created_at = time.time()
        self.metadata = dict(getattr(pack, "metadata", {})) if getattr(pack, "metadata", None) else {}

        if card_id:
            self.card_id = str(card_id)
        else:
            self.card_id = generate_card_id(self._identity_seed())

    def _identity_seed(self) -> Dict[str, Any]:
        memory_ids = [
            memory.get("id")
            for memory in self.memories
            if isinstance(memory, dict) and memory.get("id") is not None
        ]
        evidence_signatures = [
            (item.get("claim"), item.get("source_id"), item.get("target_id"), item.get("relation"))
            for item in self.evidence
            if isinstance(item, dict)
        ]
        return {
            "query": self.query,
            "memory_ids": sorted(memory_ids),
            "evidence_signatures": evidence_signatures,
            "graph_nodes": len(self.graph_slice.get("nodes", [])),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "query": self.query,
            "memories": self.memories,
            "graph_slice": self.graph_slice,
            "evidence": self.evidence,
            "contradictions": self.contradictions,
            "summary": self.summary,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize the card for programmatic use."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContextCard":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        obj = cls.__new__(cls)
        obj.card_id = str(payload.get("card_id", ""))
        obj.query = payload.get("query", "")
        obj.memories = list(payload.get("memories", []))
        obj.graph_slice = dict(payload.get("graph_slice", {}))
        obj.evidence = list(payload.get("evidence", []))
        obj.contradictions = list(payload.get("contradictions", []))
        obj.summary = payload.get("summary", "")
        obj.created_at = float(payload.get("created_at", 0.0) or 0.0)
        obj.metadata = dict(payload.get("metadata", {}))
        return obj

    @classmethod
    def from_json(cls, data: str) -> "ContextCard":
        if not isinstance(data, str):
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            else:
                raise TypeError("data must be a JSON string")
        return cls.from_dict(json.loads(data))

    def _summarize_graph(self) -> str:
        concepts = []
        for item in self.graph_slice.get("concepts", []):
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    concepts.append(name)

        if concepts:
            return " -> ".join(sorted(set(concepts))[:12])

        nodes = []
        for node in self.graph_slice.get("nodes", []):
            if not isinstance(node, dict):
                continue
            for concept in node.get("concepts", []) or []:
                if isinstance(concept, str):
                    nodes.append(concept)
        return " -> ".join(sorted(set(nodes))[:12])

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"## 🧠 Memory Card: {self.card_id}")
        lines.append(f"**Query:** {self.query}")
        lines.append(f"**Created:** {datetime.fromtimestamp(self.created_at).isoformat(sep=' ', timespec='seconds')}")
        lines.append("")

        lines.append("### Summary")
        lines.append(self.summary or "_No summary generated._")
        lines.append("")

        lines.append("### Evidence")
        if self.evidence:
            for item in self.evidence:
                if not isinstance(item, dict):
                    continue
                claim = item.get("claim")
                if isinstance(claim, str) and claim:
                    supporting = item.get("supporting_memories", []) or []
                    confidence = item.get("confidence")
                    if isinstance(confidence, (int, float)):
                        confidence_text = f"{float(confidence):.2f}"
                    else:
                        confidence_text = "n/a"
                    lines.append(
                        f"- Fact: {claim} (supported by {len(supporting)} memories, confidence: {confidence_text})"
                    )
                    continue

                source = item.get("source_id")
                target = item.get("target_id")
                relation = item.get("relation")
                if source is not None and target is not None:
                    lines.append(f"- [{source}] {relation} [{target}]")
                else:
                    lines.append(f"- Evidence record: {json.dumps(item)}")
        else:
            lines.append("- No evidence captured.")
        lines.append("")

        lines.append("### Graph Context")
        lines.append(f"Key concepts: {self._summarize_graph() or 'n/a'}")
        lines.append("")

        lines.append("### Contradictions")
        if self.contradictions:
            for item in self.contradictions:
                if isinstance(item, str):
                    lines.append(f"⚠️ {item}")
                elif isinstance(item, dict):
                    a = item.get("a") or item.get("memory_id") or item.get("memory_a")
                    b = item.get("b") or item.get("other_id") or item.get("memory_b")
                    if a is not None and b is not None:
                        lines.append(f"⚠️ Memory #{a} conflicts with Memory #{b}")
        else:
            lines.append("No contradictions noted.")

        return "\n".join(lines)

    def to_bytes(self) -> bytes:
        payload = self.to_json().encode("utf-8")
        return _pack_record(RECORD_DATA, payload) + _pack_record(RECORD_END, b"")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextCard":
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")

        for record_type, payload in _iter_records(bytes(data)):
            if record_type != RECORD_DATA:
                continue
            return cls.from_json(payload.decode("utf-8"))

        raise ValueError("No payload record found in card bytes")

    def replay(self, synapse_instance) -> "ContextPack":
        budget = int(self.metadata.get("requested_budget") or 2000)
        pack = synapse_instance.compile_context(self.query, budget=budget)

        replay_card = ContextCard(pack, card_id=self.card_id)
        diff = self.diff(replay_card)

        pack.metadata = dict(pack.metadata)
        pack.metadata["replayed_from"] = {
            "card_id": self.card_id,
            "query": self.query,
            "created_at": self.created_at,
        }
        pack.metadata["replay_diff"] = diff
        return pack

    def diff(self, other: "ContextCard") -> Dict[str, Any]:
        self_map = {
            item.get("id"): item
            for item in self.memories
            if isinstance(item, dict) and item.get("id") is not None
        }
        other_map = {
            item.get("id"): item
            for item in other.memories
            if isinstance(item, dict) and item.get("id") is not None
        }

        self_ids = set(self_map)
        other_ids = set(other_map)

        new_memories = sorted(other_ids - self_ids)
        removed_memories = sorted(self_ids - other_ids)
        changed_scores = []

        for memory_id in sorted(self_ids & other_ids):
            old_score = _normalize_fact_score(self_map[memory_id].get("score"))
            new_score = _normalize_fact_score(other_map[memory_id].get("score"))
            if old_score is None or new_score is None:
                if old_score != new_score:
                    changed_scores.append({"id": memory_id, "old": old_score, "new": new_score})
                continue
            if old_score != new_score:
                changed_scores.append({
                    "id": memory_id,
                    "old": old_score,
                    "new": new_score,
                    "delta": new_score - old_score,
                })

        return {
            "card_id": self.card_id,
            "other_card_id": other.card_id,
            "query_changed": self.query != other.query,
            "new_facts": new_memories,
            "removed_facts": removed_memories,
            "changed_scores": changed_scores,
            "evidence_delta": len(other.evidence) - len(self.evidence),
            "summary_changed": self.summary != other.summary,
            "graph_delta": {
                "nodes_added": sorted({
                    _node_signature(node)
                    for node in other.graph_slice.get("nodes", [])
                    if node is not None
                } - {
                    _node_signature(node)
                    for node in self.graph_slice.get("nodes", [])
                    if node is not None
                }),
                "nodes_removed": sorted({
                    _node_signature(node)
                    for node in self.graph_slice.get("nodes", [])
                    if node is not None
                } - {
                    _node_signature(node)
                    for node in other.graph_slice.get("nodes", [])
                    if node is not None
                }),
            },
        }


class CardDeck:
    """Collection of ContextCards. Shareable as a bundle."""

    def __init__(self, cards: Optional[Dict[str, Any]] = None, *, synapse_instance=None):
        self.cards: Dict[str, ContextCard] = {}
        self._synapse = synapse_instance

        if cards:
            self._load(cards)

    def _load(self, cards: Dict[str, Any]) -> None:
        for card_id, payload in cards.items():
            if isinstance(payload, ContextCard):
                self.cards[card_id] = payload
                continue
            if isinstance(payload, dict):
                card = ContextCard.from_dict(payload)
                if card.card_id:
                    self.cards[card.card_id] = card

    def _sync(self, card: ContextCard) -> None:
        if self._synapse is None:
            return
        payload = card.to_dict()
        if card.card_id in self._synapse.store.cards:
            self._synapse.store.update_card(payload)
        else:
            self._synapse.store.insert_card(payload)

    def add(self, card: ContextCard):
        self.cards[card.card_id] = card
        self._sync(card)

    def get(self, card_id: str) -> Optional[ContextCard]:
        return self.cards.get(card_id)

    def search(self, query: str) -> List[ContextCard]:
        needle = (query or "").strip().lower()
        if not needle:
            return sorted(self.cards.values(), key=lambda item: item.created_at, reverse=True)

        matches = []
        for card in self.cards.values():
            haystack = " ".join([
                card.card_id,
                card.query,
                card.summary,
                " ".join([str(memory.get("content", "")) for memory in card.memories if isinstance(memory, dict)]),
            ]).lower()
            if needle in haystack:
                matches.append(card)
        matches.sort(key=lambda item: item.created_at, reverse=True)
        return matches

    def export(self, path: str) -> str:
        with open(path, "wb") as handle:
            handle.write(CARD_FILE_MAGIC)
            handle.write(bytes([CARD_FILE_VERSION]))

            for card in self.cards.values():
                payload = card.to_json().encode("utf-8")
                handle.write(_pack_record(RECORD_DATA, payload))

            handle.write(_pack_record(RECORD_END, b""))
        return path

    def import_deck(self, path: str) -> int:
        with open(path, "rb") as handle:
            data = handle.read()

        if not data.startswith(CARD_FILE_MAGIC):
            raise ValueError("Invalid card deck format")
        if len(data) < len(CARD_FILE_MAGIC) + 1:
            raise ValueError("Card deck missing version")
        if data[len(CARD_FILE_MAGIC)] != CARD_FILE_VERSION:
            raise ValueError("Unsupported card deck version")

        imported = 0
        for record_type, payload in _iter_records(data, start=len(CARD_FILE_MAGIC) + 1):
            if record_type != RECORD_DATA:
                continue
            card = ContextCard.from_dict(json.loads(payload.decode("utf-8")))
            self.add(card)
            imported += 1

        return imported
