"""Brain pack primitives and helpers for topic-scoped shareable snapshots."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from context_pack import ContextCompiler, ContextPack
from entity_graph import extract_concepts


_DAY_SECONDS = 60 * 60 * 24
_MAX_FILENAME_LEN = 70
_CHECKSUM_PREFIX = "sha256:"
_STORE_DIR_NAME = ".brain_packs"


def _format_datetime(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    slug = slug.strip("-")
    if not slug:
        return "topic"
    return slug[:_MAX_FILENAME_LEN] or "topic"


def parse_range_days(value: Any) -> int:
    """Parse CLI-friendly range expressions such as ``30d`` or ``2w``."""
    if isinstance(value, bool):
        raise ValueError("range must be numeric, not boolean")
    if value is None:
        return 30
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))

    raw = str(value).strip().lower().replace(" ", "")
    if not raw:
        return 30

    match = re.fullmatch(r"(\d+)([dhwmqy]?)", raw)
    if not match:
        raise ValueError(f"Invalid range value: {value!r}")
    count = int(match.group(1))
    suffix = match.group(2) or "d"
    multipliers = {
        "d": 1,
        "h": 1 / 24,
        "w": 7,
        "m": 30,
        "q": 90,
        "y": 365,
    }
    multiplier = multipliers.get(suffix, 1)
    return max(1, int(count * multiplier))


def get_pack_directory(db_path: Optional[str] = None, *, explicit: Optional[str] = None) -> str:
    """Resolve the pack directory for a DB or explicit override."""
    if explicit:
        return os.path.abspath(explicit)

    if not db_path or db_path in {":memory:", ""}:
        return os.path.abspath(_STORE_DIR_NAME)

    return os.path.join(os.path.dirname(os.path.abspath(db_path)), _STORE_DIR_NAME)


def list_pack_files(store_dir: str) -> list[str]:
    if not os.path.isdir(store_dir):
        return []
    return sorted(
        os.path.join(store_dir, item)
        for item in os.listdir(store_dir)
        if item.endswith(".brain")
    )


def default_pack_output_path(topic: str, db_path: Optional[str] = None, *,
                            created_at: Optional[float] = None,
                            explicit_store: Optional[str] = None) -> str:
    store_dir = get_pack_directory(db_path=db_path, explicit=explicit_store)
    os.makedirs(store_dir, exist_ok=True)
    stamp = int(created_at or time.time())
    filename = f"{_slugify(topic)}-{stamp}.brain"
    return os.path.join(store_dir, filename)


def _memory_signature(text: str) -> str:
    normalized = (text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _serialize_memory(memory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "content": memory.content,
        "memory_type": memory.memory_type,
        "memory_level": memory.memory_level,
        "strength": memory.strength,
        "created_at": memory.created_at,
        "last_accessed": memory.last_accessed,
        "observed_at": memory.observed_at,
        "valid_from": memory.valid_from,
        "valid_to": memory.valid_to,
        "metadata": memory.metadata,
        "consolidated": memory.consolidated,
        "summary_of": memory.summary_of,
        "signature": _memory_signature(memory.content),
    }


def _serialize_contradiction(contradiction, *, memory_lookup: Dict[int, str]) -> Dict[str, Any]:
    left = memory_lookup.get(contradiction.memory_id_a, "")
    right = memory_lookup.get(contradiction.memory_id_b, "")
    return {
        "memory_id_a": contradiction.memory_id_a,
        "memory_id_b": contradiction.memory_id_b,
        "kind": contradiction.kind,
        "description": contradiction.description,
        "confidence": contradiction.confidence,
        "detected_at": contradiction.detected_at,
        "memory_content_a": left,
        "memory_content_b": right,
        "memory_signature_a": _memory_signature(left),
        "memory_signature_b": _memory_signature(right),
    }


def _serialize_belief(fact_key: str, version: Any) -> Dict[str, Any]:
    return {
        "fact_key": fact_key,
        "value": version.value,
        "memory_id": version.memory_id,
        "valid_from": version.valid_from,
        "valid_to": version.valid_to,
        "reason": version.reason,
        "confidence": version.confidence,
    }


def _read_context_pack(payload: Optional[Dict[str, Any]]) -> Optional[ContextPack]:
    if not payload:
        return None
    return ContextPack(
        query=payload.get("query", ""),
        memories=payload.get("memories", []),
        graph_slice=payload.get("graph_slice", {"nodes": [], "edges": [], "concepts": []}),
        summaries=payload.get("summaries", []),
        evidence=payload.get("evidence", []),
        budget_used=payload.get("budget_used", 0),
        budget_total=payload.get("budget_total", 0),
        metadata=payload.get("metadata", {}),
    )


def _collect_memories_for_topic(synapse, topic: str, *, range_days: int, until: Optional[float] = None) -> List[Any]:
    if until is None:
        until = time.time()
    cutoff = until - (range_days * _DAY_SECONDS)

    topic_text = (topic or "").strip().lower()
    topic_concepts = {_name for _name, _category in extract_concepts(topic_text)}
    all_objects = [
        synapse._memory_data_to_object(memory_data)
        for memory_data in synapse.store.memories.values()
        if not memory_data.get("consolidated", False)
    ]
    try:
        recall_ids = {
            memory.id
            for memory in synapse.recall(context=topic, limit=max(1, len(all_objects)))
            if memory.id is not None
        }
    except Exception:
        recall_ids = set()

    selected: List[Any] = []
    for memory in all_objects:
        if memory.created_at is None or memory.created_at < cutoff:
            continue

        text = (memory.content or "").lower()
        concepts = {
            str(name).lower()
            for name in synapse.concept_graph.get_memory_concepts(memory.id)
        } if memory.id is not None else set()
        matches_topic = False
        if memory.id in recall_ids:
            matches_topic = True
        elif topic_text and topic_text in text:
            matches_topic = True
        elif topic_concepts and concepts.intersection(topic_concepts):
            matches_topic = True

        if matches_topic:
            selected.append(memory)

    selected.sort(key=lambda item: item.created_at)
    return selected


def _fingerprint_memory(memory_record: Dict[str, Any]) -> str:
    return str(memory_record.get("signature") or _memory_signature(memory_record.get("content", "")))


def _fingerprint_belief(key: str, version: Dict[str, Any]) -> str:
    return f"{key}|{version.get('value', '')}"


def _checksum_of_payload(payload: Dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("checksum", None)
    serialized = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _replay_report_markdown(pack: "BrainPack", *, report: Dict[str, Any]) -> str:
    lines = [f"# 🧠 Brain Pack Replay: {pack.topic}", ""]
    match_count = len(report.get("memory_matches", []))
    missing_count = len(report.get("memory_missing", []))
    extra_count = len(report.get("memory_extra", []))
    changed_count = len(report.get("memory_changed", []))
    lines.append(f"**Topic:** `{pack.topic}`")
    lines.append(f"**Range:** last {pack.range_days} days")
    lines.append(f"**Summary:** matched={match_count} | missing={missing_count} | changed={changed_count} | extra={extra_count}")
    lines.append("")

    lines.append("## Memory Match Report")
    for memory in report.get("memory_matches", []):
        lines.append(f"- ✅ `{memory.get('signature')}`: {memory.get('content', '')}")
    for memory in report.get("memory_changed", []):
        lines.append(f"- ⚠️ changed: `{memory.get('signature')}`")
    for memory in report.get("memory_missing", []):
        lines.append(f"- ❌ missing: `{memory.get('signature')}`")
    if report.get("memory_extra"):
        lines.append("")
        lines.append("## Memories Present Locally, Not in Pack")
        for memory in report.get("memory_extra", []):
            lines.append(f"- +++ `{memory.get('signature')}` {memory.get('content')}")

    lines.append("")
    lines.append("## Belief Diff")
    for item in report.get("beliefs_missing", []):
        lines.append(f"- ❌ Missing belief: `{item['fact_key']}`")
    for item in report.get("beliefs_changed", []):
        lines.append(
            f"- ⚠️ Changed belief: `{item['fact_key']}`"
            f" pack={item['pack_value']} now={item['current_value']}"
        )
    if report.get("beliefs_new", []):
        lines.append("")
        lines.append("## Beliefs New in This Synapse")
        for item in report["beliefs_new"]:
            lines.append(f"- +++ `{item['fact_key']}`")

    lines.append("")
    lines.append("## Contradiction Diff")
    for item in report.get("contradictions_missing", []):
        lines.append(f"- ❌ Missing contradiction: `{item['kind']} {item['memory_signature_a'][:12]} / {item['memory_signature_b'][:12]}`")
    if report.get("contradictions_extra", []):
        lines.append("")
        lines.append("## Contradictions New in This Synapse")
        for item in report["contradictions_extra"]:
            lines.append(f"- +++ `{item['kind']} {item['memory_signature_a'][:12]} / {item['memory_signature_b'][:12]}`")

    return "\n".join(lines) + "\n"


def _diff_markdown(left: "BrainPack", right: "BrainPack", report: Dict[str, Any]) -> str:
    lines = [f"# 🧠 Brain Pack Diff: {left.topic} ↔ {right.topic}", ""]
    if left.topic != right.topic:
        lines.append(f"- **Topics:** `{left.topic}` → `{right.topic}`")
    lines.append(f"- **Range:** {left.range_days}d vs {right.range_days}d")
    lines.append(f"- **Created:** `{_format_datetime(left.created_at)}` ↔ `{_format_datetime(right.created_at)}`")
    lines.append("")

    lines.append("## Memory Changes")
    lines.append(f"- Added in new: {len(report.get('memory_added', []))}")
    lines.append(f"- Removed in new: {len(report.get('memory_removed', []))}")
    lines.append(f"- Changed in both: {len(report.get('memory_changed', []))}")
    lines.append("")

    if report.get("memory_added"):
        lines.append("### Added")
        for item in report["memory_added"]:
            lines.append(f"- +++ {item.get('signature')}: {item.get('content')}")
    if report.get("memory_removed"):
        lines.append("### Removed")
        for item in report["memory_removed"]:
            lines.append(f"- --- {item.get('signature')}: {item.get('content')}")
    if report.get("memory_changed"):
        lines.append("### Changed")
        for item in report["memory_changed"]:
            lines.append(f"- ⚠️ {item.get('signature')}: {item.get('left_content')} -> {item.get('right_content')}")

    lines.append("")
    lines.append("## Belief Changes")
    lines.append(f"- Added facts: {len(report.get('beliefs_added', []))}")
    lines.append(f"- Removed facts: {len(report.get('beliefs_removed', []))}")
    lines.append(f"- Changed facts: {len(report.get('beliefs_changed', []))}")
    if report.get("beliefs_changed"):
        lines.append("")
        lines.append("### Changed beliefs")
        for item in report["beliefs_changed"]:
            lines.append(
                f"- {item.get('fact_key')}: '{item.get('left_value')}' -> '{item.get('right_value')}'"
            )

    lines.append("")
    lines.append("## Contradiction Changes")
    lines.append(f"- Added contradictions: {len(report.get('contradictions_added', []))}")
    lines.append(f"- Removed contradictions: {len(report.get('contradictions_removed', []))}")

    return "\n".join(lines) + "\n"


class BrainPack:
    """A self-contained topic pack — context and evidence for one subject area."""

    def __init__(self, topic: str, range_days: int = 30):
        self.topic = topic
        self.range_days = max(1, int(range_days))
        self.memories: List[Dict[str, Any]] = []
        self.context_pack: Optional[ContextPack] = None
        self.belief_snapshot: Dict[str, Dict[str, Any]] = {}
        self.contradictions: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []
        self.graph_slice: Dict[str, Any] = {}
        self.created_at = time.time()
        self.checksum = ""

    def build(self, synapse) -> "BrainPack":
        matched_memories = _collect_memories_for_topic(
            synapse,
            self.topic,
            range_days=self.range_days,
            until=self.created_at,
        )
        self.memories = [_serialize_memory(memory) for memory in matched_memories]

        compiler = ContextCompiler(synapse)
        memory_records = [compiler._memory_record(memory) for memory in matched_memories]
        ids = [memory.id for memory in matched_memories if memory.id is not None]
        concept_names = set()
        if ids:
            graph_slice = compiler._extract_graph_slice(ids, depth=2)
            concept_names = {
                item.get("name") for item in graph_slice.get("concepts", []) if isinstance(item, dict) and item.get("name")
            }
            summaries = compiler._generate_summaries(memory_records, concept_names)
            evidence = compiler._build_evidence_chains(matched_memories)
            context_pack = compiler._pack_to_budget(
                query=self.topic,
                memories=memory_records,
                graph_slice=graph_slice,
                summaries=summaries,
                evidence=evidence,
                budget=20000,
                policy="balanced",
            )
            context_pack.metadata.update({
                "topic": self.topic,
                "range_days": self.range_days,
                "range_end_at": self.created_at,
            })
            self.context_pack = context_pack
            self.graph_slice = graph_slice
        else:
            self.context_pack = ContextPack(
                query=self.topic,
                memories=[],
                graph_slice={"nodes": [], "edges": [], "concepts": []},
                summaries=["No matching memories found."],
                evidence=[],
                budget_used=0,
                budget_total=20000,
                metadata={"topic": self.topic, "range_days": self.range_days, "range_end_at": self.created_at},
            )
            self.graph_slice = self.context_pack.graph_slice

        self.belief_snapshot = {}
        belief_data = synapse.beliefs()
        topic_lc = (self.topic or "").lower()
        topic_memory_ids = {memory["id"] for memory in self.memories if isinstance(memory.get("id"), int)}
        for fact_key, version in belief_data.items():
            value = getattr(version, "value", "") or ""
            memory_id = getattr(version, "memory_id", None)
            if fact_key.lower().find(topic_lc) >= 0 or value.lower().find(topic_lc) >= 0 or (memory_id in topic_memory_ids):
                self.belief_snapshot[fact_key] = _serialize_belief(fact_key, version)

        memory_lookup = {
            memory_id: memory_data.get("content", "")
            for memory_id, memory_data in synapse.store.memories.items()
        }
        detected_contradictions = list(synapse.contradictions() or [])
        if not detected_contradictions:
            active_memories = [
                synapse._memory_data_to_object(memory_data)
                for memory_data in synapse.store.memories.values()
                if not memory_data.get("consolidated", False)
            ]
            detected_contradictions = synapse.contradiction_detector.scan_memories(active_memories)
        self.contradictions = [
            _serialize_contradiction(contradiction, memory_lookup=memory_lookup)
            for contradiction in detected_contradictions
            if (
                contradiction.memory_id_a in topic_memory_ids
                or contradiction.memory_id_b in topic_memory_ids
                or topic_lc in (memory_lookup.get(contradiction.memory_id_a, "").lower())
                or topic_lc in (memory_lookup.get(contradiction.memory_id_b, "").lower())
            )
        ]

        self.timeline = []
        for item in matched_memories:
            meta = synapse.store.memories.get(item.id, {})
            event = {
                "type": "memory",
                "memory_id": item.id,
                "timestamp": item.created_at,
                "content": item.content,
                "memory_type": item.memory_type,
                "signature": _memory_signature(item.content),
                "supersedes": meta.get("supersedes"),
                "superseded_by": meta.get("superseded_by"),
            }
            self.timeline.append(event)
        for entry in synapse.timeline(self.topic):
            memory = entry.get("memory")
            if memory is None:
                continue
            if not memory.content:
                continue
            if memory.id in {item.get("memory_id") for item in self.timeline}:
                continue
            if memory.created_at < self.created_at - (self.range_days * _DAY_SECONDS):
                continue
            self.timeline.append({
                "type": "fact_chain",
                "memory_id": memory.id,
                "timestamp": memory.created_at,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "signature": _memory_signature(memory.content),
                "supersedes": entry.get("supersedes"),
                "superseded_by": entry.get("superseded_by"),
            })
        self.timeline.sort(key=lambda item: item.get("timestamp", 0.0))

        if self.graph_slice:
            self.graph_slice = {
                "nodes": self.graph_slice.get("nodes", []),
                "edges": self.graph_slice.get("edges", []),
                "concepts": self.graph_slice.get("concepts", []),
                "seed_memory_ids": self.graph_slice.get("seed_memory_ids", []),
                "depth": self.graph_slice.get("depth", 2),
            }
        self.checksum = _checksum_of_payload(self.to_dict())
        return self

    def save(self, path: str):
        payload = self.to_dict()
        self.checksum = _checksum_of_payload(payload)
        payload["checksum"] = self.checksum
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"{_CHECKSUM_PREFIX} {self.checksum}\n")
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "BrainPack":
        with open(path, "r", encoding="utf-8") as file:
            raw = file.read().splitlines()
        if not raw:
            raise ValueError(f"Empty pack file: {path}")

        header = raw[0].strip()
        if not header.startswith(_CHECKSUM_PREFIX):
            raise ValueError(f"Missing checksum header: {path}")
        header_checksum = header[len(_CHECKSUM_PREFIX):].strip()
        if not re.fullmatch(r"[0-9a-f]{64}", header_checksum):
            raise ValueError(f"Malformed checksum header: {path}")

        payload = json.loads("\n".join(raw[1:]) or "{}")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid pack payload: {path}")
        computed = _checksum_of_payload(payload)
        if payload.get("checksum") and payload.get("checksum") != computed:
            raise ValueError(f"Checksum mismatch: {path}")
        if header_checksum != computed:
            raise ValueError(f"Checksum header mismatch: {path}")

        pack = cls(topic=payload.get("topic", ""))
        pack.range_days = int(payload.get("range_days", 30))
        pack.memories = list(payload.get("memories", []))
        pack.context_pack = _read_context_pack(payload.get("context_pack"))
        pack.belief_snapshot = payload.get("belief_snapshot", {})
        pack.contradictions = payload.get("contradictions", [])
        pack.timeline = payload.get("timeline", [])
        pack.graph_slice = payload.get("graph_slice", {})
        pack.created_at = float(payload.get("created_at", time.time()))
        pack.checksum = computed
        return pack

    def replay(self, synapse) -> Dict[str, Any]:
        current = _collect_memories_for_topic(
            synapse,
            self.topic,
            range_days=self.range_days,
            until=self.created_at,
        )
        current_records = [_serialize_memory(memory) for memory in current]
        current_by_sig = {_fingerprint_memory(item): item for item in current_records}

        pack_by_sig = {_fingerprint_memory(item): item for item in self.memories}
        current_signature_set = set(current_by_sig)
        pack_signature_set = set(pack_by_sig)

        memory_matches = []
        memory_changed = []
        memory_missing = []
        for sig, pack_memory in pack_by_sig.items():
            current_memory = current_by_sig.get(sig)
            if not current_memory:
                memory_missing.append(pack_memory)
                continue

            if current_memory.get("content") != pack_memory.get("content"):
                memory_changed.append({
                    "signature": sig,
                    "pack_content": pack_memory.get("content"),
                    "current_content": current_memory.get("content"),
                })
            else:
                memory_matches.append(current_memory)
        memory_extra = [current_by_sig[sig] for sig in (current_signature_set - pack_signature_set)]

        pack_belief_ids = {
            _fingerprint_belief(key, value): {"fact_key": key, "value": value.get("value"), "payload": value}
            for key, value in self.belief_snapshot.items()
        }
        current_beliefs = {}
        topic_lc = (self.topic or "").lower()
        current_beliefs_obj = synapse.beliefs()
        for fact_key, version in current_beliefs_obj.items():
            serialized = _serialize_belief(fact_key, version)
            if topic_lc in fact_key.lower() or topic_lc in str(serialized.get("value", "")).lower():
                current_beliefs[_fingerprint_belief(fact_key, serialized)] = {
                    "fact_key": fact_key,
                    "value": serialized.get("value"),
                    "payload": serialized,
                }

        beliefs_matched = []
        beliefs_changed = []
        beliefs_missing = []
        for key, payload in pack_belief_ids.items():
            current = current_beliefs.get(key)
            if not current:
                beliefs_missing.append({
                    "fact_key": payload["fact_key"],
                    "pack_value": payload["value"],
                })
                continue
            if (
                current["value"] != payload["value"]
                or current.get("confidence") != payload.get("confidence")
            ):
                beliefs_changed.append({
                    "fact_key": payload["fact_key"],
                    "pack_value": payload["value"],
                    "current_value": current["value"],
                })
            else:
                beliefs_matched.append(payload)
        beliefs_new = [item["payload"] for item in current_beliefs.values() if item["fact_key"] not in self.belief_snapshot]

        def _contr_sig(entry: Dict[str, Any]) -> str:
            return f"{entry.get('kind')}:{entry.get('memory_signature_a')}:{entry.get('memory_signature_b')}"

        pack_conflicts = {_contr_sig(item): item for item in self.contradictions}
        current_conflicts = {}
        for contradiction in synapse.contradictions():
            left = synapse.store.memories.get(contradiction.memory_id_a, {}).get("content", "")
            right = synapse.store.memories.get(contradiction.memory_id_b, {}).get("content", "")
            entry = _serialize_contradiction(contradiction, memory_lookup={contradiction.memory_id_a: left, contradiction.memory_id_b: right})
            current_conflicts[_contr_sig(entry)] = entry

        conflicts_missing = [entry for key, entry in pack_conflicts.items() if key not in current_conflicts]
        conflicts_unchanged = [entry for key, entry in pack_conflicts.items() if key in current_conflicts]
        conflicts_extra = [entry for key, entry in current_conflicts.items() if key not in pack_conflicts]

        report = {
            "memory_matches": memory_matches,
            "memory_changed": memory_changed,
            "memory_missing": memory_missing,
            "memory_extra": memory_extra,
            "beliefs_matched": beliefs_matched,
            "beliefs_changed": beliefs_changed,
            "beliefs_missing": beliefs_missing,
            "beliefs_new": beliefs_new,
            "contradictions_unchanged": conflicts_unchanged,
            "contradictions_missing": conflicts_missing,
            "contradictions_extra": conflicts_extra,
        }
        report["markdown"] = _replay_report_markdown(self, report=report)
        return report

    def diff(self, other: "BrainPack") -> Dict[str, Any]:
        left_memories = {_fingerprint_memory(item): item for item in self.memories}
        right_memories = {_fingerprint_memory(item): item for item in other.memories}
        common = set(left_memories).intersection(set(right_memories))
        added = []
        removed = []
        changed = []
        for key in set(right_memories) - set(left_memories):
            added.append(right_memories[key])
        for key in set(left_memories) - set(right_memories):
            removed.append(left_memories[key])
        for key in common:
            left_content = left_memories[key].get("content", "")
            right_content = right_memories[key].get("content", "")
            if left_content != right_content:
                changed.append({
                    "signature": key,
                    "left_content": left_content,
                    "right_content": right_content,
                })

        def _belief_fp(beliefs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            return {_fingerprint_belief(key, value): {"fact_key": key, "payload": value} for key, value in beliefs.items()}

        left_beliefs = _belief_fp(self.belief_snapshot)
        right_beliefs = _belief_fp(other.belief_snapshot)
        fact_common = set(left_beliefs).intersection(set(right_beliefs))
        belief_added = []
        belief_removed = []
        belief_changed = []
        for key in set(right_beliefs) - set(left_beliefs):
            belief_added.append(right_beliefs[key]["payload"])
        for key in set(left_beliefs) - set(right_beliefs):
            belief_removed.append(left_beliefs[key]["payload"])
        for key in fact_common:
            l = left_beliefs[key]["payload"]
            r = right_beliefs[key]["payload"]
            if l.get("value") != r.get("value") or l.get("confidence") != r.get("confidence"):
                belief_changed.append({
                    "fact_key": l.get("fact_key") or r.get("fact_key"),
                    "left_value": l.get("value"),
                    "right_value": r.get("value"),
                })

        def _contr_sig(items: Dict[str, Dict[str, Any]], target: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            result = {}
            for item in target:
                sig = f"{item.get('kind')}:{item.get('memory_signature_a')}:{item.get('memory_signature_b')}"
                result[sig] = item
            return result

        left_conflicts = _contr_sig(self.contradictions, self.contradictions)
        right_conflicts = _contr_sig(other.contradictions, other.contradictions)
        common_contr = set(left_conflicts).intersection(set(right_conflicts))
        conf_added = []
        conf_removed = []
        for key in set(right_conflicts) - set(left_conflicts):
            conf_added.append(right_conflicts[key])
        for key in set(left_conflicts) - set(right_conflicts):
            conf_removed.append(left_conflicts[key])

        for key in common_contr:
            _ = key

        report = {
            "memory_added": added,
            "memory_removed": removed,
            "memory_changed": changed,
            "beliefs_added": belief_added,
            "beliefs_removed": belief_removed,
            "beliefs_changed": belief_changed,
            "contradictions_added": conf_added,
            "contradictions_removed": conf_removed,
        }
        report["markdown"] = _diff_markdown(self, other, report)
        return report

    def to_markdown(self) -> str:
        topic = self.topic or "topic"
        created = _format_datetime(self.created_at)
        summary_lines = [
            f"# 🧠 Brain Pack: {topic}",
            f"**Range:** last {self.range_days} days | **Memories:** {len(self.memories)} | **Created:** {created}",
            "",
            "## Summary",
        ]
        if self.context_pack is None:
            summary_lines.append("No compiled context pack available.")
        else:
            if self.context_pack.summaries:
                summary_lines.append("\n".join(f"- {summary}" for summary in self.context_pack.summaries))
            else:
                summary_lines.append("No summary available.")

        summary_lines.extend([
            "",
            "## Timeline",
            "",
        ])
        if self.timeline:
            for entry in self.timeline:
                summary_lines.append(
                    f"- {entry.get('type', 'memory')} [{_format_datetime(entry['timestamp'])}] "
                    f"#{entry.get('memory_id')}: {entry.get('content', '')}"
                )
        else:
            summary_lines.append("No timeline events collected.")

        summary_lines.extend([
            "",
            "## Current Beliefs",
            "",
        ])
        if self.belief_snapshot:
            for key in sorted(self.belief_snapshot):
                version = self.belief_snapshot[key]
                summary_lines.append(f"- `{key}` -> {version.get('value')}")
        else:
            summary_lines.append("No beliefs in snapshot.")

        summary_lines.extend([
            "",
            "## Contradictions",
            "",
        ])
        if self.contradictions:
            for item in self.contradictions:
                summary_lines.append(
                    f"- {item.get('kind')} [{item.get('memory_id_a')} vs {item.get('memory_id_b')}]"
                )
        else:
            summary_lines.append("No unresolved contradictions found.")

        summary_lines.extend([
            "",
            "## Graph",
            "",
        ])
        if self.graph_slice:
            concepts = self.graph_slice.get("concepts", [])
            if concepts:
                names = [concept.get("name", "") for concept in concepts if isinstance(concept, dict)]
                summary_lines.append("Concepts: " + ", ".join(sorted(names)))
            else:
                summary_lines.append("No concept-level graph data.")
            edges = self.graph_slice.get("edges", [])
            if edges:
                summary_lines.append("Edges:")
                for item in edges[:20]:
                    summary_lines.append(
                        f"- {item.get('source_id')} -[{item.get('relation')}]-> {item.get('target_id')}"
                    )
        else:
            summary_lines.append("No graph slice available.")

        summary_lines.extend(["", f"---", f"Checksum: {self.checksum}"])
        return "\n".join(summary_lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "range_days": self.range_days,
            "created_at": self.created_at,
            "memories": self.memories,
            "context_pack": asdict(self.context_pack) if self.context_pack else None,
            "belief_snapshot": self.belief_snapshot,
            "contradictions": self.contradictions,
            "timeline": self.timeline,
            "graph_slice": self.graph_slice,
            "checksum": self.checksum,
        }
