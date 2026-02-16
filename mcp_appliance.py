#!/usr/bin/env python3
"""Lean MCP server with a curated appliance-level tool surface."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from entity_graph import extract_concepts
from synapse import Synapse

from mcp_server import (
    _active_memory_count,
    _as_int,
    _error_payload,
    _json_dumps,
    _memory_to_dict,
    _ok_payload,
)

LOG = logging.getLogger("synapse-mcp-appliance")

APPLIANCE_TOOL_NAMES = {
    "remember",
    "compile_context",
    "timeline",
    "what_changed",
    "contradictions",
    "fact_history",
    "sleep",
    "stats",
}

RANGE_SECONDS = {
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
    "all": None,
}


def _tool_schema_remember() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string", "minLength": 1, "description": "Memory text to store."},
            "kind": {"type": "string", "enum": ["event", "fact", "doc_ref"], "default": "fact"},
            "metadata": {"type": "object", "default": {}, "description": "Arbitrary JSON metadata."},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["content"],
    }


def _tool_schema_compile_context() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1, "description": "Query / task to compile context for."},
            "budget_tokens": {"type": "integer", "minimum": 1, "maximum": 200000, "default": 2000},
            "mode": {"type": "string", "enum": ["balanced", "precise", "broad", "temporal"], "default": "balanced"},
        },
        "required": ["query"],
    }


def _tool_schema_timeline() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "range": {"type": "string", "enum": ["7d", "30d", "all"], "default": "7d"},
            "topic": {"type": "string", "description": "Optional concept/topic filter."},
        },
    }


def _tool_schema_what_changed() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "range": {"type": "string", "enum": ["7d", "30d", "all"], "default": "7d"},
            "topic": {"type": "string", "description": "Optional concept/topic filter."},
        },
    }


def _tool_schema_contradictions() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string", "description": "Optional concept filter."},
            "entity": {"type": "string", "description": "Optional entity filter."},
        },
    }


def _tool_schema_fact_history() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subject": {"type": "string", "description": "Subject to anchor fact history."},
            "relation": {"type": "string", "description": "Optional predicate/object filter."},
        },
        "required": ["subject"],
    }


def _tool_schema_sleep() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "consolidate": {"type": "boolean", "default": True},
            "prune": {"type": "boolean", "default": True},
        },
    }


def _tool_schema_stats() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}}


def _tool_result(payload: Dict[str, Any]) -> types.TextContent:
    return types.TextContent(type="text", text=_json_dumps(_ok_payload(payload)))


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("tags must be an array of strings")

    out: List[str] = []
    for tag in value:
        if not isinstance(tag, str):
            raise TypeError("tags must be an array of strings")
        normalized = tag.strip().lower()
        if not normalized:
            continue
        if normalized not in out:
            out.append(normalized)
    return out


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _memory_preview(content: str, max_len: int = 420) -> str:
    normalized = (content or "").strip().replace("\n", " ")
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[:max_len - 3]}..."


def _topic_matches(topic: Optional[str], memory_data: Dict[str, Any], syn: Synapse) -> bool:
    if not topic:
        return True
    target = topic.lower()
    concepts = syn.concept_graph.get_memory_concepts(memory_data.get("id"))
    if isinstance(concepts, set) and target in {c.lower() for c in concepts}:
        return True

    content = (memory_data.get("content", "") or "").lower()
    if target in content:
        return True

    for concept_name, _ in extract_concepts(content):
        if (concept_name or "").lower() == target:
            return True
    return False


def _to_memory_dict(syn: Synapse, memory_id: int) -> Dict[str, Any]:
    memory_data = syn.store.memories.get(memory_id)
    if memory_data is None:
        return {}
    return _memory_to_dict(syn._memory_data_to_object(memory_data))


def _collect_contradictions(syn: Synapse, topic: Optional[str] = None, entity: Optional[str] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in syn.contradictions() or []:
        payload = _serialize(row)
        a_id = payload.get("memory_id_a")
        b_id = payload.get("memory_id_b")
        a_memory = _to_memory_dict(syn, a_id) if isinstance(a_id, int) else {}
        b_memory = _to_memory_dict(syn, b_id) if isinstance(b_id, int) else {}

        if not a_memory and not b_memory:
            continue
        if topic:
            if not _topic_matches(topic, a_memory, syn) and not _topic_matches(topic, b_memory, syn):
                continue
        if entity:
            if not _topic_matches(entity, a_memory, syn) and not _topic_matches(entity, b_memory, syn):
                continue

        payload["memory_a"] = _memory_preview(a_memory.get("content", ""), 120)
        payload["memory_b"] = _memory_preview(b_memory.get("content", ""), 120)
        payload["memory_a_id"] = a_id
        payload["memory_b_id"] = b_id
        items.append(payload)
    return items


def _range_cutoff(range_value: str) -> Optional[float]:
    seconds = RANGE_SECONDS.get(range_value)
    if seconds is None:
        return None
    return time.time() - float(seconds)


def _normalize_entities(syn: Synapse, content: str) -> List[str]:
    seen = {name for name, _ in extract_concepts(content)}
    normalizer = getattr(syn, "_entity_normalizer", None)
    canonical = getattr(normalizer, "canonical", None)
    if not callable(canonical):
        return sorted(seen)

    output: List[str] = []
    for concept in sorted(seen):
        normalized = canonical(concept)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _belief_versions_since(syn: Synapse, cutoff: Optional[float], topic: Optional[str]) -> List[Dict[str, Any]]:
    all_versions = getattr(syn.belief_tracker, "_versions", {})
    result: List[Dict[str, Any]] = []
    for version_list in (all_versions or {}).values():
        for version in version_list:
            valid_from = float(getattr(version, "valid_from", 0.0) or 0.0)
            if cutoff is not None and valid_from < cutoff:
                continue
            fact_key = str(getattr(version, "fact_key", ""))
            if topic and topic.lower() not in fact_key.lower():
                continue

            item = _serialize(version)
            memory_id = item.get("memory_id")
            item["memory"] = _to_memory_dict(syn, memory_id) if isinstance(memory_id, int) else {}
            result.append(item)

    return sorted(result, key=lambda i: float(i.get("valid_from", 0.0)))


def _run_sleep_cycle(syn: Synapse, consolidate: bool, prune: bool) -> Dict[str, Any]:
    if not consolidate and not prune:
        return {
            "sleep_report": {
                "consolidated": 0,
                "promoted": 0,
                "patterns_found": 0,
                "contradictions": 0,
                "pruned": 0,
                "graph_cleaned": 0,
                "duration_ms": 0.0,
                "details": {"status": "skipped", "reason": "no_maintenance_selected"},
            }
        }

    if getattr(syn, "_is_sleeping", False):
        return {
            "sleep_report": {
                "consolidated": 0,
                "promoted": 0,
                "patterns_found": 0,
                "contradictions": 0,
                "pruned": 0,
                "graph_cleaned": 0,
                "duration_ms": 0.0,
                "details": {"status": "skipped", "reason": "sleep_in_progress"},
            }
        }

    start = time.time()
    runner = getattr(syn, "sleep_runner", None)
    details: Dict[str, Any] = {}
    syn._is_sleeping = True

    try:
        consolidated = 0
        promoted = 0
        patterns_found = 0
        pruned_count = 0
        graph_cleaned = 0
        contradiction_count = 0

        if consolidate and runner is not None:
            groups = syn.consolidate()
            consolidated = sum(int(item.get("source_count", 0)) for item in groups if isinstance(item, dict))
            promoted = int(runner._promote_to_semantic())
            patterns_found = int(runner._mine_patterns())
            details["promote_patterns"] = {
                "promoted": promoted,
                "patterns_found": patterns_found,
            }
            details["consolidated_groups"] = len(groups)

        if prune:
            pruned_ids = syn.prune(min_strength=0.1, min_access=0, max_age_days=90, dry_run=False)
            pruned_count = len(pruned_ids)
            details["pruned_count"] = pruned_count

        if consolidate or prune:
            if runner is not None:
                contradiction_count = len(runner.synapse.contradiction_detector.scan_memories(runner._active_memories()))
                graph_cleaned = int(runner._cleanup_graph())
            else:
                contradiction_count = len(syn.contradictions())
            details["contradictions"] = contradiction_count
            details["graph_cleaned"] = graph_cleaned

        if (consolidate or prune):
            syn.flush()
            syn._last_sleep_at = time.time()

        return {
            "sleep_report": {
                "consolidated": consolidated,
                "promoted": promoted,
                "patterns_found": patterns_found,
                "contradictions": contradiction_count,
                "pruned": pruned_count,
                "graph_cleaned": graph_cleaned,
                "duration_ms": (time.time() - start) * 1000.0,
                "details": details,
            }
        }
    finally:
        syn._is_sleeping = False


def _store_size_bytes(db_path: str) -> int:
    if not db_path or db_path == ":memory:":
        return 0
    total = 0
    for suffix in (".log", ".snapshot", ".state"):
        candidate = f"{db_path}{suffix}"
        if not os.path.exists(candidate):
            continue
        try:
            total += int(os.path.getsize(candidate))
        except OSError:
            continue
    return total


def _drift_indicators(syn: Synapse, active_count: int) -> Dict[str, Any]:
    now = time.time()
    last_sleep = syn._last_sleep_at
    hook = syn.sleep_runner.schedule_hook()
    recent_cutoff = now - 24 * 60 * 60

    recent_count = 0
    for memory_data in syn.store.memories.values():
        if memory_data.get("consolidated", False):
            continue
        if (memory_data.get("created_at") or 0.0) >= recent_cutoff:
            recent_count += 1

    return {
        "recent_activity_ratio": (recent_count / active_count) if active_count else 0.0,
        "contradiction_density": (len(syn.contradictions()) / active_count) if active_count else 0.0,
        "sleep_overdue_seconds": (now - last_sleep) if last_sleep is not None else None,
        "sleep_due": bool(hook.get("should_sleep")),
    }


def _build_server(*, syn: Synapse) -> Tuple[Server, asyncio.Lock]:
    server = Server("synapse-memory-appliance")
    db_lock = asyncio.Lock()

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="remember",
                description="Store memory with smart preprocessing and contradiction-aware indexing.",
                inputSchema=_tool_schema_remember(),
            ),
            types.Tool(
                name="compile_context",
                description="Compile a context pack from memories (LLM-ready, with summaries and evidence).",
                inputSchema=_tool_schema_compile_context(),
            ),
            types.Tool(
                name="timeline",
                description="Get chronological memory timeline for a window.",
                inputSchema=_tool_schema_timeline(),
            ),
            types.Tool(
                name="what_changed",
                description="Summarize new memories, belief changes, and resolved contradictions.",
                inputSchema=_tool_schema_what_changed(),
            ),
            types.Tool(
                name="contradictions",
                description="List unresolved contradictions.",
                inputSchema=_tool_schema_contradictions(),
            ),
            types.Tool(
                name="fact_history",
                description="Show how a fact evolved over time.",
                inputSchema=_tool_schema_fact_history(),
            ),
            types.Tool(
                name="sleep",
                description="Run maintenance cycle (consolidate/prune controls available).",
                inputSchema=_tool_schema_sleep(),
            ),
            types.Tool(
                name="stats",
                description="Get memory store health metrics and drift indicators.",
                inputSchema=_tool_schema_stats(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
        args = arguments or {}

        try:
            if name == "remember":
                content = _as_text(args.get("content"))
                if not content:
                    raise ValueError("content must be a non-empty string")

                kind = _as_text(args.get("kind", "fact"))
                if kind not in {"event", "fact", "doc_ref"}:
                    raise ValueError("kind must be one of: event, fact, doc_ref")

                metadata = args.get("metadata", None)
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object (JSON dictionary)")

                raw_tags = _as_tags(args.get("tags"))
                normalized_tags = sorted(set(_normalize_entities(syn, content)) | set(raw_tags))
                memory_type = "event" if kind == "event" else "observation" if kind == "doc_ref" else "fact"

                payload_metadata = dict(metadata)
                payload_metadata.update({"kind": kind, "tags": normalized_tags, "source": "appliance"})

                async with db_lock:
                    memory = syn.remember(
                        content=content,
                        memory_type=memory_type,
                        metadata=payload_metadata,
                        extract=False,
                    )
                    syn.flush()

                return [_tool_result({
                    "memory": _memory_to_dict(memory),
                    "memory_type": memory_type,
                    "auto_normalized_tags": normalized_tags,
                    "normalized_contradictions": _collect_contradictions(syn, entity=content),
                })]

            if name == "compile_context":
                query = _as_text(args.get("query"))
                if not query:
                    raise ValueError("query is required")

                budget = _as_int(args.get("budget_tokens", 2000), field="budget_tokens")
                if budget <= 0:
                    raise ValueError("budget_tokens must be > 0")

                mode = _as_text(args.get("mode", "balanced"))
                if mode not in {"balanced", "precise", "broad", "temporal"}:
                    raise ValueError("mode must be balanced, precise, broad, or temporal")

                async with db_lock:
                    pack = syn.compile_context(query=query, budget=budget, policy=mode)

                return [_tool_result({
                    "context_text": pack.to_system_prompt(),
                    "compact": pack.to_compact(),
                    "context_pack": {
                        "query": pack.query,
                        "memories": pack.memories,
                        "graph_slice": pack.graph_slice,
                        "summaries": pack.summaries,
                        "evidence": pack.evidence,
                        "metadata": pack.metadata,
                        "budget_used": pack.budget_used,
                        "budget_total": pack.budget_total,
                    },
                    "relevant_memories": pack.memories,
                    "graph_slice_summary": pack.graph_slice,
                    "evidence": pack.evidence,
                    "contradiction_notes": _collect_contradictions(syn),
                })]

            if name == "timeline":
                range_value = _as_text(args.get("range", "7d"))
                if range_value not in RANGE_SECONDS:
                    raise ValueError("range must be one of: 7d, 30d, all")
                topic = _as_text(args.get("topic", ""))
                cutoff = _range_cutoff(range_value)

                async with db_lock:
                    entries = []
                    for item in syn.timeline(topic=topic.lower() if topic else None):
                        memory = item.get("memory")
                        if memory is None:
                            continue
                        if cutoff is not None and (item.get("timestamp") or 0.0) < cutoff:
                            continue
                        if topic and not _topic_matches(topic, _memory_to_dict(memory), syn):
                            continue
                        entry = _memory_to_dict(memory)
                        entry["supersedes"] = item.get("supersedes")
                        entry["superseded_by"] = item.get("superseded_by")
                        entry["fact_chain_id"] = item.get("fact_chain_id")
                        entry["timestamp"] = item.get("timestamp")
                        entries.append(entry)
                entries.sort(key=lambda item: item.get("timestamp", 0.0))

                return [_tool_result({
                    "range": range_value,
                    "topic": topic or None,
                    "timeline": entries,
                    "count": len(entries),
                })]

            if name == "what_changed":
                range_value = _as_text(args.get("range", "7d"))
                if range_value not in RANGE_SECONDS:
                    raise ValueError("range must be one of: 7d, 30d, all")
                topic = _as_text(args.get("topic", ""))
                cutoff = _range_cutoff(range_value)

                new_facts: List[Dict[str, Any]] = []
                for memory_data in syn.store.memories.values():
                    if memory_data.get("consolidated", False):
                        continue
                    if cutoff is not None and (memory_data.get("created_at") or 0.0) < cutoff:
                        continue
                    if topic and not _topic_matches(topic, memory_data, syn):
                        continue
                    new_facts.append(_memory_to_dict(syn._memory_data_to_object(memory_data)))
                new_facts.sort(key=lambda item: float(item.get("created_at", 0.0)))

                changed_beliefs = _belief_versions_since(syn, cutoff, topic or None)
                resolved = [
                    version
                    for version in changed_beliefs
                    if "contradiction" in str(version.get("reason", "")).lower()
                ]

                return [_tool_result({
                    "range": range_value,
                    "topic": topic or None,
                    "new_facts": new_facts,
                    "changed_beliefs": changed_beliefs,
                    "resolved_contradictions": resolved,
                })]

            if name == "contradictions":
                topic = _as_text(args.get("topic", ""))
                entity = _as_text(args.get("entity", ""))
                items = _collect_contradictions(syn, topic=topic or None, entity=entity or None)
                return [_tool_result({
                    "contradictions": items,
                    "count": len(items),
                })]

            if name == "fact_history":
                subject = _as_text(args.get("subject"))
                if not subject:
                    raise ValueError("subject is required")

                relation = _as_text(args.get("relation", ""))
                async with db_lock:
                    history = syn.fact_history(subject)

                chain: List[Dict[str, Any]] = []
                for item in history:
                    memory = item.get("memory")
                    if memory is None:
                        continue
                    if relation and relation.lower() not in str(memory.content).lower():
                        continue
                    chain.append({
                        "version": int(item.get("version", 0)),
                        "memory": _memory_to_dict(memory),
                        "current": bool(item.get("current")),
                    })

                return [_tool_result({
                    "subject": subject,
                    "relation": relation or None,
                    "chain": chain,
                    "count": len(chain),
                })]

            if name == "sleep":
                consolidate = bool(args.get("consolidate", True))
                prune = bool(args.get("prune", True))
                async with db_lock:
                    result = _run_sleep_cycle(syn, consolidate=consolidate, prune=prune)
                return [_tool_result(result)]

            if name == "stats":
                active_count = _active_memory_count(syn)
                hot_concepts = [
                    {"name": name, "score": score}
                    for name, score in syn.hot_concepts(k=10)
                ]
                return [_tool_result({
                    "memory_count": active_count,
                    "concept_count": len(syn.concept_graph.concepts),
                    "hot_concepts": hot_concepts,
                    "contradiction_count": len(syn.contradictions()),
                    "store_size_bytes": _store_size_bytes(syn.path),
                    "last_sleep": syn._last_sleep_at,
                    "drift_indicators": _drift_indicators(syn, active_count),
                })]

            payload = _error_payload(ValueError(f"Unknown tool: {name}"), code="unknown_tool")
            return [types.TextContent(type="text", text=_json_dumps(payload))]

        except Exception as exc:
            LOG.exception("Tool call failed: %s", name)
            return [types.TextContent(type="text", text=_json_dumps(_error_payload(exc)))]

    return server, db_lock


async def _run_server(*, data_dir: str) -> None:
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    db_prefix = os.path.join(data_dir, "synapse")

    syn = Synapse(db_prefix)
    if os.environ.get("SYNAPSE_MCP_ENABLE_EMBEDDINGS", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        syn._use_embeddings = False

    server, _db_lock = _build_server(syn=syn)
    LOG.info("Synapse AI Memory MCP appliance starting. data_dir=%s db_prefix=%s", data_dir, db_prefix)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Synapse MCP appliance server.")
    parser.add_argument(
        "--data-dir",
        default="~/.synapse",
        help="Directory to store Synapse AI Memory files (default: ~/.synapse).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        asyncio.run(_run_server(data_dir=args.data_dir))
    except KeyboardInterrupt:
        return 0
    except Exception:
        LOG.exception("Fatal appliance server error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

