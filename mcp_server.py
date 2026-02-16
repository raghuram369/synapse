#!/usr/bin/env python3
"""
MCP (Model Context Protocol) server for the Synapse memory database.

This server is designed for Claude Desktop's stdio transport:
it reads JSON-RPC from stdin and writes responses to stdout.

Dependencies:
  - stdlib only
  - mcp (pip install mcp)

Synapse is imported locally from ./synapse.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from exceptions import SynapseValidationError
from synapse import Synapse, MEMORY_TYPES, EDGE_TYPES, Memory, ScoreBreakdown
from egress_guard import EgressGuard


LOG = logging.getLogger("synapse-mcp")
ALLOWED_SCOPES = ("public", "shared", "private")


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s, 10)
    raise ValueError(f"{field} must be an integer")


def _as_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError(f"{field} must be a number")


def _memory_to_dict(m: Memory) -> Dict[str, Any]:
    if is_dataclass(m):
        d = asdict(m)
    else:
        # Fallback for unexpected Memory-like objects.
        d = dict(getattr(m, "__dict__", {}))
    # Include computed field for convenience in clients.
    try:
        d["effective_strength"] = float(getattr(m, "effective_strength"))
    except Exception:
        pass
    return d


def _normalize_scope(
    value: Any,
    *,
    field: str = "scope",
    default: str = "private",
    allow_default: bool = True,
) -> str:
    if value is None:
        if not allow_default:
            raise ValueError(f"{field} is required")
        raw = default
    else:
        raw = value
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    scope = raw.strip().lower()
    if not scope:
        if not allow_default:
            raise ValueError(f"{field} cannot be empty")
        scope = default
    if scope not in ALLOWED_SCOPES:
        raise ValueError(f"{field} must be one of: {', '.join(ALLOWED_SCOPES)}")
    return scope


def _stored_scope(value: Any) -> str:
    if isinstance(value, str):
        scope = value.strip().lower()
        if scope in ALLOWED_SCOPES:
            return scope
    return "private"


def _is_scope_visible(memory_scope: str, requested_scope: str) -> bool:
    return ALLOWED_SCOPES.index(memory_scope) <= ALLOWED_SCOPES.index(requested_scope)


def _normalize_metadata(raw_metadata: Any) -> Dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _filter_text_payload(value: Any, guard: EgressGuard) -> Any:
    if isinstance(value, str):
        return guard.filter_context(value)
    if isinstance(value, list):
        return [_filter_text_payload(item, guard) for item in value]
    if isinstance(value, dict):
        return {k: _filter_text_payload(v, guard) for k, v in value.items()}
    return value


def _error_payload(exc: BaseException, *, code: str = "tool_error") -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "type": exc.__class__.__name__,
            "message": str(exc),
        },
        "ts_ms": _now_ms(),
    }


def _ok_payload(result: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "result": result,
        "ts_ms": _now_ms(),
    }


def _tool_schema_remember() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string", "minLength": 1, "description": "Memory text to store."},
            "memory_type": {"type": "string", "enum": sorted(MEMORY_TYPES), "default": "fact"},
            "metadata": {"type": "object", "default": {}, "description": "Arbitrary JSON metadata."},
            "episode": {"type": "string", "default": "", "description": "Optional episode name/group."},
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
        "required": ["content"],
    }


def _tool_schema_recall() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "default": "", "description": "Search query / context."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            "memory_type": {"type": "string", "enum": sorted(MEMORY_TYPES), "description": "Optional type filter."},
            "explain": {"type": "boolean", "default": False, "description": "Include score breakdown."},
            "show_disputes": {"type": "boolean", "default": False, "description": "Include unresolved contradiction disputes per memory."},
            "exclude_conflicted": {"type": "boolean", "default": False, "description": "Filter out memories with unresolved contradictions."},
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
        "required": ["query"],
    }


def _tool_schema_list() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "sort": {"type": "string", "enum": ["recent", "created", "access_count"], "default": "recent"},
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
    }


def _tool_schema_browse() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concept": {"type": "string", "description": "Concept name to browse."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
        "required": ["concept"],
    }


def _tool_schema_forget() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "memory_id": {"type": "integer", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    }


def _tool_schema_forget_topic() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string", "description": "Topic or concept to forget."},
        },
        "required": ["topic"],
    }


def _tool_schema_redact() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "memory_id": {"type": "integer", "description": "Memory ID to redact."},
            "fields": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["memory_id"],
    }


def _tool_schema_gdpr_delete() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "user_id": {"type": "string", "description": "User tag prefix, e.g. user:<id>."},
            "concept": {"type": "string", "description": "Concept text to delete memories for."},
        },
    }


def _tool_schema_count() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
    }


def _tool_schema_link() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "integer", "description": "Source memory ID."},
            "target_id": {"type": "integer", "description": "Target memory ID."},
            "edge_type": {"type": "string", "enum": sorted(EDGE_TYPES)},
            "weight": {"type": "number", "minimum": 0.0, "default": 1.0},
        },
        "required": ["source_id", "target_id", "edge_type"],
    }


def _tool_schema_concepts() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}}

def _tool_schema_compile_context() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1, "description": "Query / task to compile context for."},
            "budget": {"type": "integer", "minimum": 1, "maximum": 200000, "default": 4000},
            "policy": {
                "type": "string",
                "enum": ["balanced", "precise", "broad", "temporal"],
                "default": "balanced",
            },
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
        "required": ["query"],
    }


def _tool_schema_set_scope() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES)},
            "from_scope": {"type": "string", "enum": list(ALLOWED_SCOPES)},
            "memory_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "description": "Optional memory IDs to limit updates to.",
            },
        },
        "required": ["scope"],
    }


def _tool_schema_beliefs() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}}


def _tool_schema_contradictions() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
        },
    }


def _tool_schema_communities() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include_summaries": {
                "type": "boolean",
                "default": False,
                "description": "If true, compute a short summary per community from matching memories.",
            },
        },
    }


def _tool_schema_sleep() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verbose": {"type": "boolean", "default": False},
        },
    }


def _active_memory_count(syn: Synapse) -> int:
    # Mirror Synapse.recall() filtering: ignore consolidated memories.
    return sum(1 for m in syn.store.memories.values() if not m.get("consolidated", False))


def _build_server(*, syn: Synapse) -> Tuple[Server, asyncio.Lock]:
    server = Server("synapse-memory")
    db_lock = asyncio.Lock()
    egress_guard = EgressGuard(sensitivity=os.environ.get("SYNAPSE_EGRESS_SENSITIVITY", "standard"))

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="remember",
                description="Store a memory in Synapse.",
                inputSchema=_tool_schema_remember(),
            ),
            types.Tool(
                name="recall",
                description="Recall relevant memories from Synapse using the query as context.",
                inputSchema=_tool_schema_recall(),
            ),
            types.Tool(
                name="forget",
                description="Delete a memory by ID.",
                inputSchema=_tool_schema_forget(),
            ),
            types.Tool(
                name="forget_topic",
                description="Forget all memories related to a concept/topic.",
                inputSchema=_tool_schema_forget_topic(),
            ),
            types.Tool(
                name="redact",
                description="Redact memory content while keeping metadata and graph links.",
                inputSchema=_tool_schema_redact(),
            ),
            types.Tool(
                name="set_scope",
                description="Bulk update memory scope labels (public/shared/private).",
                inputSchema=_tool_schema_set_scope(),
            ),
            types.Tool(
                name="gdpr_delete",
                description="Delete memories by user ID or concept.",
                inputSchema=_tool_schema_gdpr_delete(),
            ),
            types.Tool(
                name="count",
                description="Count stored (non-consolidated) memories.",
                inputSchema=_tool_schema_count(),
            ),
            types.Tool(
                name="link",
                description="Create an edge between two memories.",
                inputSchema=_tool_schema_link(),
            ),
            types.Tool(
                name="concepts",
                description="List concepts in the concept graph.",
                inputSchema=_tool_schema_concepts(),
            ),
            types.Tool(
                name="list",
                description="List all memories (paginated, sortable).",
                inputSchema=_tool_schema_list(),
            ),
            types.Tool(
                name="browse",
                description="Browse memories by concept.",
                inputSchema=_tool_schema_browse(),
            ),
            types.Tool(
                name="consolidate",
                description="Consolidate similar memories into higher-level patterns.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_cluster_size": {"type": "integer", "description": "Minimum cluster size", "default": 3},
                        "similarity_threshold": {"type": "number", "description": "Similarity threshold (0-1)", "default": 0.5},
                    },
                },
            ),
            types.Tool(
                name="fact_history",
                description="Get temporal chain of how a fact evolved over time.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query to find fact history for"},
                        "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="hot_concepts",
                description="Get most frequently activated concepts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "description": "Number of concepts to return", "default": 5},
                    },
                },
            ),
            types.Tool(
                name="prune",
                description="Remove old, weak memories that haven't been accessed.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_age_days": {"type": "integer", "description": "Max age in days", "default": 90},
                        "min_strength": {"type": "number", "description": "Minimum strength threshold", "default": 0.3},
                    },
                },
            ),
            types.Tool(
                name="timeline",
                description="Get fact changes over time for a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query to get timeline for"},
                        "scope": {"type": "string", "enum": list(ALLOWED_SCOPES), "default": "private"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="compile_context",
                description="Compile an LLM-ready ContextPack (memories + graph + summaries + evidence).",
                inputSchema=_tool_schema_compile_context(),
            ),
            types.Tool(
                name="beliefs",
                description="Show current belief versions (worldview) derived from triples.",
                inputSchema=_tool_schema_beliefs(),
            ),
            types.Tool(
                name="contradictions",
                description="List unresolved contradictions detected in memory.",
                inputSchema=_tool_schema_contradictions(),
            ),
            types.Tool(
                name="communities",
                description="List detected concept communities (optionally with summaries).",
                inputSchema=_tool_schema_communities(),
            ),
            types.Tool(
                name="sleep",
                description="Run the full sleep maintenance cycle (consolidate/promote/mine/prune/cleanup).",
                inputSchema=_tool_schema_sleep(),
            ),
            types.Tool(
                name="ingest",
                description="Process text through the memory router for intelligent capture.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text content to ingest"},
                        "source": {"type": "string", "description": "Source identifier (e.g., 'api', 'chat', 'file')", "default": "api"},
                        "policy": {"type": "string", "enum": ["auto", "minimal", "review", "off"], "description": "Routing policy", "default": "auto"},
                        "metadata": {"type": "object", "description": "Additional metadata to attach"},
                    },
                    "required": ["text"],
                },
            ),
            types.Tool(
                name="watch_stream",
                description="Set up stream watching with memory router (returns watcher ID for callback-based streams).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "stream_type": {"type": "string", "enum": ["callback"], "description": "Stream type (only 'callback' supported in MCP)", "default": "callback"},
                        "policy": {"type": "string", "enum": ["auto", "minimal", "review", "off"], "description": "Routing policy", "default": "auto"},
                        "batch_size": {"type": "integer", "description": "Messages per batch", "default": 5},
                        "batch_timeout": {"type": "number", "description": "Batch timeout in seconds", "default": 30.0},
                        "source": {"type": "string", "description": "Source identifier", "default": "stream"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
        args = arguments or {}
        requested_scope = "private"

        def _memory_visible_in_scope(memory_data: Dict[str, Any]) -> bool:
            return _is_scope_visible(_stored_scope(memory_data.get("scope")), requested_scope)

        try:
            if name == "remember":
                content = args.get("content", "")
                if not isinstance(content, str):
                    raise ValueError("content must be a string")
                content = content.strip()
                if not content:
                    raise ValueError("content cannot be empty")

                memory_type = args.get("memory_type", "fact")
                if not isinstance(memory_type, str):
                    raise ValueError("memory_type must be a string")
                if memory_type not in MEMORY_TYPES:
                    raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of {sorted(MEMORY_TYPES)}")

                metadata = args.get("metadata", None)
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object (JSON dictionary)")

                episode = args.get("episode", None)
                if episode is not None and not isinstance(episode, str):
                    raise ValueError("episode must be a string")
                episode = (episode or "").strip() or None
                scope = _normalize_scope(args.get("scope", "private"))

                async with db_lock:
                    mem = syn.remember(
                        content=content,
                        memory_type=memory_type,
                        metadata=metadata,
                        episode=episode,
                        extract=False,
                        scope=scope,
                    )
                    syn.flush()

                payload = _ok_payload({"memory": _memory_to_dict(mem)})

            elif name == "recall":
                query = args.get("query", "")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")

                limit = args.get("limit", 10)
                limit_i = _as_int(limit, field="limit")
                if limit_i < 1 or limit_i > 100:
                    raise ValueError("limit must be between 1 and 100")

                memory_type = args.get("memory_type", None)
                if memory_type is not None:
                    if not isinstance(memory_type, str):
                        raise ValueError("memory_type must be a string")
                    if memory_type not in MEMORY_TYPES:
                        raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of {sorted(MEMORY_TYPES)}")

                do_explain = bool(args.get("explain", False))
                scope = _normalize_scope(args.get("scope", "private"))
                requested_scope = scope

                async with db_lock:
                    memories = syn.recall(context=query, limit=limit_i,
                                          memory_type=memory_type, explain=do_explain,
                                          show_disputes=bool(args.get("show_disputes", False)),
                                          exclude_conflicted=bool(args.get("exclude_conflicted", False)),
                                          scope=scope)

                mem_dicts = []
                for m in memories:
                    d = _memory_to_dict(m)
                    if do_explain and m.score_breakdown is not None:
                        from dataclasses import asdict as _asdict
                        d["score_breakdown"] = _asdict(m.score_breakdown)
                    mem_dicts.append(d)

                payload = _ok_payload(
                    {
                        "memories": mem_dicts,
                        "returned": len(memories),
                    }
                )

            elif name == "forget":
                memory_id = _as_int(args.get("memory_id"), field="memory_id")
                async with db_lock:
                    deleted = bool(syn.forget(memory_id))
                    if deleted:
                        syn.flush()
                payload = _ok_payload({"deleted": deleted, "memory_id": memory_id})

            elif name == "forget_topic":
                topic = args.get("topic", "")
                if not isinstance(topic, str) or not topic.strip():
                    raise ValueError("topic must be a non-empty string")

                async with db_lock:
                    result = syn.forget_topic(topic=topic.strip())
                    if result.get("deleted_count", 0):
                        syn.flush()
                payload = _ok_payload(result)

            elif name == "redact":
                memory_id = _as_int(args.get("memory_id"), field="memory_id")
                fields = args.get("fields", None)
                if fields is not None and not isinstance(fields, list):
                    raise ValueError("fields must be a list of strings if provided")

                normalized_fields = None
                if isinstance(fields, list):
                    normalized_fields = []
                    for field_name in fields:
                        if not isinstance(field_name, str):
                            raise ValueError("fields entries must be strings")
                        field_name = field_name.strip()
                        if not field_name:
                            continue
                        normalized_fields.append(field_name)
                    if not normalized_fields:
                        normalized_fields = None

                async with db_lock:
                    result = syn.redact(memory_id=memory_id, fields=normalized_fields)
                    if result.get("redacted", False):
                        syn.flush()
                payload = _ok_payload(result)

            elif name == "set_scope":
                target_scope = _normalize_scope(
                    args.get("scope"),
                    field="scope",
                    allow_default=False,
                )
                from_scope_raw = args.get("from_scope", None)
                from_scope = (
                    _normalize_scope(from_scope_raw, field="from_scope")
                    if from_scope_raw is not None
                    else None
                )
                memory_ids_raw = args.get("memory_ids", None)
                memory_ids_filter: Optional[set[int]] = None
                if memory_ids_raw is not None:
                    if not isinstance(memory_ids_raw, list):
                        raise ValueError("memory_ids must be an array of integers")
                    memory_ids_filter = set()
                    for raw_id in memory_ids_raw:
                        memory_ids_filter.add(_as_int(raw_id, field="memory_ids[]"))

                updated_ids: List[int] = []
                scanned = 0
                async with db_lock:
                    for memory_id, memory_data in syn.store.memories.items():
                        if memory_ids_filter is not None and memory_id not in memory_ids_filter:
                            continue
                        scanned += 1
                        current_scope = _stored_scope(memory_data.get("scope"))
                        if from_scope is not None and current_scope != from_scope:
                            continue
                        if current_scope == target_scope:
                            continue
                        syn.store.update_memory(
                            memory_id,
                            {"scope": target_scope},
                        )
                        updated_ids.append(memory_id)
                    if updated_ids:
                        syn.flush()
                payload = _ok_payload(
                    {
                        "scope": target_scope,
                        "from_scope": from_scope,
                        "updated_count": len(updated_ids),
                        "scanned_count": scanned,
                        "updated_memory_ids": updated_ids,
                    }
                )

            elif name == "gdpr_delete":
                user_id = args.get("user_id", None)
                concept = args.get("concept", None)
                if user_id is None and concept is None:
                    raise ValueError("one of user_id or concept must be provided")
                if user_id is not None and not isinstance(user_id, str):
                    raise ValueError("user_id must be a string")
                if concept is not None:
                    if not isinstance(concept, str) or not concept.strip():
                        raise ValueError("concept must be a non-empty string if provided")
                    concept = concept.strip()

                async with db_lock:
                    result = syn.gdpr_delete(user_id=user_id, concept=concept)
                    if result.get("deleted_count", 0):
                        syn.flush()
                payload = _ok_payload(result)

            elif name == "count":
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    c = sum(
                        1
                        for memory_data in syn.store.memories.values()
                        if not memory_data.get("consolidated", False)
                        and _memory_visible_in_scope(memory_data)
                    )
                payload = _ok_payload({"count": c})

            elif name == "link":
                source_id = _as_int(args.get("source_id"), field="source_id")
                target_id = _as_int(args.get("target_id"), field="target_id")
                edge_type = args.get("edge_type")
                if not isinstance(edge_type, str):
                    raise ValueError("edge_type must be a string")
                if edge_type not in EDGE_TYPES:
                    raise ValueError(f"Invalid edge_type: {edge_type}. Must be one of {sorted(EDGE_TYPES)}")

                weight = args.get("weight", 1.0)
                weight_f = _as_float(weight, field="weight")
                if weight_f < 0.0:
                    raise ValueError("weight must be >= 0.0")

                async with db_lock:
                    syn.link(source_id, target_id, edge_type, weight=weight_f)
                    syn.flush()
                payload = _ok_payload(
                    {
                        "linked": True,
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type,
                        "weight": weight_f,
                    }
                )

            elif name == "concepts":
                async with db_lock:
                    concepts = syn.concepts()
                payload = _ok_payload({"concepts": concepts, "returned": len(concepts)})

            elif name == "list":
                limit_i = _as_int(args.get("limit", 50), field="limit")
                offset_i = _as_int(args.get("offset", 0), field="offset")
                sort_str = args.get("sort", "recent")
                if sort_str not in ("recent", "created", "access_count"):
                    raise ValueError("sort must be one of: recent, created, access_count")
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    memories = syn.list(limit=limit_i, offset=offset_i, sort=sort_str)
                    memories = [
                        memory for memory in memories
                        if _is_scope_visible(_stored_scope(getattr(memory, "scope", "private")), requested_scope)
                    ]
                    total = sum(
                        1
                        for memory_data in syn.store.memories.values()
                        if not memory_data.get("consolidated", False)
                        and _memory_visible_in_scope(memory_data)
                    )
                payload = _ok_payload({
                    "memories": [_memory_to_dict(m) for m in memories],
                    "returned": len(memories),
                    "total": total,
                })

            elif name == "browse":
                concept = args.get("concept", "")
                if not isinstance(concept, str) or not concept.strip():
                    raise ValueError("concept is required")
                limit_i = _as_int(args.get("limit", 50), field="limit")
                offset_i = _as_int(args.get("offset", 0), field="offset")
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    memories = syn.browse(concept=concept.strip(), limit=limit_i, offset=offset_i)
                    memories = [
                        memory for memory in memories
                        if _is_scope_visible(_stored_scope(getattr(memory, "scope", "private")), requested_scope)
                    ]
                payload = _ok_payload({
                    "memories": [_memory_to_dict(m) for m in memories],
                    "returned": len(memories),
                })

            elif name == "consolidate":
                min_cluster = _as_int(args.get("min_cluster_size", 3), field="min_cluster_size")
                sim_thresh = _as_float(args.get("similarity_threshold", 0.5), field="similarity_threshold")
                async with db_lock:
                    result = syn.consolidate(min_cluster_size=min_cluster, similarity_threshold=sim_thresh)
                    syn.flush()
                payload = _ok_payload({"consolidated": result, "count": len(result)})

            elif name == "fact_history":
                query = args.get("query", "")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query is required")
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    chain = syn.fact_history(query.strip())
                    chain = [
                        entry for entry in chain
                        if _is_scope_visible(
                            _stored_scope(getattr(entry.get("memory"), "scope", "private")),
                            requested_scope,
                        )
                    ]
                payload = _ok_payload({"chain": chain, "count": len(chain)})

            elif name == "hot_concepts":
                k = _as_int(args.get("k", 5), field="k")
                async with db_lock:
                    hot = syn.hot_concepts(k=k)
                payload = _ok_payload({"concepts": [{"name": n, "score": s} for n, s in hot]})

            elif name == "prune":
                max_age = _as_int(args.get("max_age_days", 90), field="max_age_days")
                min_str = _as_float(args.get("min_strength", 0.3), field="min_strength")
                async with db_lock:
                    pruned = syn.prune(max_age_days=max_age, min_strength=min_str)
                    syn.flush()
                payload = _ok_payload({"pruned_count": pruned})

            elif name == "timeline":
                query = args.get("query", "")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query is required")
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    chain = syn.fact_history(query.strip())
                    chain = [
                        entry for entry in chain
                        if _is_scope_visible(
                            _stored_scope(getattr(entry.get("memory"), "scope", "private")),
                            requested_scope,
                        )
                    ]
                payload = _ok_payload({"timeline": chain, "count": len(chain)})

            elif name == "compile_context":
                query = args.get("query", "")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string")

                budget = args.get("budget", 4000)
                budget_i = _as_int(budget, field="budget")
                if budget_i <= 0:
                    raise ValueError("budget must be > 0")

                policy = args.get("policy", "balanced")
                if not isinstance(policy, str) or not policy.strip():
                    policy = "balanced"
                policy = policy.strip().lower()
                scope = _normalize_scope(args.get("scope", "private"))
                requested_scope = scope

                async with db_lock:
                    pack = syn.compile_context(
                        query=query.strip(),
                        budget=budget_i,
                        policy=policy,
                        scope=scope,
                    )
                context_pack = pack.to_dict()
                compact = pack.to_compact()
                system_prompt = pack.to_system_prompt()
                payload = _ok_payload(
                    {
                        "context_pack": context_pack,
                        "compact": compact,
                        "system_prompt": system_prompt,
                    }
                )

            elif name == "beliefs":
                async with db_lock:
                    beliefs = syn.beliefs()

                result: Dict[str, Any] = {}
                for fact_key, version in (beliefs or {}).items():
                    if is_dataclass(version):
                        result[fact_key] = asdict(version)
                    else:
                        result[fact_key] = dict(getattr(version, "__dict__", {}))

                payload = _ok_payload({"beliefs": result, "count": len(result)})

            elif name == "contradictions":
                requested_scope = _normalize_scope(args.get("scope", "private"))
                async with db_lock:
                    contradictions = syn.contradictions()
                    store_view = dict(syn.store.memories)

                items: List[Dict[str, Any]] = []
                for c in contradictions or []:
                    if is_dataclass(c):
                        row = asdict(c)
                    else:
                        row = dict(getattr(c, "__dict__", {}))
                    a_id = row.get("memory_id_a")
                    b_id = row.get("memory_id_b")
                    if not (
                        isinstance(a_id, int)
                        and isinstance(b_id, int)
                        and a_id in store_view
                        and b_id in store_view
                    ):
                        continue
                    if not (
                        _memory_visible_in_scope(store_view[a_id])
                        and _memory_visible_in_scope(store_view[b_id])
                    ):
                        continue
                    row["memory_a"] = (store_view[a_id].get("content") or "")[:400]
                    row["memory_b"] = (store_view[b_id].get("content") or "")[:400]
                    items.append(row)

                payload = _ok_payload({"contradictions": items, "count": len(items)})

            elif name == "communities":
                include_summaries = bool(args.get("include_summaries", False))
                async with db_lock:
                    communities = syn.communities()

                items: List[Dict[str, Any]] = []
                for community in communities or []:
                    if is_dataclass(community):
                        row = asdict(community)
                    else:
                        row = dict(getattr(community, "__dict__", {}))
                    if isinstance(row.get("concepts"), set):
                        row["concepts"] = sorted(row["concepts"])
                    if include_summaries:
                        hubs = row.get("hub_concepts") or []
                        hub = hubs[0] if hubs else None
                        if isinstance(hub, str) and hub.strip():
                            row["summary"] = syn.community_summary(hub)
                    items.append(row)

                payload = _ok_payload({"communities": items, "count": len(items)})

            elif name == "sleep":
                verbose = bool(args.get("verbose", False))
                async with db_lock:
                    report = syn.sleep(verbose=verbose)
                    syn.flush()

                if is_dataclass(report):
                    payload = _ok_payload({"sleep_report": asdict(report)})
                else:
                    payload = _ok_payload({"sleep_report": dict(getattr(report, "__dict__", {}))})

            elif name == "ingest":
                text = args.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a non-empty string")
                
                source = args.get("source", "api")
                if not isinstance(source, str):
                    raise ValueError("source must be a string")
                
                policy = args.get("policy", "auto")
                if policy not in ["auto", "minimal", "review", "off"]:
                    raise ValueError("policy must be one of: auto, minimal, review, off")
                
                metadata = args.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object")
                
                async with db_lock:
                    # Import here to avoid circular imports
                    from capture import ingest
                    from review_queue import ReviewQueue
                    
                    review_queue = ReviewQueue(syn) if policy in ['review', 'auto'] else None
                    
                    result = ingest(
                        text=text.strip(),
                        synapse=syn,
                        review_queue=review_queue,
                        source=source,
                        meta=metadata,
                        policy=policy
                    )
                    
                    syn.flush()
                
                payload = _ok_payload({
                    "result": result.name,
                    "description": {
                        "STORED": "Memory stored automatically",
                        "QUEUED_FOR_REVIEW": "Memory queued for manual review",
                        "IGNORED_FLUFF": "Ignored as conversational fluff",
                        "REJECTED_SECRET": "Rejected due to sensitive content",
                        "IGNORED_POLICY": "Ignored due to policy settings"
                    }.get(result.name, "Unknown result")
                })

            elif name == "watch_stream":
                stream_type = args.get("stream_type", "callback")
                if stream_type != "callback":
                    raise ValueError("Only 'callback' stream type is supported in MCP")
                
                policy = args.get("policy", "auto")
                if policy not in ["auto", "minimal", "review", "off"]:
                    raise ValueError("policy must be one of: auto, minimal, review, off")
                
                batch_size = _as_int(args.get("batch_size", 5), field="batch_size")
                batch_timeout = _as_float(args.get("batch_timeout", 30.0), field="batch_timeout")
                source = args.get("source", "stream")
                
                async with db_lock:
                    from watch import create_callback_watcher
                    from review_queue import ReviewQueue
                    
                    review_queue = ReviewQueue(syn) if policy in ['review', 'auto'] else None
                    
                    watcher = create_callback_watcher(
                        syn,
                        review_queue=review_queue,
                        policy=policy,
                        batch_size=batch_size,
                        batch_timeout=batch_timeout
                    )
                    
                    # Store watcher reference (in real implementation, you'd want a registry)
                    watcher_id = f"watcher_{int(time.time() * 1000000)}"
                    
                payload = _ok_payload({
                    "watcher_id": watcher_id,
                    "status": "created",
                    "config": {
                        "stream_type": stream_type,
                        "policy": policy,
                        "batch_size": batch_size,
                        "batch_timeout": batch_timeout,
                        "source": source
                    },
                    "note": "To send messages to this watcher, you would call watcher.on_message(text) in your integration code"
                })

            else:
                payload = _error_payload(ValueError(f"Unknown tool: {name}"), code="unknown_tool")

        except Exception as exc:
            # Important: keep stdout clean for JSON-RPC; log errors to stderr only.
            LOG.exception("Tool call failed: %s", name)
            payload = _error_payload(exc)

        if payload.get("ok") and requested_scope != "private":
            payload = _ok_payload(_filter_text_payload(payload.get("result"), egress_guard))

        return [types.TextContent(type="text", text=_json_dumps(payload))]

    return server, db_lock


async def _run_server(*, data_dir: str) -> None:
    # Store uses "path" as a file prefix; it writes "<path>.log" and "<path>.snapshot".
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    db_prefix = os.path.join(data_dir, "synapse")

    syn = Synapse(db_prefix)
    # Synapse AI Memory can optionally call a local Ollama server for embeddings; for MCP usage
    # we default this off to avoid surprising latency/timeouts. Opt-in via env var.
    if os.environ.get("SYNAPSE_MCP_ENABLE_EMBEDDINGS", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        syn._use_embeddings = False  # intentional: avoid lazy availability check during remember/recall
    server, _db_lock = _build_server(syn=syn)

    # Avoid accidental stdout writes from logging; only stderr.
    LOG.info("Synapse AI Memory MCP server starting. data_dir=%s db_prefix=%s", data_dir, db_prefix)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for Synapse AI Memory (stdio transport).")
    parser.add_argument(
        "--data-dir",
        default="~/.synapse",
        help="Directory to store Synapse AI Memory files (default: ~/.synapse).",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "appliance"],
        default="appliance",
        help="Tool surface to expose: full (21 tools) or appliance (8 tools).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        if args.mode == "full":
            asyncio.run(_run_server(data_dir=args.data_dir))
        else:
            from mcp_appliance import _run_server as _run_appliance_server

            asyncio.run(_run_appliance_server(data_dir=args.data_dir))
    except KeyboardInterrupt:
        return 0
    except Exception:
        LOG.exception("Fatal server error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
