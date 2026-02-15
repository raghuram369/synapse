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

from synapse import Synapse, MEMORY_TYPES, EDGE_TYPES, Memory


LOG = logging.getLogger("synapse-mcp")


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
        },
        "required": ["query"],
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


def _tool_schema_count() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}}


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


def _active_memory_count(syn: Synapse) -> int:
    # Mirror Synapse.recall() filtering: ignore consolidated memories.
    return sum(1 for m in syn.store.memories.values() if not m.get("consolidated", False))


def _build_server(*, syn: Synapse) -> Tuple[Server, asyncio.Lock]:
    server = Server("synapse-memory")
    db_lock = asyncio.Lock()

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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
        args = arguments or {}

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

                async with db_lock:
                    mem = syn.remember(
                        content=content,
                        memory_type=memory_type,
                        metadata=metadata,
                        episode=episode,
                        extract=False,
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

                async with db_lock:
                    memories = syn.recall(context=query, limit=limit_i, memory_type=memory_type)

                payload = _ok_payload(
                    {
                        "memories": [_memory_to_dict(m) for m in memories],
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

            elif name == "count":
                async with db_lock:
                    c = _active_memory_count(syn)
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

            else:
                payload = _error_payload(ValueError(f"Unknown tool: {name}"), code="unknown_tool")

        except Exception as exc:
            # Important: keep stdout clean for JSON-RPC; log errors to stderr only.
            LOG.exception("Tool call failed: %s", name)
            payload = _error_payload(exc)

        return [types.TextContent(type="text", text=_json_dumps(payload))]

    return server, db_lock


async def _run_server(*, data_dir: str) -> None:
    # Store uses "path" as a file prefix; it writes "<path>.log" and "<path>.snapshot".
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    db_prefix = os.path.join(data_dir, "synapse")

    syn = Synapse(db_prefix)
    # Synapse can optionally call a local Ollama server for embeddings; for MCP usage
    # we default this off to avoid surprising latency/timeouts. Opt-in via env var.
    if os.environ.get("SYNAPSE_MCP_ENABLE_EMBEDDINGS", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        syn._use_embeddings = False  # intentional: avoid lazy availability check during remember/recall
    server, _db_lock = _build_server(syn=syn)

    # Avoid accidental stdout writes from logging; only stderr.
    LOG.info("Synapse MCP server starting. data_dir=%s db_prefix=%s", data_dir, db_prefix)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for Synapse memory DB (stdio transport).")
    parser.add_argument(
        "--data-dir",
        default="~/.synapse",
        help="Directory to store Synapse files (default: ~/.synapse).",
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
        LOG.exception("Fatal server error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
