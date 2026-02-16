#!/usr/bin/env python3
"""Unified CLI for the Synapse AI Memory engine.

Sub-commands cover the core engine, portable format, and federation.
"""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict
import asyncio
import glob
import json
import os
import signal
import subprocess
import tempfile
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from typing import Any, Dict

from pathlib import Path
from client import SynapseClient, SynapseRequestError
from exceptions import SynapseConnectionError
from demo_runner import DemoRunner
from doctor import run_doctor


# ─── Formatting helpers ──────────────────────────────────────────────────────

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_DIM = "\033[2m"
_DAY_SECONDS = 60 * 60 * 24


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def _bold(text: str) -> str:
    return _color(text, ANSI_BOLD)


def _cyan(text: str) -> str:
    return _color(text, ANSI_CYAN)


def _green(text: str) -> str:
    return _color(text, ANSI_GREEN)


def _yellow(text: str) -> str:
    return _color(text, ANSI_YELLOW)


def _red(text: str) -> str:
    return _color(text, ANSI_RED)


def _blue(text: str) -> str:
    return _color(text, ANSI_BLUE)


def _dim(text: str) -> str:
    return _color(text, ANSI_DIM)


def _resolve_db_path(args) -> str:
    return args.db or ":memory:"


APPLIANCE_DB_DEFAULT = "./synapse_store"
APPLIANCE_BANNER = "Synapse AI Memory v0.6.0 — MCP server ready"
APPLIANCE_DAEMON_PORT = 9470
APPLIANCE_STATE_DIR = os.path.expanduser("~/.synapse")


def _resolve_appliance_db_path(args, default: str = APPLIANCE_DB_DEFAULT) -> str:
    return (
        getattr(args, "db", None)
        or getattr(args, "data", None)
        or default
    )


def _status_marker(level: str) -> str:
    if level == "pass":
        return _green("[PASS]")
    if level == "warn":
        return _yellow("[WARN]")
    return _red("[FAIL]")


def _report_status(level: str, label: str, detail: str | None = None):
    marker = _status_marker(level)
    suffix = f" — {detail}" if detail else ""
    print(f"{marker} {label}{suffix}")


def _collect_mcp_tools(db_path: str) -> list[Dict[str, Any]]:
    from synapse import Synapse

    try:
        import mcp_server
    except Exception as exc:
        raise RuntimeError(
            "MCP package is not available in this environment. "
            "Install 'mcp' to run appliance commands."
        ) from exc

    syn = Synapse(db_path)
    try:
        server, _db_lock = mcp_server._build_server(syn=syn)
        loop = asyncio.new_event_loop()
        try:
            tools = loop.run_until_complete(server.list_tools())
        finally:
            loop.close()
    finally:
        try:
            syn.close()
        finally:
            # keep parity if close fails for any reason
            pass
        # avoid flake warnings if syn.close raises before join; explicit finally keeps scope clear

    catalog: list[Dict[str, Any]] = []
    for tool in tools:
        catalog.append({
            "name": tool.name,
            "description": tool.description or "",
            "inputSchema": getattr(tool, "inputSchema", {}),
        })
    return catalog


def _format_tool_schema(schema: Any, prefix: str = "    ") -> str:
    if not schema:
        return f"{prefix}{{}}"
    return "\n".join(f"{prefix}{line}" for line in json.dumps(schema, indent=2, sort_keys=True).splitlines())


def _collect_store_snapshot(db_path: str) -> Dict[str, Any]:
    from synapse import Synapse

    syn = Synapse(db_path)
    try:
        contradictions = _collect_contradictions(syn)
        hook = syn.sleep_runner.schedule_hook()
        hot = syn.hot_concepts(k=5)
    finally:
        syn.close()

    return {
        "memory_count": syn.count(),
        "concept_count": len(syn.concept_graph.concepts),
        "edge_count": len(syn.store.edges),
        "contradiction_count": len(contradictions),
        "belief_count": len(syn.beliefs() or {}),
        "last_sleep_at": hook.get("last_sleep_at"),
        "top_hot_concepts": hot,
        "active_memories": hook.get("active_memory_count"),
        "store_path": db_path,
    }


def _appliance_state_dir() -> str:
    return APPLIANCE_STATE_DIR


def _appliance_pid_path() -> str:
    return os.path.join(_appliance_state_dir(), "synapse.pid")


def _daemon_state() -> Dict[str, Any]:
    path = _appliance_pid_path()
    try:
        with open(path, "r", encoding="utf-8") as fp:
            text = fp.read().strip()
    except (FileNotFoundError, OSError):
        return {}
    try:
        return {"pid": int(text)}
    except (TypeError, ValueError):
        return {}


def _read_daemon_pid() -> int | None:
    return _daemon_state().get("pid")


def _write_daemon_pid(path: str, pid: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(str(int(pid)))


def _clear_daemon_pid(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _format_uptime(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        days, hours = divmod(hours, 24)
        if days:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _wait_for_pid_file(path: str, timeout: float = 2.0, poll_interval: float = 0.05) -> int | None:
    deadline = time.time() + timeout
    if path is None:
        path = _appliance_pid_path()
    while time.time() < deadline:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read().strip()
            return int(text)
        except (FileNotFoundError, OSError, ValueError):
            time.sleep(poll_interval)
            continue
    return None


def _daemon_command(port: int, db_path: str, mode: str) -> list[str]:
    return [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "synapsed.py"),
        "--port", str(port),
        "--db", db_path,
        "--mode", mode,
        "--daemon",
        "--pid-file", _appliance_pid_path(),
    ]


def _start_synapse_daemon(command: list[str]):
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _send_signal(pid: int, sig: int) -> None:
    os.kill(pid, sig)


def _collect_sleep_status(db_path: str) -> Dict[str, Any]:
    from synapse import Synapse
    syn = Synapse(db_path)
    try:
        return syn.sleep_runner.schedule_hook()
    finally:
        syn.close()


def _store_file_size(path: str) -> int:
    if not path or path == ":memory:":
        return 0
    base = os.path.abspath(os.path.expanduser(path))
    total = 0
    for suffix in ("", ".log", ".snapshot", ".state"):
        candidate = f"{base}{suffix}"
        try:
            total += int(os.path.getsize(candidate))
        except OSError:
            continue
    return total


def _scan_portable_exports(db_path: str) -> list[tuple[str, bool, str]]:
    export_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    from portable import inspect_synapse

    reports: list[tuple[str, bool, str]] = []
    for path in sorted(glob.glob(os.path.join(export_dir, "*.synapse"))):
        try:
            info = inspect_synapse(path)
            crc_ok = bool(info.get("crc_valid", False))
            if crc_ok:
                reports.append((path, True, "CRC valid"))
            else:
                reports.append((path, False, "CRC invalid"))
        except Exception as exc:
            reports.append((path, False, f"Unable to inspect: {exc}"))
    return reports


def _run_store_latency_probe(db_path: str) -> float:
    tmpdir = tempfile.mkdtemp(prefix="synapse-doctor-")
    probe_path = os.path.join(tmpdir, "probe_store")

    from synapse import Synapse

    s = Synapse(probe_path)
    try:
        start = time.perf_counter()
        s.remember("Synapse MCP appliance health check", deduplicate=False)
        s.recall("MCP appliance health check", limit=1)
        end = time.perf_counter()
    finally:
        s.close()
        try:
            pass
        finally:
            # best effort cleanup for probe folder
            for artifact in ("probe_store.log", "probe_store.snapshot"):
                try:
                    os.remove(os.path.join(tmpdir, artifact))
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    return (end - start) * 1000.0


def _check_store_files(db_path: str) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    base_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    log_path = f"{db_path}.log"
    snapshot_path = f"{db_path}.snapshot"

    try:
        os.makedirs(base_dir, exist_ok=True)
        checks.append(("storage directory", "pass", f"{base_dir}"))
    except OSError as exc:
        checks.append(("storage directory", "fail", str(exc)))
        return checks

    for path in (log_path, snapshot_path):
        try:
            with open(path, "a", encoding="utf-8"):
                pass
            with open(path, "r", encoding="utf-8"):
                pass
            checks.append((path, "pass", "read/write"))
        except OSError as exc:
            checks.append((path, "fail", str(exc)))

    try:
        if os.path.exists(snapshot_path):
            with open(snapshot_path, "r", encoding="utf-8") as fp:
                json.load(fp)
            checks.append(("snapshot json", "pass", "valid JSON"))
        else:
            checks.append(("snapshot json", "warn", "not found yet"))
    except Exception as exc:
        checks.append(("snapshot json", "fail", str(exc)))

    return checks


def _run_mcp_stdio_server(db_path: str):
    import asyncio

    try:
        import mcp_server
        from mcp.server.stdio import stdio_server
        from synapse import Synapse
    except Exception as exc:
        print("Error: MCP package is required for stdio mode.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)

    syn = Synapse(db_path)
    try:
        server, _db_lock = mcp_server._build_server(syn=syn)

        async def _run() -> None:
            async with stdio_server() as streams:
                read_stream, write_stream = streams
                await server.run(read_stream, write_stream, server.create_initialization_options())

        asyncio.run(_run())
    finally:
        syn.close()


def _run_mcp_http_server(db_path: str, port: int):
    tools = _collect_mcp_tools(db_path)

    class _Handler(BaseHTTPRequestHandler):
        server_version = "SynapseMCP/0.6.0"

        def _respond_json(self, payload: Any, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _json_ok(payload: Any) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "result": payload}

        @staticmethod
        def _json_error(message: str, code: int = -32000) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "error": {"code": code, "message": message}}

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/tools", "/tools/list"}:
                self._respond_json(self._json_ok({"tools": tools}))
            else:
                self._respond_json(self._json_error("Not found", -32601), status=404)

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if path not in {"/", "/jsonrpc", "/rpc"}:
                self._respond_json(self._json_error("Not found", -32601), status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length).decode("utf-8")
                request = json.loads(payload) if payload else {}
            except Exception as exc:
                self._respond_json(self._json_error(f"Invalid request: {exc}", -32700), status=400)
                return

            method = str(request.get("method", ""))
            if method in {"tools/list", "list_tools"}:
                self._respond_json(self._json_ok({"tools": tools}))
                return
            if method == "tools/call":
                self._respond_json(self._json_error("Tool calls are not implemented in this HTTP shim."), status=501)
                return
            if method == "initialize":
                self._respond_json(
                    self._json_ok({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "synapse-memory", "version": "0.6.0"},
                    })
                )
                return

            self._respond_json(self._json_error(f"Unsupported method: {method}", -32601))

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    try:
        print(f"Listening on 127.0.0.1:{port}/jsonrpc")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down HTTP server")
    finally:
        server.server_close()


def _format_datetime(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_concept_name(concept_graph, concept: str) -> str | None:
    target = (concept or "").strip().lower()
    if not target:
        return None
    if target in concept_graph.concepts:
        return target
    for name in concept_graph.concepts:
        if name.lower() == target:
            return name
    return None


def _resolve_concepts_for_memory(synapse, memory_id: int):
    concepts = synapse.concept_graph.get_memory_concepts(memory_id)
    return sorted(str(c) for c in concepts)


def _collect_contradictions(synapse):
    from contradictions import ContradictionDetector
    detector = ContradictionDetector()
    memories = [
        synapse._memory_data_to_object(memory_data)
        for memory_data in synapse.store.memories.values()
        if not memory_data.get('consolidated', False)
    ]
    return detector.scan_memories(memories)


def _memory_objects_by_id(synapse):
    return {
        memory_id: synapse._memory_data_to_object(memory_data)
        for memory_id, memory_data in synapse.store.memories.items()
        if not memory_data.get('consolidated', False)
    }


def _concept_activation(synapse, concept: str) -> float:
    return synapse.concept_graph.concept_activation_strength(concept)


def _format_conflict_kind(kind: str) -> str:
    return {
        "mutual_exclusion": "exclusion",
        "numeric_range": "numeric",
        "temporal_conflict": "temporal",
        "polarity": "polarity",
    }.get(kind, kind)


def _suggest_resolution(kind: str) -> str:
    return {
        "polarity": "Supersede or replace the older statement to maintain consistency.",
        "exclusion": "Keep one value and supersede conflicting alternatives in the same fact chain.",
        "numeric": "Keep the most authoritative numeric source and close or override the older one.",
        "temporal": "Prefer the most recent temporally-valid statement and supersede older facts.",
    }.get(_format_conflict_kind(kind), "Review and resolve the pair manually.")


def _memory_graph_score(synapse, memory_id: int) -> float:
    edges = synapse.edge_graph.get_all_edges(memory_id)
    if not edges:
        return 0.0
    total_weight = sum(float(edge.weight) for _, edge in edges)
    return min(1.0, total_weight / (1.0 + len(edges)))


def _history_snapshot(memory_id: int, synapse) -> list[dict]:
    try:
        chain = synapse.history(memory_id)
        if len(chain) >= 2:
            return chain
    except Exception:
        return []
    return []


def _memory_brief(content: str, max_len: int = 64) -> str:
    cleaned = (content or "").replace("\n", " ")
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len-3]}..."


def format_memory(memory: Dict[str, Any], show_metadata: bool = False) -> str:
    lines = []
    lines.append(f"🧠 Memory #{memory['id']} ({memory['memory_type']})")
    lines.append(f"   Content: {memory['content']}")
    lines.append(f"   Strength: {memory.get('strength', 'N/A')}")
    if 'effective_strength' in memory:
        lines.append(f"   Effective: {memory['effective_strength']:.3f}")
    if 'created_at' in memory:
        import datetime
        created = datetime.datetime.fromtimestamp(memory['created_at'])
        lines.append(f"   Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
    if show_metadata and memory.get('metadata'):
        lines.append(f"   Metadata: {json.dumps(memory['metadata'], indent=2)}")
    return '\n'.join(lines)


def format_concept(concept: Dict[str, Any]) -> str:
    return f"🏷️  {concept['name']} ({concept['category']}) - {concept['memory_count']} memories"


def format_stats(stats: Dict[str, Any]) -> str:
    lines = ["📊 Server Statistics:"]
    lines.append(f"   Memories: {stats.get('total_memories', 0)}")
    lines.append(f"   Concepts: {stats.get('total_concepts', 0)}")
    lines.append(f"   Edges: {stats.get('total_edges', 0)}")
    lines.append(f"   Clients: {stats.get('client_count', 0)}")
    if 'data_directory' in stats:
        lines.append(f"   Data Dir: {stats['data_directory']}")
    return '\n'.join(lines)


def _human_size(size: int | float) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ─── Core commands (daemon client) ───────────────────────────────────────────

def cmd_remember(args, client: SynapseClient):
    try:
        memory = client.remember(content=args.content, memory_type=args.type, extract=args.extract)
        print("✅ Memory stored successfully!")
        print(format_memory(memory))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_recall(args, client: SynapseClient):
    try:
        kwargs = dict(context=args.query, limit=args.limit)
        if getattr(args, 'explain', False):
            kwargs['explain'] = True
        if getattr(args, 'at', None):
            kwargs['temporal'] = args.at
        memories = client.recall(**kwargs)
        if not memories:
            print("🔍 No memories found")
            return
        print(f"🔍 Found {len(memories)} memories:\n")
        for i, memory in enumerate(memories):
            if i > 0:
                print()
            print(format_memory(memory, show_metadata=args.metadata))
            if getattr(args, 'explain', False) and 'score_breakdown' in memory:
                bd = memory['score_breakdown']
                print(f"   Score Breakdown:")
                print(f"     BM25: {bd.get('bm25_score', 0):.3f}")
                print(f"     Concept: {bd.get('concept_score', 0):.3f}")
                print(f"     Temporal: {bd.get('temporal_score', 0):.3f}")
                print(f"     Episode: {bd.get('episode_score', 0):.3f}")
                print(f"     Concept Activation: {bd.get('concept_activation_score', 0):.3f}")
                print(f"     Embedding: {bd.get('embedding_score', 0):.3f}")
                print(f"     Sources: {bd.get('match_sources', [])}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_list(args, client: SynapseClient):
    try:
        memories = client.list(limit=args.limit, offset=args.offset, sort=args.sort)
        if not memories:
            print("📋 No memories found")
            return
        print(f"📋 Listing {len(memories)} memories:\n")
        for i, memory in enumerate(memories):
            if i > 0:
                print()
            print(format_memory(memory))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_count(args, client: SynapseClient):
    try:
        count = client.count()
        print(f"🔢 Total memories: {count}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_browse(args, client: SynapseClient):
    try:
        memories = client.browse(concept=args.concept, limit=args.limit, offset=args.offset)
        if not memories:
            print(f"🏷️  No memories found for concept '{args.concept}'")
            return
        print(f"🏷️  {len(memories)} memories for concept '{args.concept}':\n")
        for i, memory in enumerate(memories):
            if i > 0:
                print()
            print(format_memory(memory))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_forget(args, client: SynapseClient):
    try:
        deleted = client.forget(args.id)
        if deleted:
            print(f"✅ Memory #{args.id} deleted successfully")
        else:
            print(f"⚠️  Memory #{args.id} not found")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_link(args, client: SynapseClient):
    try:
        client.link(source_id=args.source, target_id=args.target,
                     edge_type=args.edge_type, weight=args.weight)
        print(f"✅ Link created: #{args.source} --[{args.edge_type}]--> #{args.target}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_concepts(args, client: SynapseClient):
    try:
        concepts = client.concepts()
        if not concepts:
            print("🏷️  No concepts found")
            return
        print(f"🏷️  Found {len(concepts)} concepts:\n")
        for concept in concepts[:args.limit]:
            print(format_concept(concept))
        if len(concepts) > args.limit:
            print(f"... and {len(concepts) - args.limit} more")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_hot_concepts(args, client: SynapseClient):
    try:
        hot = client.hot_concepts(k=args.k)
        if not hot:
            print("🔥 No active concepts yet")
            return
        print(f"🔥 Top {len(hot)} hot concepts:\n")
        for name, strength in hot:
            print(f"  - {name}: {strength:.3f}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_prune(args, client: SynapseClient):
    try:
        pruned = client.prune(
            min_strength=args.min_strength,
            min_access=args.min_access,
            max_age_days=args.max_age,
            dry_run=args.dry_run,
        )
        if not pruned:
            print("🧹 Nothing to prune")
            return
        if args.dry_run:
            print(f"🧹 Would prune {len(pruned)} memories: {pruned}")
        else:
            print(f"🧹 Pruned {len(pruned)} memories: {pruned}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_stats(args, client: SynapseClient | None = None):
    if getattr(args, 'db', None):
        from synapse import Synapse

        s = Synapse(args.db)
        try:
            print(_bold("📊 Synapse Debug Stats"))

            active_memories = [
                mid for mid, memory_data in s.store.memories.items()
                if not memory_data.get('consolidated', False)
            ]
            total_memories = len(active_memories)
            total_concepts = len(s.concept_graph.concepts)
            total_edges = len(s.store.edges)

            contradictions = _collect_contradictions(s)
            unresolved = len(contradictions)

            print(_cyan("Core totals:"))
            print(f"  Total memories: {total_memories}")
            print(f"  Total concepts: {total_concepts}")
            print(f"  Total edges: {total_edges}")
            print(f"  Contradictions: {unresolved}")

            hot = s.hot_concepts(k=10)
            print(_cyan("Top 10 hottest concepts:"))
            if hot:
                for name, score in hot:
                    print(f"  - {_green(name)}: {score:.3f}")
            else:
                print("  - none")

            now = time.time()
            buckets = defaultdict(int)
            for memory_id in active_memories:
                memory_data = s.store.memories[memory_id]
                age_days = (now - memory_data['created_at']) / _DAY_SECONDS
                if age_days < 1:
                    buckets["<1 day"] += 1
                elif age_days < 7:
                    buckets["1-6 days"] += 1
                elif age_days < 30:
                    buckets["7-29 days"] += 1
                elif age_days < 90:
                    buckets["30-89 days"] += 1
                else:
                    buckets["90+ days"] += 1

            print(_cyan("Memory age distribution:"))
            for bucket, count in buckets.items():
                print(f"  - {bucket}: {count}")

            hook = s.sleep_runner.schedule_hook()
            last_sleep = hook.get("last_sleep_at")
            if last_sleep is None:
                print(_cyan("Last sleep:") + " never")
            else:
                print(_cyan("Last sleep:") + f" {_format_datetime(last_sleep)}")
                if hook.get("seconds_since_last_sleep") is not None:
                    since = hook["seconds_since_last_sleep"]
                    if since >= 0:
                        print(_dim(f"  ({int(since)}s ago, next due at { _format_datetime(hook.get('next_due_at') or now)})"))

        finally:
            s.close()
        return

    if client is None:
        print("❌ Server stats requires a running daemon. Pass --db for local stats.")
        sys.exit(1)

    try:
        stats = client.stats()
        print(format_stats(stats))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def _format_policy_overview(policy: Dict[str, Any] | None) -> None:
    if not policy:
        print("No active policy. Set one with `synapse policy set <preset>`.")
        return

    name = str(policy.get("name", "")).strip() or "unknown"
    print(f"✅ Active policy: {name}")
    description = policy.get("description")
    if description:
        print(f"  description: {description}")
    print(f"  ttl_days: {policy.get('ttl_days')}")
    print(f"  auto_prune: {policy.get('auto_prune')}")
    print(f"  prune_min_access: {policy.get('prune_min_access')}")
    print(f"  redact_pii: {policy.get('redact_pii')}")
    if policy.get("keep_tags"):
        print(f"  keep_tags: {policy.get('keep_tags')}")
    if policy.get("auto_tag_project"):
        print("  auto_tag_project: enabled")
    if policy.get("pin_tag"):
        print(f"  pin_tag: {policy.get('pin_tag')}")
    if policy.get("pii_patterns"):
        print(f"  pii_patterns: {policy.get('pii_patterns')}")


def cmd_policy(args, client: SynapseClient):
    try:
        action = getattr(args, "policy_action", "show")
        if action == "list":
            presets = client.policy_list()
            print("Available policy presets:")
            for name, cfg in presets.items():
                desc = cfg.get("description", "")
                print(f"  - {name}: {desc}")
            return

        if action == "set":
            active = client.policy_set(args.preset)
            _format_policy_overview(active)
            return

        policy = client.policy_show()
        _format_policy_overview(policy)
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_ping(args, client: SynapseClient):
    try:
        response = client.ping()
        print(f"📡 Server response: {response}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_shutdown(args, client: SynapseClient):
    try:
        if not args.force:
            response = input("⚠️  Are you sure you want to shut down the server? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Shutdown cancelled")
                return
        client.shutdown()
        print("🛑 Server shutdown initiated")
    except SynapseConnectionError:
        print("✅ Server shut down successfully")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


# ─── New Feature Commands ────────────────────────────────────────────────────

def cmd_nlforget(args, client: SynapseClient):
    """Handle natural language forget commands."""
    try:
        result = client.request('natural_forget', {
            'command': args.command,
            'dry_run': args.dry_run,
            'confirm': not args.no_confirm
        })
        
        if result.get('status') == 'dry_run':
            print(f"🔍 {_yellow('DRY RUN:')} {result['message']}")
            if 'memories' in result:
                print("\nMemories that would be deleted:")
                for mem in result['memories']:
                    print(f"  [{mem['id']}] {mem['content']}")
            print("\nRun without --dry-run to actually delete.")
        elif result.get('status') == 'deleted':
            print(f"✅ {_green('Success:')} {result['message']}")
        elif result.get('status') == 'not_found':
            print(f"ℹ️  {_blue('Info:')} {result['message']}")
        elif result.get('status') == 'error':
            print(f"❌ {_red('Error:')} {result['message']}")
        else:
            print(f"📋 Result: {result}")
            
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_inbox(args, client: SynapseClient):
    """Handle inbox management commands."""
    if not hasattr(args, 'inbox_action') or not args.inbox_action:
        print("❌ No inbox action specified. Use 'synapse inbox -h' for help.")
        return
    
    try:
        if args.inbox_action == 'list':
            result = client.request('list_pending', {'limit': args.limit})
            pending = result.get('pending', [])
            
            if not pending:
                print("📥 Inbox is empty")
                return
            
            print(f"📥 {_bold(f'{len(pending)} pending memories:')}")
            for item in pending:
                item_id = item.get('id', 'unknown')
                content = item.get('content', '')[:100]
                submitted_at = item.get('submitted_at', 0)
                time_str = datetime.datetime.fromtimestamp(submitted_at).strftime('%Y-%m-%d %H:%M')
                print(f"  [{_cyan(item_id)}] {time_str} - {content}...")
        
        elif args.inbox_action == 'approve':
            result = client.request('approve_memory', {'item_id': args.item_id})
            if result.get('success'):
                memory_id = result.get('memory_id')
                print(f"✅ {_green('Approved')} memory {args.item_id} → memory ID {memory_id}")
            else:
                print(f"❌ {_red('Failed')} to approve: {result.get('error', 'Unknown error')}")
        
        elif args.inbox_action == 'reject':
            result = client.request('reject_memory', {'item_id': args.item_id})
            if result.get('success'):
                print(f"🗑️  {_yellow('Rejected')} and deleted item {args.item_id}")
            else:
                print(f"❌ {_red('Failed')} to reject: {result.get('error', 'Unknown error')}")
        
        elif args.inbox_action == 'redact':
            result = client.request('redact_memory', {
                'item_id': args.item_id, 
                'redacted_content': args.new_content
            })
            if result.get('success'):
                memory_id = result.get('memory_id')
                print(f"✏️  {_yellow('Redacted')} and approved → memory ID {memory_id}")
            else:
                print(f"❌ {_red('Failed')} to redact: {result.get('error', 'Unknown error')}")
        
        elif args.inbox_action == 'pin':
            result = client.request('pin_memory', {'item_id': args.item_id})
            if result.get('success'):
                memory_id = result.get('memory_id')
                print(f"📌 {_green('Pinned')} and approved → memory ID {memory_id}")
            else:
                print(f"❌ {_red('Failed')} to pin: {result.get('error', 'Unknown error')}")
        
        elif args.inbox_action == 'query':
            result = client.request('query_pending', {'query': args.query, 'limit': args.limit})
            results = result.get('results', [])
            
            if not results:
                print(f"🔍 No pending memories match '{args.query}'")
                return
            
            print(f"🔍 {_bold(f'{len(results)} matching pending memories:')}")
            for item in results:
                item_id = item.get('id', 'unknown')
                content = item.get('content', '')[:100]
                score = item.get('search_score', 0)
                print(f"  [{_cyan(item_id)}] ({score} matches) {content}...")
                
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_vault(args, client: SynapseClient):
    """Handle vault management commands."""
    if not hasattr(args, 'vault_action') or not args.vault_action:
        print("❌ No vault action specified. Use 'synapse vault -h' for help.")
        return
    
    try:
        if args.vault_action == 'list':
            result = client.request('list_vaults', {})
            vaults = result.get('vaults', [])
            
            if not vaults:
                print("🏛️  No vaults found")
                return
            
            print(f"🏛️  {_bold(f'{len(vaults)} vaults:')}")
            for vault in vaults:
                vault_id = vault.get('vault_id', 'unknown')
                user_id = vault.get('user_id', 'no user')
                created_at = vault.get('created_at', 0)
                time_str = datetime.datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M')
                print(f"  [{_cyan(vault_id)}] User: {user_id} | Created: {time_str}")
        
        elif args.vault_action == 'create':
            result = client.request('create_vault', {
                'vault_id': args.vault_id,
                'user_id': args.user_id
            })
            if result.get('success'):
                print(f"🏛️  {_green('Created')} vault '{args.vault_id}'")
            else:
                print(f"❌ {_red('Failed')} to create vault: {result.get('error', 'Unknown error')}")
        
        elif args.vault_action == 'switch':
            result = client.request('switch_vault', {'vault_id': args.vault_id})
            if result.get('success'):
                print(f"🏛️  {_green('Switched')} to vault '{args.vault_id}'")
            else:
                print(f"❌ {_red('Failed')} to switch vault: {result.get('error', 'Unknown error')}")
                
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


# ─── Portable Format commands (standalone, no daemon needed) ─────────────────

def cmd_bench(args):
    from bench.consumer_bench import main as bench_main
    bench_main(
        scenario=getattr(args, 'scenario', None),
        output=getattr(args, 'output', None),
        fmt=getattr(args, 'bench_format', 'both'),
    )


def cmd_history(args):
    from synapse import Synapse
    db_path = args.db or ":memory:"
    s = Synapse(db_path)
    chain = s.fact_history(args.query)
    if not chain:
        print("🔍 No fact chain found")
    else:
        print(f"📜 Fact history ({len(chain)} versions):\n")
        for entry in chain:
            m = entry['memory']
            marker = " ← current" if entry['current'] else ""
            import datetime
            created = datetime.datetime.fromtimestamp(m.created_at)
            print(f"  v{entry['version']} [{created.strftime('%Y-%m-%d %H:%M')}] #{m.id}: {m.content}{marker}")
    s.close()


def cmd_why(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        memory_data = s.store.memories.get(args.id)
        if not memory_data or memory_data.get('consolidated', False):
            print(_red(f"⚠️  Memory #{args.id} not found"))
            return
        memory = s._memory_data_to_object(memory_data)

        scored = None
        matches = s.recall(context=memory.content, limit=25, explain=True)
        for candidate in matches:
            if candidate.id == args.id:
                scored = candidate
                break

        if scored is None:
            scored = memory
            # Provide synthetic score components for unavailable breakdowns.
            scored.score_breakdown = type("ScoreBreakdown", (), {
                "bm25_score": 0.0,
                "concept_score": 0.0,
                "temporal_score": 0.0,
                "episode_score": 0.0,
                "concept_activation_score": 0.0,
                "embedding_score": 0.0,
                "match_sources": [],
            })()

        bd = scored.score_breakdown

        print(_bold(f"🧠 Why memory #{memory.id}"))
        print(f"  Content: {_yellow(_memory_brief(memory.content, max_len=120))}")
        print(f"  Type: {memory.memory_type}")
        print(f"  Strength: {memory.strength:.3f}")

        print(_cyan("  Score components:"))
        print(f"    BM25: {bd.bm25_score:.3f}")
        print(f"    Concept: {bd.concept_score:.3f}")
        print(f"    Temporal: {bd.temporal_score:.3f}")
        print(f"    Episode: {bd.episode_score:.3f}")
        print(f"    Activation: {bd.concept_activation_score:.3f}")
        print(f"    Graph: {_memory_graph_score(s, memory.id):.3f}")

        concepts = _resolve_concepts_for_memory(s, memory.id)
        if concepts:
            print(_cyan("  Linked concepts:"))
            for concept in concepts:
                print(f"    - {concept}")
        else:
            print(_dim("  No concepts linked"))

        contradictions = _collect_contradictions(s)
        relevant = [
            c for c in contradictions
            if args.id in {c.memory_id_a, c.memory_id_b}
        ]
        print(_cyan("  Contradictions:"))
        if relevant:
            for conflict in relevant:
                other = conflict.memory_id_b if conflict.memory_id_a == args.id else conflict.memory_id_a
                other_memory = s.store.memories.get(other)
                snippet = other_memory['content'] if other_memory else "<missing>"
                print(f"    - {_yellow(_format_conflict_kind(conflict.kind))} with #{other}: {snippet}")
                print(f"      confidence: {conflict.confidence:.2f}")
        else:
            print(_dim("    None"))

        chain = _history_snapshot(args.id, s)
        if chain:
            print(_cyan("  Belief chain:"))
            for entry in chain:
                state = "current" if entry['current'] else "archived"
                m = entry['memory']
                ts = _format_datetime(m.created_at)
                print(f"    - #{m.id} [{ts}] ({state}) { _memory_brief(m.content, max_len=90)}")
        else:
            print(_dim("  Belief chain: none"))
    finally:
        s.close()


def cmd_graph(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        concept = _normalize_concept_name(s.concept_graph, args.concept)
        if concept is None:
            print(_red(f"⚠️  Concept '{args.concept}' not found"))
            return

        base_activation = _concept_activation(s, concept)
        print(_bold(f"🔗 Concept neighborhood: {concept}"))
        print(f"  Activation: {base_activation:.3f}")

        base_memories = set(s.concept_graph.get_concept_memories(concept))
        one_hop: Dict[str, int] = defaultdict(int)
        two_hop: Dict[str, int] = defaultdict(int)

        for memory_id in base_memories:
            for linked in s.concept_graph.get_memory_concepts(memory_id):
                if linked.lower() == concept.lower():
                    continue
                one_hop[linked] += 1

        print(_cyan("  1-hop neighbors:"))
        if not one_hop:
            print(_dim("    none"))
        for concept_name in sorted(one_hop, key=lambda item: (-one_hop[item], item)):
            mem_ids = sorted(s.concept_graph.get_concept_memories(concept_name))
            strength = _concept_activation(s, concept_name)
            print(f"    + {concept_name} (w={one_hop[concept_name]}) [act={strength:.2f}]")
            if mem_ids:
                preview = []
                for memory_id in mem_ids[:4]:
                    memory_data = s.store.memories.get(memory_id)
                    if memory_data:
                        preview.append(f"#{memory_id}:{_memory_brief(memory_data.get('content', ''), max_len=60)}")
                print(f"      memories: {', '.join(preview)}")

            for memory_id in mem_ids:
                for linked in s.concept_graph.get_memory_concepts(memory_id):
                    if linked.lower() in {concept.lower(), concept_name.lower()}:
                        continue
                    two_hop[linked] += 1

        if two_hop:
            print(_cyan("  2-hop neighbors:"))
            for concept_name in sorted(two_hop, key=lambda item: (-two_hop[item], item)):
                print(f"    + {concept_name} (w={two_hop[concept_name]})")
        else:
            print(_cyan("  2-hop neighbors: none"))
    finally:
        s.close()


def cmd_conflicts(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        memory_by_id = _memory_objects_by_id(s)
        conflicts = _collect_contradictions(s)
        print(_bold("⚠️  Active contradictions"))
        if not conflicts:
            print(_dim("No unresolved contradictions"))
            return

        for idx, conflict in enumerate(conflicts, start=1):
            left = memory_by_id.get(conflict.memory_id_a)
            right = memory_by_id.get(conflict.memory_id_b)
            kind = _format_conflict_kind(conflict.kind)
            print(f"{idx}. {_yellow(kind.upper())} (confidence={conflict.confidence:.2f})")
            print(f"   #{conflict.memory_id_a}: {_memory_brief(left.content if left else '[missing]', 110)}")
            print(f"   #{conflict.memory_id_b}: {_memory_brief(right.content if right else '[missing]', 110)}")
            print(f"   Suggestion: {_blue(_suggest_resolution(kind))}")
    finally:
        s.close()


def cmd_beliefs(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        beliefs = s.beliefs()
        print(_bold("🧭 Current worldview"))
        if not beliefs:
            print(_dim("No belief versions found"))
            return

        now = time.time()
        for fact_key in sorted(beliefs):
            version = beliefs[fact_key]
            history = s.belief_tracker.get_history(fact_key)
            version_count = len(history)
            low_confidence = version.confidence < 0.7
            recent = (now - version.valid_from) <= _DAY_SECONDS
            flags = []
            if low_confidence:
                flags.append("low-confidence")
            if recent:
                flags.append("recent")
            if not flags:
                flags.append("stable")

            print(_cyan(f"- {fact_key}"))
            print(f"  current: {version.value} (conf={version.confidence:.2f}, memory=#{version.memory_id})")
            print(f"  versions: {version_count} | flags: {', '.join(flags)}")
    finally:
        s.close()


def cmd_timeline(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        concept = args.concept.lower() if args.concept else None
        now = time.time()
        cutoff = None if args.days is None else now - (args.days * _DAY_SECONDS)

        entries = []
        for memory_id, memory_data in s.store.memories.items():
            if memory_data.get('consolidated', False):
                continue
            created_at = memory_data.get('created_at')
            if created_at is None:
                continue
            if cutoff is not None and created_at < cutoff:
                continue
            if concept is not None:
                concepts = {c.lower() for c in s.concept_graph.get_memory_concepts(memory_id)}
                if concept not in concepts:
                    continue

            m = s._memory_data_to_object(memory_data)
            entries.append(m)

        if not entries:
            print(_dim("🔍 No memories in timeline"))
            return

        entries.sort(key=lambda m: m.created_at)
        print(_bold(f"📅 Timeline ({len(entries)} entries)"))
        for entry in entries:
            ts = _format_datetime(entry.created_at)
            v_from = entry.valid_from
            v_to = entry.valid_to
            window = ""
            if v_from is not None or v_to is not None:
                from_txt = _format_datetime(v_from) if v_from is not None else "open"
                to_txt = _format_datetime(v_to) if v_to is not None else "open"
                window = f" validity=[{from_txt} -> {to_txt}]"
            print(f"  [{ts}] #{entry.id} ({entry.memory_type}){window} {entry.content}")
    finally:
        s.close()


def cmd_consolidate(args):
    from synapse import Synapse
    db_path = args.db or ":memory:"
    s = Synapse(db_path)
    results = s.consolidate(
        min_cluster_size=args.min_cluster,
        similarity_threshold=args.threshold,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )
    if not results:
        print("🔍 No clusters found to consolidate")
    else:
        mode = "DRY RUN — " if args.dry_run else ""
        print(f"🧠 {mode}{len(results)} cluster(s) consolidated:\n")
        for i, r in enumerate(results):
            print(f"  Cluster {i+1}: {r['source_count']} memories → strength {r['strength']:.2f}")
            print(f"    Concepts: {', '.join(r['concepts'][:10])}")
            print(f"    Summary: {r['summary'][:200]}")
            if 'consolidated_id' in r:
                print(f"    New memory ID: #{r['consolidated_id']}")
            print()
    if not args.dry_run:
        s.flush()
    s.close()


def cmd_sleep(args):
    from synapse import Synapse

    s = Synapse(args.db or ":memory:")
    try:
        report = s.sleep(verbose=getattr(args, "verbose", False))
        if getattr(args, "digest", False):
            print(report.to_digest())
            return

        print("🌙 Sleep complete")
        print(f"  Consolidated: {report.consolidated}")
        print(f"  Promoted: {report.promoted}")
        print(f"  Patterns found: {report.patterns_found}")
        print(f"  Contradictions: {report.contradictions}")
        print(f"  Pruned: {report.pruned}")
        print(f"  Graph cleaned: {report.graph_cleaned}")
        print(f"  Duration: {report.duration_ms:.2f} ms")
    finally:
        s.close()


def cmd_export(args):
    from portable import export_synapse
    from synapse import Synapse
    db_path = args.db or ":memory:"
    s = Synapse(db_path)
    concepts = args.concepts.split(',') if args.concepts else None
    tags = args.tags.split(',') if args.tags else None
    types = args.types.split(',') if args.types else None
    path = export_synapse(s, args.output, since=args.since, until=args.until,
                          concepts=concepts, tags=tags, memory_types=types,
                          source_agent=args.source or "cli")
    size = os.path.getsize(path)
    print(f"✓ Exported to {path} ({_human_size(size)})")
    s.close()


def cmd_import(args):
    from importers import ImportReport, MemoryImporter
    from portable import import_synapse
    from synapse import Synapse

    db_path = args.db or ":memory:"
    s = Synapse(db_path)

    mode = (args.input or "").strip().lower()
    path = args.path

    if mode in {"chat", "notes", "jsonl", "csv", "clipboard"}:
        if mode in {"chat", "notes", "jsonl", "csv"} and not path:
            raise ValueError(f"`{mode}` import requires a path argument")

        if mode == "chat":
            importer: ImportReport = MemoryImporter(s).from_chat_transcript(
                path, format=args.format
            )
            label = "chat transcript"
        elif mode == "notes":
            importer = MemoryImporter(s).from_markdown_folder(
                path, recursive=getattr(args, "recursive", False)
            )
            label = "markdown notes"
        elif mode == "jsonl":
            importer = MemoryImporter(s).from_jsonl(
                path, text_field=args.text_field, metadata_fields=None
            )
            label = "jsonl"
        elif mode == "csv":
            importer = MemoryImporter(s).from_csv(path, text_column=args.text_column)
            label = "csv"
        else:
            importer = MemoryImporter(s).from_clipboard()
            label = "clipboard"

        print(f"✓ Imported from {label} source")
        print(f"  Imported: {importer.imported}")
        print(f"  Skipped: {importer.skipped}")
        print(f"  Errors: {importer.errors}")
        print(f"  Duration: {importer.duration_ms:.2f} ms")
        if importer.tags_created:
            print(f"  Tags created: {', '.join(importer.tags_created)}")
        s.close()
        return

    # Legacy `.synapse` import
    stats = import_synapse(s, args.input, deduplicate=not args.no_dedup,
                           similarity_threshold=args.threshold)
    print(f"✓ Imported from {args.input}")
    print(f"  Memories: {stats['memories']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Concepts: {stats['concepts']}")
    print(f"  Episodes: {stats['episodes']}")
    if stats.get('skipped_duplicates'):
        print(f"  Skipped duplicates: {stats['skipped_duplicates']}")
    s.flush()
    s.close()


def cmd_inspect(args):
    db_path = _resolve_appliance_db_path(args)
    tools = _collect_mcp_tools(db_path)
    snapshot = _collect_store_snapshot(db_path)

    if args.json:
        payload = {
            "tools": tools,
            "store": {
                "memory_count": snapshot["memory_count"],
                "concept_count": snapshot["concept_count"],
                "edge_count": snapshot["edge_count"],
                "belief_count": snapshot["belief_count"],
                "contradictions": snapshot["contradiction_count"],
                "last_sleep_at": snapshot["last_sleep_at"],
                "top_hot_concepts": snapshot["top_hot_concepts"],
            },
            "store_path": db_path,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(_bold("🧰 MCP tool catalog"))
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
        print(_format_tool_schema(tool.get("inputSchema"), prefix="  "))

    print(_bold("📦 Store summary"))
    print(f"- Memories: {snapshot['memory_count']}")
    print(f"- Concepts: {snapshot['concept_count']}")
    print(f"- Edges: {snapshot['edge_count']}")
    print(f"- Beliefs: {snapshot['belief_count']}")
    print(f"- Contradictions: {snapshot['contradiction_count']}")
    last_sleep = snapshot["last_sleep_at"]
    if last_sleep is None:
        print("- Last sleep: never")
    else:
        print(f"- Last sleep: {_format_datetime(last_sleep)}")
    print("- Top 5 hot concepts:")
    if snapshot["top_hot_concepts"]:
        for name, score in snapshot["top_hot_concepts"]:
            print(f"  - {name}: {score:.3f}")
    else:
        print("  - none")


def cmd_inspector(args):
    from inspector import SynapseInspector
    from synapse import Synapse

    syn = Synapse(args.db or ":memory:")
    inspector = SynapseInspector(syn, port=args.port)
    try:
        inspector.start()
    finally:
        syn.close()


def cmd_inspect_export(args):
    from portable import inspect_synapse
    info = inspect_synapse(args.input)
    print(f"═══ {info['path']} ═══")
    print(f"  Format:      {info.get('format', 'unknown')}")
    print(f"  Size:        {info.get('file_size_human', '?')}")
    print(f"  CRC Valid:   {'✓' if info.get('crc_valid') else '✗'}")
    if 'version' in info:
        print(f"  Version:     {info['version']}")
    flags = info.get('flags', {})
    if flags:
        active_flags = [k for k, v in flags.items() if v]
        print(f"  Flags:       {', '.join(active_flags) if active_flags else 'none'}")
    print(f"  Memories:    {info.get('memory_count', '?')}")
    print(f"  Edges:       {info.get('edge_count', '?')}")
    print(f"  Concepts:    {info.get('concept_count', '?')}")
    print(f"  Episodes:    {info.get('episode_count', '?')}")
    print(f"  Source:      {info.get('source_agent', '?')}")
    filt = info.get('filter', {})
    if filt and filt.get('type') != 'full':
        print(f"  Filter:      {json.dumps(filt)}")
    if 'exported_at' in info:
        from datetime import datetime
        dt = datetime.fromtimestamp(info['exported_at'])
        print(f"  Exported:    {dt.isoformat()}")


def cmd_doctor(args):
    db_path = _resolve_appliance_db_path(args)

    issues = 0
    print(_bold("🩺 Synapse health check"))
    print(f"  Store: {db_path}")

    for label, status, detail in _check_store_files(db_path):
        _report_status(status, label, detail)
        if status != "pass":
            issues += 1

    try:
        snapshot = _collect_store_snapshot(db_path)
        _report_status(
            "pass",
            "Store counts",
            (
                f"memories={snapshot['memory_count']} "
                f"concepts={snapshot['concept_count']} "
                f"edges={snapshot['edge_count']} "
                f"beliefs={snapshot['belief_count']} "
                f"contradictions={snapshot['contradiction_count']}"
            ),
        )
    except Exception as exc:
        issues += 1
        _report_status("fail", "Store counts", f"Unable to load store: {exc}")
        snapshot = {
            "memory_count": 0,
            "concept_count": 0,
            "edge_count": 0,
            "belief_count": 0,
            "contradiction_count": 0,
            "last_sleep_at": None,
            "top_hot_concepts": [],
        }

    exports = _scan_portable_exports(db_path)
    if not exports:
        _report_status("pass", "Portable CRC checks", "no exports found")
    for path, crc_ok, detail in exports:
        if crc_ok:
            _report_status("pass", f"Portable: {path}", detail)
        else:
            issues += 1
            _report_status("fail", f"Portable: {path}", detail)

    try:
        latency_ms = _run_store_latency_probe(db_path)
        if latency_ms <= 500:
            _report_status("pass", "Performance", f"remember+recall {latency_ms:.2f} ms")
        else:
            issues += 1
            _report_status("warn", "Performance", f"remember+recall {latency_ms:.2f} ms")
    except Exception as exc:
        issues += 1
        _report_status("fail", "Performance", str(exc))

    if snapshot["contradiction_count"]:
        issues += 1
        _report_status("warn", "Unresolved contradictions", str(snapshot["contradiction_count"]))
    else:
        _report_status("pass", "Unresolved contradictions", "0")

    if issues:
        _report_status("fail", "Health status", f"{issues} issue(s)")
        sys.exit(1)

    print(_green("Health check complete"))


def cmd_up(args):
    db_path = _resolve_appliance_db_path(args)
    port = args.port
    mode = args.mode

    pid_path = _appliance_pid_path()
    existing_pid = _read_daemon_pid()
    if existing_pid is not None and _pid_is_running(existing_pid):
        print(f"⚠️  Synapse already running (pid {existing_pid}, port {port})")
        return

    if existing_pid is not None:
        _clear_daemon_pid(pid_path)

    command = _daemon_command(port=port, db_path=db_path, mode=mode)
    process = _start_synapse_daemon(command)
    daemon_pid = _wait_for_pid_file(pid_path)
    if daemon_pid is None:
        daemon_pid = process.pid
        _write_daemon_pid(pid_path, daemon_pid)

    print(f"🧠 Synapse AI Memory running (pid {daemon_pid}, port {port})")
    print(f"MCP stdio: synapse serve | HTTP: http://localhost:{port}")


def cmd_down(args):
    pid_path = _appliance_pid_path()
    pid = _read_daemon_pid()

    if pid is None:
        _clear_daemon_pid(pid_path)
        print("Synapse stopped.")
        return

    running = _pid_is_running(pid)
    if running:
        _send_signal(pid, signal.SIGTERM)
        deadline = time.time() + 2.0
        while time.time() < deadline and running:
            time.sleep(0.05)
            running = _pid_is_running(pid)
        if running:
            _send_signal(pid, signal.SIGKILL)
            deadline = time.time() + 0.5
            while time.time() < deadline and running:
                time.sleep(0.05)
                running = _pid_is_running(pid)

    _clear_daemon_pid(pid_path)
    print("Synapse stopped.")


def cmd_status(args):
    db_path = _resolve_appliance_db_path(args)
    port = args.port

    pid = _read_daemon_pid()
    pid_file = _appliance_pid_path()
    running = pid is not None and _pid_is_running(pid)
    started_at = os.path.getmtime(pid_file) if os.path.exists(pid_file) else None
    uptime = _format_uptime(time.time() - started_at) if running and started_at is not None else _dim("n/a")

    state_text = _green("running") if running else _red("stopped")
    print(_bold("🧠 Synapse Memory Status"))
    print(f"  State: {state_text}")
    print(f"  Uptime: {uptime}")
    print(f"  Port: {_blue(str(port))}")
    print(f"  DB: {_blue(db_path)}")

    if not running and pid is not None:
        _clear_daemon_pid(pid_file)
        running = False

    if not running:
        return

    try:
        snapshot = _collect_store_snapshot(db_path)
    except Exception as exc:
        print(f"  Snapshot unavailable: {exc}")
        snapshot = {
            "memory_count": 0,
            "concept_count": 0,
            "edge_count": 0,
            "contradiction_count": 0,
            "top_hot_concepts": [],
            "last_sleep_at": None,
        }

    try:
        sleep_status = _collect_sleep_status(db_path)
    except Exception:
        sleep_status = None

    print(_bold("  Memory"))
    print(f"    Count: {_cyan(str(snapshot['memory_count']))}")
    print(f"    Concepts: {_cyan(str(snapshot['concept_count']))}")
    print(f"    Edges: {_cyan(str(snapshot['edge_count']))}")
    print(f"    Contradictions: {_red(str(snapshot['contradiction_count']))}")

    last_sleep = snapshot.get("last_sleep_at")
    if last_sleep is None:
        print("    Last sleep: never")
    else:
        print(f"    Last sleep: {_format_datetime(last_sleep)}")

    if sleep_status is None:
        print("    Auto-sleep: unavailable")
    else:
        should_sleep = "on" if sleep_status.get("should_sleep") else "off"
        if should_sleep == "on":
            due = _format_datetime(sleep_status.get("next_due_at") or time.time())
            print(f"    Auto-sleep: {_yellow('on')} (next due: {due})")
        else:
            print(f"    Auto-sleep: {_yellow('off')}")

    store_size = _store_file_size(db_path)
    print(f"    Store size: {_human_size(store_size)}")

    print(_bold("  Top 5 hot concepts:"))
    hot = snapshot.get("top_hot_concepts") or []
    if hot:
        for name, score in hot[:5]:
            print(f"    - {name}: {score:.3f}")
    else:
        print("    - none")


def cmd_serve(args):
    db_path = _resolve_appliance_db_path(args)
    try:
        tools = _collect_mcp_tools(db_path)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(APPLIANCE_BANNER)
    print(_bold("MCP tools:"))
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")

    if args.http:
        print(f"HTTP JSON-RPC on localhost:{args.port}")
        _run_mcp_http_server(db_path=db_path, port=args.port)
    else:
        _run_mcp_stdio_server(db_path=db_path)


def cmd_merge(args):
    from portable import merge_synapse, import_synapse, export_synapse
    from synapse import Synapse
    s = Synapse(":memory:")
    files = args.inputs
    if len(files) < 2:
        print("Error: Need at least 2 files to merge")
        sys.exit(1)
    stats = import_synapse(s, files[0], deduplicate=False)
    print(f"✓ Base: {files[0]} ({stats['memories']} memories)")
    for f in files[1:]:
        stats = merge_synapse(s, f, conflict_resolution=args.conflict,
                              similarity_threshold=args.threshold)
        print(f"✓ Merged: {f} (+{stats['memories_added']} new, "
              f"{stats['memories_updated']} updated, {stats['memories_skipped']} skipped)")
    output = args.output or "merged.synapse"
    export_synapse(s, output, source_agent="merge-cli")
    size = os.path.getsize(output)
    print(f"✓ Output: {output} ({_human_size(size)})")
    s.close()


def cmd_diff(args):
    from portable import diff_synapse
    result = diff_synapse(args.file_a, args.file_b, similarity_threshold=args.threshold)
    print(f"═══ Diff: {result['file_a']} vs {result['file_b']} ═══")
    print(f"  Source A:    {result['source_a']}")
    print(f"  Source B:    {result['source_b']}")
    print(f"  Memories A:  {result['memories_a']}")
    print(f"  Memories B:  {result['memories_b']}")
    print(f"  Shared:      {result['shared']}")
    print(f"  Modified:    {result['modified']}")
    print(f"  Only in A:   {result['only_in_a']}")
    print(f"  Only in B:   {result['only_in_b']}")
    if result['only_in_a_samples']:
        print(f"\n  Only in A (samples):")
        for s in result['only_in_a_samples']:
            print(f"    [{s['id']}] {s['content']}")
    if result['only_in_b_samples']:
        print(f"\n  Only in B (samples):")
        for s in result['only_in_b_samples']:
            print(f"    [{s['id']}] {s['content']}")
    if result['modified_samples']:
        print(f"\n  Modified (samples):")
        for m in result['modified_samples']:
            print(f"    [{m['id_a']}↔{m['id_b']}] sim={m['similarity']}")
            print(f"      A: {m['content_a']}")
            print(f"      B: {m['content_b']}")


def cmd_pack(args):
    from packs import (
        BrainPack,
        default_pack_output_path,
        get_pack_directory,
        list_pack_files,
        parse_range_days,
    )
    from synapse import Synapse

    db_path = args.db or args.data or ":memory:"
    pack_dir = get_pack_directory(db_path=db_path, explicit=args.pack_dir)

    if args.list:
        files = list_pack_files(pack_dir)
        if not files:
            print("No brain packs found.")
            return
        print(_bold("🧠 Saved brain packs"))
        for path in files:
            try:
                pack = BrainPack.load(path)
            except Exception as exc:
                print(f"- {os.path.basename(path)} | ⚠️  {exc}")
                continue
            print(
                f"- {os.path.basename(path)} | topic: {pack.topic} | "
                f"memories: {len(pack.memories)} | created: {_format_datetime(pack.created_at)}"
            )
        return

    if args.diff:
        pack_a = BrainPack.load(args.diff[0])
        pack_b = BrainPack.load(args.diff[1])
        report = pack_a.diff(pack_b)
        print(report["markdown"])
        return

    if args.replay:
        pack = BrainPack.load(args.replay)
        s = Synapse(db_path)
        try:
            report = pack.replay(s)
        finally:
            s.close()
        print(report["markdown"])
        return

    if args.topic is None:
        print("❌ No pack action specified.")
        return

    range_days = parse_range_days(args.range)
    s = Synapse(db_path)
    try:
        pack = BrainPack(topic=args.topic, range_days=range_days).build(s)
        output = args.output or default_pack_output_path(
            topic=args.topic,
            db_path=db_path,
            created_at=pack.created_at,
            explicit_store=pack_dir,
        )
        if not output.lower().endswith(".brain"):
            output = f"{output}.brain"
        pack.save(output)
    finally:
        s.close()
    print(f"✓ Saved brain pack: {output}")
    try:
        from signing import sign_artifact
        sign_artifact(output)
    except Exception:
        pass


def cmd_card(args):
    action = getattr(args, 'card_action', None)
    if action == 'create':
        cmd_card_create(args)
        return
    if action == 'show':
        cmd_card_show(args)
        return
    if action == 'export':
        cmd_card_export(args)
        return
    if action == 'share':
        cmd_card_share(args)
        return
    print("❌ Unknown card action")
    sys.exit(1)


def cmd_card_create(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        card = s.create_card(args.query, budget=args.budget)
        print(card.to_markdown())
        s.flush()
    finally:
        s.close()


def cmd_card_show(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        card = s.cards.get(args.card_id)
        if card is None:
            print(f"Card not found: {args.card_id}")
            sys.exit(1)
        print(card.to_markdown())
    finally:
        s.close()


def cmd_card_export(args):
    from synapse import Synapse
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        s.cards.export(args.path)
        print(f"✓ Exported cards to {args.path}")
    finally:
        s.close()


def cmd_checkpoint(args):
    action = getattr(args, 'checkpoint_action', None)
    if action == 'create':
        cmd_checkpoint_create(args)
        return
    if action == 'list':
        cmd_checkpoint_list(args)
        return
    if action == 'diff':
        cmd_checkpoint_diff(args)
        return
    if action == 'restore':
        cmd_checkpoint_restore(args)
        return
    if action == 'delete':
        cmd_checkpoint_delete(args)
        return
    print("❌ Unknown checkpoint action")
    sys.exit(1)


def cmd_checkpoint_create(args):
    from synapse import Synapse
    s = Synapse(_resolve_db_path(args))
    try:
        cp = s.checkpoint(args.name, desc=getattr(args, 'desc', ''))
        print(f"✅ Created checkpoint '{cp.name}'")
        print(f"🧭 Description: {cp.description or '(none)'}")
        print(f"📦 Snapshot: {cp.snapshot_path}")
        print(f"🔐 Checksum: {cp.checksum}")
        print(f"🧠 Memories: {cp.stats.get('memory_count', 0)}")
        print(f"🔗 Concepts: {cp.stats.get('concept_count', 0)}")
    finally:
        s.close()


def cmd_checkpoint_list(args):
    from synapse import Synapse
    s = Synapse(_resolve_db_path(args))
    try:
        checkpoints = s.checkpoints.list()
    finally:
        s.close()

    if not checkpoints:
        print("⚪ No checkpoints found")
        return

    print("🗂  Checkpoints:")
    for item in checkpoints:
        print(f"- {item.name} ({_format_datetime(item.created_at)})")
        print(f"  desc: {item.description or '(none)'}")
        print(f"  memories: {item.stats.get('memory_count', 0)}")
        print(f"  checksum: {item.checksum}")


def cmd_checkpoint_diff(args):
    from synapse import Synapse
    s = Synapse(_resolve_db_path(args))
    try:
        payload = s.checkpoints.diff(args.a, args.b)
    finally:
        s.close()

    print(f"🔎 Diff {payload['checkpoint_a']} -> {payload['checkpoint_b']}")
    print(f"  memories_added: {len(payload['memories_added'])}")
    print(f"  memories_removed: {len(payload['memories_removed'])}")
    print(f"  beliefs_changed: {len(payload['beliefs_changed'])}")
    print(f"  contradictions_introduced: {len(payload['contradictions_introduced'])}")
    print(f"  contradictions_resolved: {len(payload['contradictions_resolved'])}")
    print(f"  concepts_added: {len(payload['concepts_added'])}")
    print(f"  concepts_removed: {len(payload['concepts_removed'])}")
    if payload["beliefs_changed"]:
        print("  sample belief delta:")
        for item in payload["beliefs_changed"][:5]:
            print(f"    {item['fact']}: {item['old_value']} -> {item['new_value']}")


def cmd_checkpoint_restore(args):
    from synapse import Synapse
    if not args.confirm:
        print("⚠️  Restoring a checkpoint is destructive. Use --confirm to proceed.")
        return

    s = Synapse(_resolve_db_path(args))
    try:
        report = s.checkpoints.restore(args.name)
    finally:
        s.close()

    print(f"♻️  Restored checkpoint '{report['checkpoint']}'")
    print(f"🧠 Memories restored: {report.get('memories_restored', 0)}")
    print(f"🔗 Concepts restored: {report.get('concepts_restored', 0)}")
    print(f"⚡ Beliefs restored: {report.get('beliefs_restored', 0)}")


def cmd_checkpoint_delete(args):
    from synapse import Synapse
    s = Synapse(_resolve_db_path(args))
    try:
        deleted = s.checkpoints.delete(args.name)
    finally:
        s.close()

    if deleted:
        print(f"✅ Deleted checkpoint '{args.name}'")
    else:
        print(f"⚠️  Checkpoint '{args.name}' not found")


def cmd_demo(args):
    runner = DemoRunner()
    output = "markdown" if args.markdown else "terminal"

    if args.scenario == "all":
        result = runner.run_all(output=output)
    else:
        result = runner.run(scenario=args.scenario, output=output)

    if args.markdown:
        print(result)


def cmd_uninstall(args):
    from installer import (
        uninstall_claude,
        uninstall_cursor,
        uninstall_windsurf,
        uninstall_continue,
        uninstall_openclaw,
        uninstall_nanoclaw,
        uninstall_telegram,
        uninstall_ollama,
        uninstall_all,
    )
    target = args.target
    if target == "claude":
        uninstall_claude()
    elif target == "cursor":
        uninstall_cursor()
    elif target == "windsurf":
        uninstall_windsurf()
    elif target == "continue":
        uninstall_continue()
    elif target == "openclaw":
        uninstall_openclaw()
    elif target == "nanoclaw":
        uninstall_nanoclaw()
    elif target == "telegram":
        uninstall_telegram()
    elif target == "ollama":
        uninstall_ollama()
    elif target == "all":
        uninstall_all()


def cmd_service(args):
    from service import install_service, uninstall_service, service_status
    action = getattr(args, "service_action", None)
    if action == "install":
        db_path = _resolve_appliance_db_path(args)
        install_service(db_path, sleep_schedule=getattr(args, "sleep_schedule", "daily"))
    elif action == "uninstall":
        uninstall_service()
    elif action == "status":
        service_status()
    else:
        print("Usage: synapse service install|uninstall|status")


def cmd_watch(args):
    """Enhanced watch command with memory router support."""
    from capture import clipboard_watch
    from review_queue import ReviewQueue
    from synapse import Synapse
    from watch import watch_stdin, watch_file
    
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    rq = ReviewQueue(s) if args.policy in ['review', 'auto'] else None
    
    try:
        if args.stdin:
            watch_stdin(
                s, 
                review_queue=rq,
                policy=args.policy, 
                batch_size=args.batch_size, 
                batch_timeout=args.batch_timeout
            )
        elif args.file:
            watch_file(
                args.file, 
                s, 
                review_queue=rq,
                policy=args.policy, 
                batch_size=args.batch_size, 
                batch_timeout=args.batch_timeout
            )
        elif args.clipboard:
            clipboard_watch(
                s, 
                interval=args.interval, 
                tags=args.tags, 
                policy=args.policy,
                review_queue=rq
            )
        else:
            print("❌ Please specify --stdin, --file, or --clipboard")
            sys.exit(1)
    finally:
        s.close()


def cmd_clip(args):
    """Enhanced clip command with memory router support."""
    from capture import clip_text, clip_stdin
    from review_queue import ReviewQueue
    from synapse import Synapse
    
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    rq = ReviewQueue(s) if args.policy in ['review', 'auto'] else None
    
    try:
        if args.text:
            result = clip_text(s, args.text, tags=args.tags, policy=args.policy, review_queue=rq)
            if result.name == "STORED":
                print(f"✅ Stored: {args.text[:80]}")
            elif result.name == "QUEUED_FOR_REVIEW":
                print(f"📋 Queued for review: {args.text[:80]}")
            elif result.name == "IGNORED_FLUFF":
                print(f"🙄 Ignored (fluff): {args.text[:80]}")
            elif result.name == "REJECTED_SECRET":
                print(f"🔒 Rejected (contains secrets)")
            elif result.name == "IGNORED_POLICY":
                print(f"⏸️  Ignored (policy): {args.text[:80]}")
        else:
            clip_stdin(s, tags=args.tags, policy=args.policy, review_queue=rq)
        s.flush()
    finally:
        s.close()


def cmd_ingest(args):
    """New ingest command for one-shot memory router processing."""
    from capture import ingest
    from review_queue import ReviewQueue
    from synapse import Synapse
    
    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    rq = ReviewQueue(s) if args.policy in ['review', 'auto'] else None
    
    try:
        if args.file and args.text:
            print("❌ Provide either text or --file, not both")
            return

        if args.file:
            # Ingest from file
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if not text:
                    print("⚠️  File is empty")
                    return

                result = ingest(
                    text=text,
                    synapse=s,
                    review_queue=rq,
                    source=f"file:{args.file}",
                    meta={"file_path": args.file},
                    policy=args.policy
                )

            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return
        else:
            # Ingest from command line argument
            if not args.text:
                print("❌ Missing text. Usage: synapse ingest \"text\" or synapse ingest --file path")
                return
            text = args.text.strip()
            if not text:
                print("⚠️  No text provided")
                return
                
            result = ingest(
                text=text,
                synapse=s,
                review_queue=rq,
                source="cli",
                meta={},
                policy=args.policy
            )
        
        # Print result
        preview = text[:100].replace('\n', ' ')
        if result.name == "STORED":
            print(f"✅ Stored: {preview}")
        elif result.name == "QUEUED_FOR_REVIEW":
            print(f"📋 Queued for review: {preview}")
            if rq:
                pending_count = rq.count()
                print(f"📊 {pending_count} items now pending review")
        elif result.name == "IGNORED_FLUFF":
            print(f"🙄 Ignored (conversational fluff): {preview}")
        elif result.name == "REJECTED_SECRET":
            print(f"🔒 Rejected (contains sensitive information)")
        elif result.name == "IGNORED_POLICY":
            print(f"⏸️  Ignored (policy={args.policy}): {preview}")
            
        s.flush()
    finally:
        s.close()


def cmd_capture(args):
    """New capture command for managing capture modes."""
    import os
    
    config_dir = os.path.expanduser("~/.synapse")
    config_file = os.path.join(config_dir, "capture_config.json")
    os.makedirs(config_dir, exist_ok=True)
    
    if args.mode:
        # Set capture mode
        import json
        config = {
            "capture_mode": args.mode,
            "updated_at": time.time()
        }
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Capture mode set to: {args.mode}")
            
            if args.mode == "auto":
                print("🤖 Auto mode: High-confidence memories stored automatically, medium-confidence go to review")
            elif args.mode == "minimal": 
                print("🎯 Minimal mode: Only very high-confidence memories stored automatically")
            elif args.mode == "review":
                print("📋 Review mode: All memories go to review queue for manual approval")
            elif args.mode == "off":
                print("⏸️  Off mode: Memory capture disabled")
                
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    else:
        # Show current mode
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            mode = config.get("capture_mode", "auto")
            updated_at = config.get("updated_at", 0)
            print(f"📋 Current capture mode: {mode}")
            if updated_at:
                import datetime
                updated = datetime.datetime.fromtimestamp(updated_at)
                print(f"🕐 Last updated: {updated.strftime('%Y-%m-%d %H:%M:%S')}")
        except FileNotFoundError:
            print("📋 Current capture mode: auto (default)")
        except Exception as e:
            print(f"❌ Error reading config: {e}")


def cmd_review(args):
    from review_queue import ReviewQueue
    from synapse import Synapse
    action = getattr(args, "review_action", None)

    if action == "count":
        rq = ReviewQueue(None)
        print(f"🔢 Pending: {rq.count()}")
        return

    if action == "list":
        rq = ReviewQueue(None)
        items = rq.list_pending()
        if not items:
            print("📋 No pending items")
            return
        print(f"📋 {len(items)} pending item(s):\n")
        for item in items:
            preview = item["content"][:100].replace("\n", " ")
            print(f"  [{item['id']}] {preview}")
        return

    if action == "reject":
        rq = ReviewQueue(None)
        if rq.reject(args.item_id):
            print(f"✅ Rejected item {args.item_id}")
        else:
            print(f"⚠️  Item {args.item_id} not found")
        return

    if action == "approve":
        db_path = _resolve_db_path(args)
        s = Synapse(db_path)
        rq = ReviewQueue(s)
        try:
            if args.item_id == "all":
                results = rq.approve_all()
                print(f"✅ Approved {len(results)} item(s)")
            else:
                memory = rq.approve(args.item_id)
                if memory:
                    print(f"✅ Approved and remembered item {args.item_id}")
                else:
                    print(f"⚠️  Item {args.item_id} not found")
            s.flush()
        finally:
            s.close()
        return

    print("Usage: synapse review list|approve|reject|count")


def cmd_sign(args):
    from signing import sign_artifact
    sign_artifact(args.file)


def cmd_verify(args):
    from signing import verify_artifact
    verify_artifact(args.file)


def cmd_install(args):
    from installer import ClientInstaller, install_all

    if args.list:
        print("Available install targets:")
        for target in sorted(ClientInstaller.TARGETS):
            print(f" - {target}")
        return

    target = (args.target or "").strip().lower()
    dry_run = getattr(args, "dry_run", False)
    verify_only = getattr(args, "verify_only", False)

    if target == "all":
        install_all(args.db or APPLIANCE_DB_DEFAULT, dry_run=dry_run)
        return

    if not target:
        print("Usage: synapse install --list | synapse install <target> [--db PATH]")
        print("Example: synapse install claude --db ./synapse_store")
        raise SystemExit(1)

    if dry_run or verify_only:
        if target == "telegram":
            # Telegram setup is wizard-style and does not currently support a separate
            # verify-only path; still trigger installer so setup text/flow stays in sync.
            from installer import install_telegram
            install_telegram(args.db or APPLIANCE_DB_DEFAULT)
            return
        enhanced = ClientInstaller.ENHANCED_TARGETS.get(target)
        if enhanced is None:
            print(f"Unknown install target: {target}")
            print(f"Available targets: {', '.join(sorted(ClientInstaller.TARGETS))}")
            raise SystemExit(1)
        enhanced(args.db or APPLIANCE_DB_DEFAULT, dry_run=dry_run, verify_only=verify_only)
    else:
        installer = ClientInstaller.TARGETS.get(target)
        if installer is None:
            print(f"Unknown install target: {target}")
            print(f"Available targets: {', '.join(sorted(ClientInstaller.TARGETS))}")
            raise SystemExit(1)
        installer(args.db or APPLIANCE_DB_DEFAULT)


# ─── Start command ────────────────────────────────────────────────────────────

def cmd_start(args):
    from installer import _detect_targets, _claude_config_path

    db_path = _resolve_appliance_db_path(args)
    port = args.port
    inspect_flag = getattr(args, "inspect_flag", False)
    sleep_schedule = getattr(args, "sleep_schedule", None)

    # Start daemon (reuse cmd_up logic)
    pid_path = _appliance_pid_path()
    existing_pid = _read_daemon_pid()
    already_running = existing_pid is not None and _pid_is_running(existing_pid)

    if not already_running:
        if existing_pid is not None:
            _clear_daemon_pid(pid_path)
        command = _daemon_command(port=port, db_path=db_path, mode="appliance")
        process = _start_synapse_daemon(command)
        daemon_pid = _wait_for_pid_file(pid_path)
        if daemon_pid is None:
            daemon_pid = process.pid
            _write_daemon_pid(pid_path, daemon_pid)

    # Handle sleep schedule
    if sleep_schedule and sleep_schedule != "off":
        state_dir = _appliance_state_dir()
        os.makedirs(state_dir, exist_ok=True)
        config_path = os.path.join(state_dir, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                config = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
        config["sleep_schedule"] = sleep_schedule
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2)

    # Optionally launch inspector
    inspector_url = None
    if inspect_flag:
        inspector_port = 7463
        inspector_url = f"http://127.0.0.1:{inspector_port}"
        # Launch inspector in background
        inspector_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli.py"),
            "inspector", "--port", str(inspector_port), "--db", db_path,
        ]
        subprocess.Popen(
            inspector_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Detect targets
    targets = _detect_targets(db_path)

    # Print status box
    print()
    print(_bold("╔══════════════════════════════════════════════════════╗"))
    print(_bold("║") + "  🧠 Synapse AI Memory                               " + _bold("║"))
    print(_bold("╠══════════════════════════════════════════════════════╣"))
    print(_bold("║") + f"  {_green('✅')} Memory service running on localhost:{port}       " + _bold("║"))

    if targets.get("claude"):
        print(_bold("║") + f"  {_green('✅')} Claude Desktop: connected                     " + _bold("║"))
    else:
        print(_bold("║") + f"  ⬜ Claude Desktop: not configured                 " + _bold("║"))
        print(_bold("║") + f"     run: synapse install claude                    " + _bold("║"))

    if targets.get("openclaw"):
        print(_bold("║") + f"  {_green('✅')} OpenClaw: skill installed                     " + _bold("║"))
    else:
        print(_bold("║") + f"  ⬜ OpenClaw: not configured                       " + _bold("║"))
        print(_bold("║") + f"     run: synapse install openclaw                  " + _bold("║"))

    if inspector_url:
        print(_bold("║") + f"  {_green('✅')} Inspector: {inspector_url}              " + _bold("║"))

    if sleep_schedule and sleep_schedule != "off":
        print(_bold("║") + f"  {_green('✅')} Auto-sleep: {sleep_schedule}                          " + _bold("║"))

    print(_bold("╠══════════════════════════════════════════════════════╣"))
    print(_bold("║") + "  Next steps:                                        " + _bold("║"))
    print(_bold("║") + "   synapse remember 'something important'             " + _bold("║"))
    print(_bold("║") + "   synapse recall 'what was important?'               " + _bold("║"))
    print(_bold("║") + "   synapse doctor                                     " + _bold("║"))
    print(_bold("╚══════════════════════════════════════════════════════╝"))
    print()


# ─── Enhanced doctor ──────────────────────────────────────────────────────────

def cmd_doctor_enhanced(args):
    db_path = _resolve_appliance_db_path(args)
    json_output = getattr(args, "json", False)
    results = []

    def add_check(name, status, detail=""):
        results.append({"name": name, "status": status, "detail": detail})

    # Storage health
    base = os.path.abspath(os.path.expanduser(db_path))
    store_size = _store_file_size(db_path)
    for label, status, detail in _check_store_files(db_path):
        add_check(f"storage:{label}", status, detail)
    add_check("storage:size", "pass", _human_size(store_size))

    # Index consistency
    try:
        from synapse import Synapse
        s = Synapse(db_path)
        try:
            mem_count = s.count()
            bm25_count = len(s.inverted_index.doc_lengths)
            if bm25_count == mem_count:
                add_check("index:bm25_consistency", "pass", f"docs={bm25_count} matches memories={mem_count}")
            else:
                add_check("index:bm25_consistency", "warn", f"BM25 docs={bm25_count} vs memories={mem_count}")

            # Contradictions
            contradictions = _collect_contradictions(s)
            count_c = len(contradictions)
            add_check("contradictions:pending", "warn" if count_c > 0 else "pass", str(count_c))

            # Sleep status
            hook = s.sleep_runner.schedule_hook()
            last_sleep = hook.get("last_sleep_at")
            if last_sleep:
                add_check("sleep:last", "pass", _format_datetime(last_sleep))
            else:
                add_check("sleep:last", "warn", "never")

            # Auto-sleep config
            state_dir = _appliance_state_dir()
            config_path = os.path.join(state_dir, "config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as fp:
                    cfg = json.load(fp)
                sched = cfg.get("sleep_schedule", "off")
                add_check("sleep:auto", "pass" if sched != "off" else "warn", sched)
            except (FileNotFoundError, json.JSONDecodeError):
                add_check("sleep:auto", "warn", "not configured")

            # Compile latency
            try:
                t0 = time.perf_counter()
                s.compile_context("test health check", budget=200)
                latency = (time.perf_counter() - t0) * 1000
                add_check("performance:compile_context", "pass" if latency < 500 else "warn", f"{latency:.1f}ms")
            except Exception as exc:
                add_check("performance:compile_context", "fail", str(exc))

        finally:
            s.close()
    except Exception as exc:
        add_check("store:load", "fail", str(exc))

    # Connected targets
    try:
        from installer import _detect_targets
        targets = _detect_targets(db_path)
        for name, installed in targets.items():
            add_check(f"target:{name}", "pass" if installed else "warn",
                       "installed" if installed else "not configured")
    except Exception:
        pass

    # MCP server reachable
    pid = _read_daemon_pid()
    if pid and _pid_is_running(pid):
        add_check("daemon:running", "pass", f"pid={pid}")
    else:
        add_check("daemon:running", "warn", "not running")

    if json_output:
        print(json.dumps(results, indent=2))
        return

    # Pretty print
    issues = 0
    print(_bold("🩺 Synapse Health Check"))
    print(f"  Store: {db_path}")
    print()
    for check in results:
        status = check["status"]
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(status, "❓")
        detail = f" — {check['detail']}" if check["detail"] else ""
        print(f"  {icon} {check['name']}{detail}")
        if status != "pass":
            issues += 1

    print()
    if issues:
        print(_yellow(f"  {issues} issue(s) found"))
    else:
        print(_green("  All checks passed!"))


# ─── Card share command ───────────────────────────────────────────────────────

def cmd_card_share(args):
    from synapse import Synapse
    from card_share import share_card, generate_caption

    db_path = _resolve_db_path(args)
    s = Synapse(db_path)
    try:
        fmt = getattr(args, "format", "markdown")
        if fmt == "md":
            fmt = "markdown"
        result = share_card(
            s,
            args.query,
            budget=args.budget,
            policy=args.policy,
            redact=getattr(args, "redact", None),
            format=fmt,
        )
        output = getattr(args, "output", None)
        if output:
            with open(output, "w", encoding="utf-8") as fp:
                fp.write(result)
            print(f"✅ Card written to {output}")
            try:
                from signing import sign_artifact
                sign_artifact(output)
            except Exception:
                pass
        else:
            print(result)

        # Generate caption if requested
        if getattr(args, "caption", False):
            pack = s.compile_context(args.query, budget=args.budget, policy=args.policy)
            caption = generate_caption(args.query, pack)
            print("\n" + "─" * 50)
            print("📋 Ready-to-post caption (copy & paste):")
            print("─" * 50)
            print(caption)
            print("─" * 50)
    finally:
        s.close()


# ─── Import wizard ────────────────────────────────────────────────────────────

def cmd_import_wizard(args):
    from importers import MemoryImporter
    from synapse import Synapse

    db_path = _resolve_db_path(args)

    print(_bold("🧙 Synapse Import Wizard"))
    print()
    print("  1) ChatGPT export")
    print("  2) Claude export")
    print("  3) Notes folder")
    print("  4) Clipboard")
    print("  5) JSONL/CSV")
    print()

    try:
        choice = input("Select source [1-5]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    source_map = {"1": "chatgpt", "2": "claude", "3": "notes", "4": "clipboard", "5": "jsonl"}
    source = source_map.get(choice)
    if not source:
        print("Invalid selection.")
        return

    path = None
    if source != "clipboard":
        # Auto-detect common paths
        suggestions = []
        import glob as _glob
        if source == "chatgpt":
            suggestions = _glob.glob(os.path.expanduser("~/Downloads/*chatgpt*.json"))
        elif source == "claude":
            suggestions = _glob.glob(os.path.expanduser("~/Downloads/*claude*.json"))

        if suggestions:
            print(f"\n  Auto-detected: {suggestions[0]}")
            try:
                use_detected = input(f"  Use this? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
            if use_detected in ("", "y", "yes"):
                path = suggestions[0]

        if not path:
            try:
                path = input("  File/folder path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
            if not path:
                print("No path provided.")
                return

    # Preview
    if path and os.path.isfile(path):
        size = os.path.getsize(path)
        lines = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                for _ in fp:
                    lines += 1
        except Exception:
            pass
        est_memories = max(1, lines // 10)
        print(f"\n  Preview: ~{lines:,} lines, ~{est_memories:,} est. memories, {_human_size(size)}")
    elif path and os.path.isdir(path):
        file_count = sum(1 for _ in Path(path).rglob("*.md")) if Path else 0
        print(f"\n  Preview: ~{file_count} markdown files found")

    # Policy selection
    print("\n  Apply policy first?")
    print("  1) minimal  2) private  3) work  4) ephemeral  5) none")
    try:
        policy_choice = input("  Policy [5]: ").strip() or "5"
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return
    policy_map = {"1": "minimal", "2": "private", "3": "work", "4": "ephemeral", "5": None}
    policy = policy_map.get(policy_choice)

    # Import
    print("\n  Importing...")
    s = Synapse(db_path)
    try:
        importer = MemoryImporter(s)
        if policy:
            s.policy_manager.set_preset(policy)

        if source == "clipboard":
            report = importer.from_clipboard()
        elif source in ("chatgpt", "claude"):
            report = importer.from_chat_transcript(path, format=source if source == "chatgpt" else "claude")
        elif source == "notes":
            report = importer.from_markdown_folder(path, recursive=True)
        elif source == "jsonl":
            if path.endswith(".csv"):
                report = importer.from_csv(path)
            else:
                report = importer.from_jsonl(path)
        else:
            print("Unsupported source.")
            return

        print(f"\n  ✅ Imported: {report.imported}")
        print(f"  ⏭  Skipped: {report.skipped}")
        print(f"  ❌ Errors: {report.errors}")
        print(f"  ⏱  Duration: {report.duration_ms:.1f}ms")
    finally:
        s.close()


# ─── Federation commands (standalone) ────────────────────────────────────────

def cmd_serve_federation(args):
    from federation.node import SynapseNode
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    host = args.host
    if getattr(args, "expose_network", False) and host in loopback_hosts:
        host = "0.0.0.0"
    if (not getattr(args, "expose_network", False)) and host not in loopback_hosts:
        print(
            "Error: refusing to bind federation server to a non-loopback host by default.\n"
            "       Re-run with --expose-network to explicitly opt in, or pass --fed-host 127.0.0.1.",
            file=sys.stderr,
        )
        sys.exit(2)
    node = SynapseNode(node_id=args.node_id, path=args.data or ":memory:",
                       auth_token=args.token)
    for ns in (args.share or []):
        node.share(ns)
    node.listen(port=args.port, host=host, expose_network=getattr(args, "expose_network", False))
    if args.discover:
        node.start_discovery(port=args.port)
    print(f"Synapse AI Memory node '{args.node_id}' listening on {host}:{args.port}")
    print(f"  Memories: {len(node.store.memories)}")
    print(f"  Shared namespaces: {sorted(node.store.shared_namespaces) or '(all)'}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        node.save()
        node.stop()


def cmd_push_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.push(args.peer, ns)
    print(f"Pushed {result['pushed']} memories to {args.peer}")


def cmd_pull_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.pull(args.peer, ns)
    print(f"Pulled {result['pulled']} memories from {args.peer}")
    if args.data:
        store.save()


def cmd_sync_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.sync(args.peer, ns)
    print(f"Synced with {args.peer}: pulled {result['pulled']}, pushed {result['pushed']}")
    if args.data:
        store.save()


def cmd_peers_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore("cli-temp")
    client = SyncClient(store)
    try:
        status = client.status(args.peer)
        print(f"Node: {status['node_id']}")
        print(f"  Memories: {status['memory_count']}")
        print(f"  Edges: {status['edge_count']}")
        print(f"  Root hash: {status['root_hash'][:16]}...")
        print(f"  Shared: {status.get('shared_namespaces', [])}")
        peers = client.list_peers(args.peer)
        peer_list = peers.get("peers", {})
        if peer_list:
            print(f"  Known peers:")
            for url, info in peer_list.items():
                print(f"    - {url} ({info.get('node_id', '?')})")
        else:
            print("  No known peers")
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='synapse',
        description='Synapse AI Memory — Private, portable AI memory engine with federation'
    )
    parser.add_argument('--host', default='127.0.0.1', help='Daemon host')
    parser.add_argument('--port', type=int, default=7654, help='Daemon port')
    parser.add_argument('--timeout', type=float, default=30.0, help='Request timeout')
    parser.add_argument('--node-id', default='default', help='Federation node ID')
    parser.add_argument('--data', '-d', help='Data file path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── Core commands (need daemon) ──
    p = subparsers.add_parser('remember', help='Store a new memory')
    p.add_argument('content', help='Memory content')
    p.add_argument('--type', default='fact')
    p.add_argument('--extract', action='store_true')
    p.add_argument('--no-extract', dest='extract', action='store_false')
    p.add_argument('--review', action='store_true', help='Route to review queue instead of saving directly')
    p.set_defaults(extract=None)

    p = subparsers.add_parser('recall', help='Search for memories')
    p.add_argument('query', nargs='?', default='')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--metadata', action='store_true')
    p.add_argument('--explain', action='store_true', help='Show score breakdown')
    p.add_argument('--at', default=None, help='Temporal query: date (2024-03), "all", or "latest"')

    p = subparsers.add_parser('list', help='List all memories')
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--sort', default='recent', choices=['recent', 'created', 'access_count'])

    p = subparsers.add_parser('count', help='Count total memories')
    p = subparsers.add_parser('policy', help='Manage memory policy presets')
    p.set_defaults(policy_action='show')
    p_policy = p.add_subparsers(dest='policy_action', help='Policy actions')

    p_set = p_policy.add_parser('set', help='Set active policy preset')
    p_set.add_argument('preset', choices=['minimal', 'private', 'work', 'ephemeral'])
    p_set.set_defaults(policy_action='set')

    p_apply = p_policy.add_parser('apply', help='Set active policy preset (alias for set)')
    p_apply.add_argument('preset', choices=['minimal', 'private', 'work', 'ephemeral'])
    p_apply.set_defaults(policy_action='set')

    p_list = p_policy.add_parser('list', help='List available policy presets')
    p_list.set_defaults(policy_action='list')

    p_show = p_policy.add_parser('show', help='Show active policy')
    p_show.set_defaults(policy_action='show')

    p = subparsers.add_parser('browse', help='Browse memories by concept')
    p.add_argument('--concept', required=True, help='Concept to browse')
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--offset', type=int, default=0)

    p = subparsers.add_parser('forget', help='Delete a memory')
    p.add_argument('id', type=int)
    
    # Natural language forget command
    p = subparsers.add_parser('nlforget', help='Forget using plain English')
    p.add_argument('command', help='Natural language forget command (e.g., "forget my phone number")')
    p.add_argument('--dry-run', action='store_true', help='Preview what would be deleted without actually deleting')
    p.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompts')
    
    # Inbox management commands
    inbox_parser = subparsers.add_parser('inbox', help='Memory inbox management')
    inbox_subparsers = inbox_parser.add_subparsers(dest='inbox_action', help='Inbox actions')
    
    p_inbox_list = inbox_subparsers.add_parser('list', help='List pending memories')
    p_inbox_list.add_argument('--limit', type=int, default=20, help='Maximum number of items to show')
    
    p_inbox_approve = inbox_subparsers.add_parser('approve', help='Approve a pending memory')
    p_inbox_approve.add_argument('item_id', help='ID of the item to approve')
    
    p_inbox_reject = inbox_subparsers.add_parser('reject', help='Reject a pending memory') 
    p_inbox_reject.add_argument('item_id', help='ID of the item to reject')
    
    p_inbox_redact = inbox_subparsers.add_parser('redact', help='Redact content and approve')
    p_inbox_redact.add_argument('item_id', help='ID of the item to redact')
    p_inbox_redact.add_argument('new_content', help='New redacted content')
    
    p_inbox_pin = inbox_subparsers.add_parser('pin', help='Pin and approve a memory')
    p_inbox_pin.add_argument('item_id', help='ID of the item to pin')
    
    p_inbox_query = inbox_subparsers.add_parser('query', help='Search pending memories')
    p_inbox_query.add_argument('query', help='Search query')
    p_inbox_query.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    
    # Vault management commands
    vault_parser = subparsers.add_parser('vault', help='Vault management')
    vault_subparsers = vault_parser.add_subparsers(dest='vault_action', help='Vault actions')
    
    p_vault_list = vault_subparsers.add_parser('list', help='List all vaults')
    
    p_vault_create = vault_subparsers.add_parser('create', help='Create a new vault')
    p_vault_create.add_argument('vault_id', help='Vault ID to create')
    p_vault_create.add_argument('--user-id', help='User ID to associate with vault')
    
    p_vault_switch = vault_subparsers.add_parser('switch', help='Switch to a different vault')
    p_vault_switch.add_argument('vault_id', help='Vault ID to switch to')

    p = subparsers.add_parser('link', help='Link two memories')
    p.add_argument('source', type=int)
    p.add_argument('target', type=int)
    p.add_argument('--edge-type', default='related')
    p.add_argument('--weight', type=float, default=1.0)

    p = subparsers.add_parser('concepts', help='List all concepts')
    p.add_argument('--limit', type=int, default=50)

    p = subparsers.add_parser('hot-concepts', help='Show most active concepts')
    p.add_argument('-k', type=int, default=10, help='Number of concepts')

    p = subparsers.add_parser('prune', help='Auto-prune weak, old memories (forgetting as a feature)')
    p.add_argument('--min-strength', type=float, default=0.1)
    p.add_argument('--min-access', type=int, default=0)
    p.add_argument('--max-age', type=float, default=90, help='Only prune memories older than N days')
    p.add_argument('--dry-run', action='store_true', help='Preview prune results')
    p.add_argument('--no-dry-run', dest='dry_run', action='store_false')
    p.set_defaults(dry_run=True)

    p = subparsers.add_parser('stats', help='Show statistics')
    p.add_argument('--db', help='Synapse AI Memory database path')
    subparsers.add_parser('ping', help='Ping the server')

    p = subparsers.add_parser('shutdown', help='Shutdown the server')
    p.add_argument('--force', action='store_true')

    # ── Temporal fact chain commands (standalone) ──
    p = subparsers.add_parser('history', help='Show fact evolution history')
    p.add_argument('query', help='Query to find the fact')
    p.add_argument('--db', help='Synapse AI Memory database path')

    p = subparsers.add_parser('timeline', help='Show timeline of fact changes')
    p.add_argument('--concept', default=None, help='Filter by concept')
    p.add_argument('--db', help='Synapse AI Memory database path')
    p.add_argument('--days', type=float, default=None, help='Only show entries within last N days')

    p = subparsers.add_parser('why', help='Explain why a memory was retrieved')
    p.add_argument('id', type=int, help='Memory ID')
    p.add_argument('--db', help='Synapse AI Memory database path')

    p = subparsers.add_parser('graph', help='Show concept neighborhood graph')
    p.add_argument('concept', help='Concept name')
    p.add_argument('--db', help='Synapse AI Memory database path')

    p = subparsers.add_parser('conflicts', help='List unresolved contradictions')
    p.add_argument('--db', help='Synapse AI Memory database path')

    p = subparsers.add_parser('beliefs', help='Show current worldview beliefs')
    p.add_argument('--db', help='Synapse AI Memory database path')

    # ── Consolidate command (standalone) ──
    p = subparsers.add_parser('consolidate', help='Consolidate similar memories')
    p.add_argument('--db', help='Synapse AI Memory database path')
    p.add_argument('--dry-run', action='store_true', help='Preview without modifying')
    p.add_argument('--min-cluster', type=int, default=3, help='Min cluster size')
    p.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    p.add_argument('--max-age-days', type=float, default=None, help='Only consolidate memories older than N days')

    p = subparsers.add_parser('sleep', help='Run sleep maintenance pipeline')
    p.add_argument('--db', help='Synapse AI Memory database path')
    p.add_argument('--verbose', action='store_true', help='Include verbose sleep internals')
    p.add_argument('--digest', action='store_true', help='Emit human-readable sleep digest')

    # ── Portable Format commands ──
    p = subparsers.add_parser('export', help='Export database to .synapse file')
    p.add_argument('output', help='Output .synapse file path')
    p.add_argument('--db', help='Synapse AI Memory database path')
    p.add_argument('--since', help='Export since date (ISO)')
    p.add_argument('--until', help='Export until date (ISO)')
    p.add_argument('--concepts', help='Filter by concepts (comma-separated)')
    p.add_argument('--tags', help='Filter by tags (comma-separated)')
    p.add_argument('--types', help='Filter by memory types (comma-separated)')
    p.add_argument('--source', help='Source agent identifier')

    p = subparsers.add_parser('import', help='Import memories from a file source')
    p.add_argument('input', help='Import type (chat, notes, clipboard, jsonl, csv) or .synapse path')
    p.add_argument('path', nargs='?', help='Source path/folder for non-clipboard imports')
    p.add_argument('--format', choices=['auto', 'chatgpt', 'claude', 'whatsapp', 'plain'], default='auto',
                   help='Chat transcript format (for `chat` import)')
    p.add_argument('--db', help='Target database path')
    p.add_argument('--no-dedup', action='store_true')
    p.add_argument('--threshold', type=float, default=0.85)
    p.add_argument('--recursive', action='store_true', help='Recurse folders when importing notes')
    p.add_argument('--text-field', default='text', help='Field name with memory text in JSONL rows')
    p.add_argument('--text-column', default='text', help='Column with memory text in CSV rows')

    p = subparsers.add_parser('inspect-export', help='Inspect .synapse file')
    p.add_argument('input', help='.synapse file path')

    p = subparsers.add_parser('merge', help='Merge multiple .synapse files')
    p.add_argument('inputs', nargs='+', help='.synapse files to merge')
    p.add_argument('-o', '--output', default='merged.synapse')
    p.add_argument('--conflict', choices=['newer_wins', 'keep_both'], default='newer_wins')
    p.add_argument('--threshold', type=float, default=0.85)

    p = subparsers.add_parser('diff', help='Compare two .synapse files')
    p.add_argument('file_a')
    p.add_argument('file_b')
    p.add_argument('--threshold', type=float, default=0.85)

    p = subparsers.add_parser('pack', help='Create and compare topic brain packs')
    p_mode = p.add_mutually_exclusive_group(required=True)
    p_mode.add_argument('--topic', help='Build a brain pack for this topic')
    p_mode.add_argument('--replay', help='Replay a .brain file against current Synapse')
    p_mode.add_argument('--diff', nargs=2, metavar=('FILE1', 'FILE2'), help='Compare two .brain files')
    p_mode.add_argument('--list', action='store_true', help='List saved brain packs')
    p.add_argument('--range', default='30d', help='History window for build mode (e.g. 30d, 2w, 90)')
    p.add_argument('--output', help='Output .brain file for pack mode')
    p.add_argument('--pack-dir', help='Directory for storing and listing packs')
    p.add_argument('--db', help='Synapse database path')

    # ── Context card commands ─────────────────────────────────────────────
    p = subparsers.add_parser('card', help='Create and manage context cards')
    card_sub = p.add_subparsers(dest='card_action', help='Card actions')

    p_create = card_sub.add_parser('create', help='Create a context card from a query')
    p_create.add_argument('query')
    p_create.add_argument('--budget', type=int, default=2000)
    p_create.add_argument('--db', help='Synapse AI Memory database path')
    p_create.set_defaults(card_action='create')

    p_show = card_sub.add_parser('show', help='Show a stored card by id')
    p_show.add_argument('card_id')
    p_show.add_argument('--db', help='Synapse AI Memory database path')
    p_show.set_defaults(card_action='show')

    p_export = card_sub.add_parser('export', help='Export card deck to file')
    p_export.add_argument('path')
    p_export.add_argument('--db', help='Synapse AI Memory database path')
    p_export.set_defaults(card_action='export')

    p_share = card_sub.add_parser('share', help='Generate shareable memory card')
    p_share.add_argument('query')
    p_share.add_argument('--budget', type=int, default=1600)
    p_share.add_argument('--policy', default='balanced')
    p_share.add_argument('--redact', choices=['pii', 'names', 'numbers'], default=None)
    p_share.add_argument('--format', choices=['md', 'markdown', 'html'], default='markdown')
    p_share.add_argument('--output', help='Output file path')
    p_share.add_argument('--caption', action='store_true', help='Generate ready-to-post social media caption')
    p_share.add_argument('--db', help='Synapse AI Memory database path')
    p_share.set_defaults(card_action='share')

    p_checkpoint = subparsers.add_parser('checkpoint', help='Create and manage checkpoints')
    p_checkpoint.add_argument('--db', help='Synapse AI Memory database path')
    cp_sub = p_checkpoint.add_subparsers(dest='checkpoint_action', help='Checkpoint actions')

    cp_create = cp_sub.add_parser('create', help='Create a checkpoint')
    cp_create.add_argument('name')
    cp_create.add_argument('--desc', default='')
    cp_create.set_defaults(checkpoint_action='create')

    cp_list = cp_sub.add_parser('list', help='List checkpoints')
    cp_list.set_defaults(checkpoint_action='list')

    cp_diff = cp_sub.add_parser('diff', help='Compare two checkpoints')
    cp_diff.add_argument('a')
    cp_diff.add_argument('b')
    cp_diff.set_defaults(checkpoint_action='diff')

    cp_restore = cp_sub.add_parser('restore', help='Restore to checkpoint state')
    cp_restore.add_argument('name')
    cp_restore.add_argument('--confirm', action='store_true', help='Skip confirmation and apply restore')
    cp_restore.set_defaults(checkpoint_action='restore')

    cp_delete = cp_sub.add_parser('delete', help='Delete a checkpoint')
    cp_delete.add_argument('name')
    cp_delete.set_defaults(checkpoint_action='delete')

    p = subparsers.add_parser('demo', help='Run built-in demo scenarios')
    p.add_argument('--scenario', default='diet', choices=['diet', 'travel', 'project', 'all'],
                   help='Demo scenario to run')
    p.add_argument('--markdown', action='store_true',
                   help='Emit markdown output for sharing')

    # ── Appliance commands ──
    p = subparsers.add_parser('inspect', help='Inspect MCP tool catalog and store')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='MCP storage path')
    p.add_argument('--json', action='store_true', help='Output JSON summary')

    p = subparsers.add_parser('inspector', help='Launch local inspector UI')
    p.add_argument('--port', type=int, default=9471, help='Local inspector port')
    p.add_argument('--db', help='Synapse database path')

    p = subparsers.add_parser('doctor', help='Check Synapse health and client integrations')
    p.add_argument('--db', help='Memory store path (default: ~/.synapse)')
    p.add_argument('--json', action='store_true', help='Machine-readable JSON output')

    p = subparsers.add_parser('serve', help='Start MCP memory appliance')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Storage path')
    p.add_argument('--port', type=int, default=8765, help='HTTP JSON-RPC port (requires --http)')
    p.add_argument('--http', action='store_true', help='Run as HTTP server on localhost')

    p = subparsers.add_parser('start', help='Golden path: start Synapse with status')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Storage path')
    p.add_argument('--port', type=int, default=APPLIANCE_DAEMON_PORT, help='Service port')
    p.add_argument('--inspect', dest='inspect_flag', action='store_true', help='Launch inspector web UI')
    p.add_argument('--sleep', dest='sleep_schedule', choices=['daily', 'hourly', 'off'], default=None, help='Auto-sleep schedule')

    p = subparsers.add_parser('up', help='Start Synapse appliance daemon')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Storage path')
    p.add_argument('--port', type=int, default=APPLIANCE_DAEMON_PORT, help='Service port')
    p.add_argument('--mode', choices=['appliance', 'full'], default='appliance', help='Daemon startup mode')

    p = subparsers.add_parser('down', help='Stop Synapse appliance daemon')

    p = subparsers.add_parser('status', help='Show Synapse appliance daemon status')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Storage path')
    p.add_argument('--port', type=int, default=APPLIANCE_DAEMON_PORT, help='Service port')

    # ── Federation commands ──
    p = subparsers.add_parser('serve-federation', help='Start federation server')
    p.add_argument('--fed-port', '-p', type=int, default=9470, dest='port')
    p.add_argument('--fed-host', default='127.0.0.1', dest='host', help='Bind address (default: 127.0.0.1)')
    p.add_argument(
        '--expose-network',
        action='store_true',
        help='Opt-in: allow binding federation to a non-loopback interface (default binds localhost only)',
    )
    p.add_argument('--share', nargs='*', help='Namespaces to share')
    p.add_argument('--discover', action='store_true')
    p.add_argument('--token', help='Auth token')

    p = subparsers.add_parser('push', help='Push memories to peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('pull', help='Pull memories from peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('sync', help='Bidirectional sync with peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('peers', help='Show peer info')
    p.add_argument('peer', help='Peer URL to query')

    p = subparsers.add_parser('install', help='Install Synapse client integrations')
    p.add_argument('target', nargs='?', help='Install target (claude, cursor, windsurf, continue, openclaw, nanoclaw, telegram, ollama, all)')
    p.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Synapse database path')
    p.add_argument('--list', action='store_true', help='Show available install targets')
    p.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    p.add_argument('--verify-only', action='store_true', help='Only verify existing installation')

    p = subparsers.add_parser('import-wizard', help='Interactive import wizard')
    p.add_argument('--db', help='Synapse AI Memory database path')

    # ── Uninstall command ──
    p = subparsers.add_parser('uninstall', help='Remove Synapse client integrations')
    p.add_argument('target', choices=['claude', 'cursor', 'windsurf', 'continue', 'openclaw', 'nanoclaw', 'telegram', 'ollama', 'all'], help='Uninstall target')

    # ── Service command ──
    p_service = subparsers.add_parser('service', help='Manage autostart service')
    service_sub = p_service.add_subparsers(dest='service_action', help='Service actions')

    p_si = service_sub.add_parser('install', help='Install autostart service')
    p_si.add_argument('--db', default=APPLIANCE_DB_DEFAULT, help='Synapse database path')
    p_si.add_argument('--sleep', dest='sleep_schedule', default='daily', choices=['daily', 'hourly', 'off'])
    p_si.set_defaults(service_action='install')

    p_su = service_sub.add_parser('uninstall', help='Remove autostart service')
    p_su.set_defaults(service_action='uninstall')

    p_ss = service_sub.add_parser('status', help='Check service status')
    p_ss.set_defaults(service_action='status')

    # ── Memory Router commands ──
    p = subparsers.add_parser('ingest', help='Ingest text through memory router')
    p.add_argument('text', nargs='?', help='Text to ingest')
    p.add_argument('--file', help='File to ingest')
    p.add_argument('--policy', choices=['auto', 'minimal', 'review', 'off'], default='auto', 
                   help='Memory router policy (default: auto)')
    p.add_argument('--db', help='Synapse database path')

    p = subparsers.add_parser('watch', help='Watch streams and feed to memory router')
    watch_group = p.add_mutually_exclusive_group(required=True)
    watch_group.add_argument('--stdin', action='store_true', help='Watch stdin stream')
    watch_group.add_argument('--file', help='Watch file for new lines (tail -f style)')
    watch_group.add_argument('--clipboard', action='store_true', help='Watch clipboard')
    p.add_argument('--policy', choices=['auto', 'minimal', 'review', 'off'], default='auto',
                   help='Memory router policy (default: auto)')
    p.add_argument('--batch-size', type=int, default=5, help='Messages per batch (default: 5)')
    p.add_argument('--batch-timeout', type=float, default=30.0, help='Batch timeout in seconds (default: 30)')
    p.add_argument('--interval', type=float, default=2.0, help='Clipboard poll interval in seconds (default: 2)')
    p.add_argument('--tag', action='append', dest='tags', help='Tag for captured memories')
    p.add_argument('--db', help='Synapse database path')

    p = subparsers.add_parser('clip', help='Capture text to memory via router')
    p.add_argument('text', nargs='?', default=None, help='Text to capture (or pipe via stdin)')
    p.add_argument('--policy', choices=['auto', 'minimal', 'review', 'off'], default='auto',
                   help='Memory router policy (default: auto)')
    p.add_argument('--tag', action='append', dest='tags', help='Tag for captured memory')
    p.add_argument('--db', help='Synapse database path')

    p = subparsers.add_parser('capture', help='Manage memory capture mode')
    p.add_argument('--mode', choices=['auto', 'minimal', 'review', 'off'], 
                   help='Set capture mode (auto/minimal/review/off)')
    # When no --mode specified, shows current mode

    # ── Review queue commands ──
    p_review = subparsers.add_parser('review', help='Memory review queue')
    review_sub = p_review.add_subparsers(dest='review_action', help='Review actions')

    p_rl = review_sub.add_parser('list', help='Show pending items')
    p_rl.set_defaults(review_action='list')

    p_ra = review_sub.add_parser('approve', help='Approve pending item(s)')
    p_ra.add_argument('item_id', help='Item ID or "all"')
    p_ra.add_argument('--db', help='Synapse database path')
    p_ra.set_defaults(review_action='approve')

    p_rr = review_sub.add_parser('reject', help='Reject a pending item')
    p_rr.add_argument('item_id', help='Item ID')
    p_rr.set_defaults(review_action='reject')

    p_rc = review_sub.add_parser('count', help='Count pending items')
    p_rc.set_defaults(review_action='count')

    # ── Sign & verify commands ──
    p = subparsers.add_parser('sign', help='Sign a .brain or .synapse file')
    p.add_argument('file', help='File to sign')

    p = subparsers.add_parser('verify', help='Verify a signed file')
    p.add_argument('file', help='File to verify')

    p = subparsers.add_parser('bench', help='Run consumer-facing benchmark suite')
    p.add_argument('--scenario', choices=['recall', 'timetravel', 'contradictions'],
                   help='Run a single scenario')
    p.add_argument('--output', help='Output directory for artifacts')
    p.add_argument('--format', dest='bench_format', choices=['md', 'json', 'both'],
                   default='both', help='Output format')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Standalone commands (no daemon needed)
    standalone = {
        'bench': cmd_bench,
        'history': cmd_history,
        'why': cmd_why,
        'graph': cmd_graph,
        'conflicts': cmd_conflicts,
        'beliefs': cmd_beliefs,
        'timeline': cmd_timeline,
        'consolidate': cmd_consolidate,
        'sleep': cmd_sleep,
        'export': cmd_export,
        'import': cmd_import,
        'inspect': cmd_inspect,
        'inspector': cmd_inspector,
        'inspect-export': cmd_inspect_export,
        'checkpoint': cmd_checkpoint,
        'merge': cmd_merge,
        'diff': cmd_diff,
        'pack': cmd_pack,
        'install': cmd_install,
        'doctor': run_doctor,
        'start': cmd_start,
        'import-wizard': cmd_import_wizard,
        'up': cmd_up,
        'down': cmd_down,
        'status': cmd_status,
        'serve': cmd_serve,
        'serve-federation': cmd_serve_federation,
        'card': cmd_card,
        'push': cmd_push_fed,
        'pull': cmd_pull_fed,
        'sync': cmd_sync_fed,
        'peers': cmd_peers_fed,
        'demo': cmd_demo,
        'uninstall': cmd_uninstall,
        'service': cmd_service,
        'watch': cmd_watch,
        'clip': cmd_clip,
        'ingest': cmd_ingest,
        'capture': cmd_capture,
        'review': cmd_review,
        'sign': cmd_sign,
        'verify': cmd_verify,
    }

    if args.command in standalone:
        standalone[args.command](args)
        return

    if args.command == 'stats' and getattr(args, 'db', None):
        cmd_stats(args)
        return

    # Handle --review flag for remember command
    if args.command == 'remember' and getattr(args, 'review', False):
        from review_queue import ReviewQueue
        rq = ReviewQueue(None)
        item_id = rq.submit(args.content, metadata={"memory_type": args.type})
        print(f"📋 Submitted for review (id: {item_id})")
        print(f"   Run: synapse review approve {item_id}")
        return

    # Daemon-dependent commands
    try:
        client = SynapseClient(host=args.host, port=args.port, timeout=args.timeout)
        client.connect()
    except SynapseConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Make sure the Synapse AI Memory daemon is running on {args.host}:{args.port}")
        sys.exit(1)

    daemon_commands = {
        'remember': cmd_remember,
        'recall': cmd_recall,
        'forget': cmd_forget,
        'nlforget': cmd_nlforget,
        'inbox': cmd_inbox,
        'vault': cmd_vault,
        'link': cmd_link,
        'concepts': cmd_concepts,
        'hot-concepts': cmd_hot_concepts,
        'prune': cmd_prune,
        'stats': cmd_stats,
        'ping': cmd_ping,
        'shutdown': cmd_shutdown,
        'list': cmd_list,
        'count': cmd_count,
        'browse': cmd_browse,
        'policy': cmd_policy,
    }

    try:
        if args.command in daemon_commands:
            daemon_commands[args.command](args, client)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
