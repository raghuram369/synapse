#!/usr/bin/env python3
"""Health check helpers for `synapse doctor`.

This module is intentionally dependency-free (stdlib only) and read-only:
all checks inspect local files / process state and avoid mutating state.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

Status = str  # one of: "ok", "warn", "error"
CheckResult = tuple[Status, str]

# Optional network timeout for Ollama.
_OLLAMA_URL = os.environ.get("SYNAPSE_OLLAMA_URL", "http://127.0.0.1:11434")
_OLLAMA_TIMEOUT = float(os.environ.get("SYNAPSE_OLLAMA_TIMEOUT", "2.0"))


def _home_path(path: str) -> str:
    return os.path.expanduser(path)


def _tilde(path: str) -> str:
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _read_version() -> str:
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fp:
        text = fp.read()
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise RuntimeError("__version__ is missing in __init__.py")
    return match.group(1)


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _contains_server_entry(payload: Dict[str, Any], server_name: str = "synapse") -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("mcpServers", "mcp_servers", "servers", "server"):
        section = payload.get(key)
        if isinstance(section, dict) and server_name in section:
            return True
    for key in ("mcp", "config", "settings"):
        section = payload.get(key)
        if isinstance(section, dict):
            for nested in ("servers", "mcpServers"):
                nested_section = section.get(nested)
                if isinstance(nested_section, dict) and server_name in nested_section:
                    return True
    return False


def _human_bytes(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _store_size(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                candidate = os.path.join(root, name)
                try:
                    total += os.path.getsize(candidate)
                except OSError:
                    pass
        return total

    total = 0
    for candidate in (path, f"{path}.log", f"{path}.snapshot", f"{path}.state"):
        try:
            total += os.path.getsize(candidate)
        except OSError:
            continue
    return total


def _resolve_store_path(configured: Optional[str]) -> tuple[str, str]:
    path = _home_path(configured) if configured else _home_path("~/.synapse")
    if os.path.isdir(path):
        candidate = os.path.join(path, "synapse")
        if any(os.path.exists(f) for f in (candidate, f"{candidate}.log", f"{candidate}.snapshot")):
            return candidate, path
    return path, path


def check_synapse_version() -> CheckResult:
    try:
        return "ok", f"Synapse {_read_version()}"
    except Exception as exc:
        return "error", f"Unable to read version: {exc}"


def check_memory_store(configured_path: Optional[str]) -> CheckResult:
    base_path, display_path = _resolve_store_path(configured_path)
    try:
        status: Status = "ok"
        extra_message = ""

        store_paths = (base_path, f"{base_path}.log", f"{base_path}.snapshot")
        has_store_artifact = any(os.path.exists(p) for p in store_paths)
        if not has_store_artifact and not os.path.isdir(base_path):
            status = "warn"
            return status, f"{_tilde(display_path)}: not initialized (0 memories, 0.0 B)"

        from synapse import Synapse

        try:
            instance = Synapse(base_path)
            try:
                count = instance.count()
            finally:
                instance.close()
        except Exception as exc:
            return "error", f"{_tilde(display_path)}: unable to read store ({exc})"

        if not has_store_artifact and os.path.isdir(base_path):
            status = "warn"
            extra_message = "not initialized"

        size = _store_size(base_path if os.path.exists(base_path) else display_path)
        summary = f"{count} memories, {_human_bytes(size)}"
        if extra_message:
            summary = f"{extra_message} ({summary})"
        return status, f"{_tilde(display_path)} ({summary})"
    except Exception as exc:
        return "error", f"{_tilde(display_path)}: {exc}"


def check_claude_desktop() -> CheckResult:
    if os.name == "nt":
        root = os.environ.get("APPDATA")
        if not root:
            return "error", "Windows APPDATA not set; cannot locate Claude config"
        candidate = os.path.join(root, "Claude", "claude_desktop_config.json")
    else:
        candidate = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    payload = _load_json_file(candidate)
    if payload is None:
        return "warn", "not configured (run: synapse install claude)"
    if _contains_server_entry(payload, "synapse"):
        return "ok", "MCP configured"
    return "warn", "not configured"


def _possible_cursor_paths() -> List[str]:
    paths = [
        os.path.expanduser("~/.cursor/mcp.json"),
        os.path.expanduser("~/.cursor/config.json"),
        os.path.expanduser("~/.config/Cursor/User/settings.json"),
        os.path.expanduser("~/Library/Application Support/Cursor/User/settings.json"),
        os.path.expanduser("~/Library/Application Support/Cursor/User/globalStorage/cursor/settings.json"),
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(os.path.join(appdata, "Cursor", "User", "settings.json"))
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        paths.append(os.path.join(localapp, "Cursor", "User", "settings.json"))
    return paths


def _possible_windsurf_paths() -> List[str]:
    paths = [
        os.path.expanduser("~/.windsurf/mcp.json"),
        os.path.expanduser("~/.config/Windsurf/User/settings.json"),
        os.path.expanduser("~/Library/Application Support/Windsurf/User/settings.json"),
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(os.path.join(appdata, "Windsurf", "User", "settings.json"))
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        paths.append(os.path.join(localapp, "Windsurf", "User", "settings.json"))
    return paths


def check_cursor() -> CheckResult:
    for path in _possible_cursor_paths():
        payload = _load_json_file(path)
        if payload is None:
            continue
        if _contains_server_entry(payload, "synapse"):
            return "ok", "MCP configured"
    return "warn", "not configured (run: synapse install cursor)"


def check_windsurf() -> CheckResult:
    for path in _possible_windsurf_paths():
        payload = _load_json_file(path)
        if payload is None:
            continue
        if _contains_server_entry(payload, "synapse"):
            return "ok", "MCP configured"
    return "error", "not detected"


def check_continue() -> CheckResult:
    path = os.path.expanduser("~/.continue/config.json")
    payload = _load_json_file(path)
    if payload is None:
        return "warn", "not configured"
    if isinstance(payload, dict) and (
        payload.get("models")
        or payload.get("modelsPath")
        or payload.get("mcpServers")
    ):
        return "ok", "configured"
    return "warn", "config file exists but no known models/mcp settings"


def check_ollama() -> CheckResult:
    endpoint = f"{_OLLAMA_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=_OLLAMA_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models")
        if not isinstance(models, list):
            return "warn", "reachable, response format unexpected"
        model_names = [
            item.get("name", "") for item in models
            if isinstance(item, dict) and item.get("name")
        ]
        if model_names:
            shown = ", ".join(model_names)
            return "ok", f"running ({shown})"
        return "warn", "running (no models installed)"
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        return "error", f"not running ({exc})"
    except Exception as exc:
        return "error", f"not running ({exc})"


def check_privacy_preset(configured_path: Optional[str]) -> CheckResult:
    path, _display = _resolve_store_path(configured_path)
    has_store_artifact = any(
        os.path.exists(candidate)
        for candidate in (path, f"{path}.log", f"{path}.snapshot")
    )
    if not has_store_artifact and not os.path.isdir(path):
        return "warn", "no active policy (run: synapse policy set private)"

    try:
        from synapse import Synapse

        instance = Synapse(path)
        try:
            policy = instance.policy()
        finally:
            instance.close()
        if isinstance(policy, dict) and policy.get("name"):
            return "ok", f"{policy['name']} preset"
        return "warn", "no active policy (run: synapse policy set private)"
    except Exception as exc:
        return "error", f"unable to read policy: {exc}"


def check_inbox_mode() -> CheckResult:
    pending_dir = os.path.expanduser("~/.synapse/pending")
    if not os.path.isdir(pending_dir):
        return "warn", "not enabled"
    try:
        pending = [
            f for f in os.listdir(pending_dir)
            if f.endswith('.json')
        ]
        return "ok", f"enabled ({len(pending)} pending item{'s' if len(pending)!=1 else ''})"
    except OSError as exc:
        return "error", f"pending dir unreadable ({exc})"


def check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        return "error", f"Python {major}.{minor} (upgrade to >=3.10 required)"
    return "ok", f"{major}.{minor}"


def gather_checks(configured_store: Optional[str]) -> List[Dict[str, Any]]:
    label_checks: list[tuple[str, CheckResult]] = [
        ("Synapse", check_synapse_version()),
        ("Memory store", check_memory_store(configured_store)),
        ("Claude Desktop", check_claude_desktop()),
        ("Cursor", check_cursor()),
        ("Windsurf", check_windsurf()),
        ("VS Code / Continue", check_continue()),
        ("Ollama", check_ollama()),
        ("Privacy", check_privacy_preset(configured_store)),
        ("Inbox mode", check_inbox_mode()),
        ("Python", check_python_version()),
    ]

    results: List[Dict[str, Any]] = []
    for name, (status, message) in label_checks:
        results.append({"name": name, "status": status, "message": message})
    return results


def run_doctor(args) -> None:
    configured_store = getattr(args, "db", None)
    checks = gather_checks(configured_store)

    if getattr(args, "json", False):
        print(json.dumps(checks, indent=2, sort_keys=True))
        return

    icon = {
        "ok": "✅",
        "warn": "⚠️",
        "error": "❌",
    }

    print("🧠 Synapse Doctor")
    print("═" * 47)
    print()

    for check in checks:
        status = check["status"]
        name = check["name"]
        message = check["message"]
        prefix = icon.get(status, "❓")
        print(f"  {prefix} {name}: {message}")

    print()
    print("═" * 47)
    connected = sum(1 for item in checks if item["status"] == "ok")
    available = sum(1 for item in checks if item["status"] == "warn")
    not_found = sum(1 for item in checks if item["status"] == "error")
    print(f"  {connected} connected  {available} available  {not_found} not found")


if __name__ == "__main__":  # pragma: no cover
    class Dummy:
        db = None
        json = False

    run_doctor(Dummy())
