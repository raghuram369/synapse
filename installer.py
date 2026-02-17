#!/usr/bin/env python3
"""Cross-client installation helpers for Synapse."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


def _python_binary() -> str:
    return "python3"


def _mcp_server_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))


def _default_data_dir() -> str:
    """Return the canonical shared Synapse data directory."""
    return os.path.expanduser("~/.synapse")


def _resolve_db_path(db_path: str) -> str:
    """Ensure db_path is absolute. Default to ~/.synapse if relative/default."""
    if db_path in ("./synapse_store", "synapse_store", ""):
        return _default_data_dir()
    return os.path.abspath(db_path)


def _mcp_command() -> List[str]:
    """Return the best command to launch synapse-mcp.

    Uses the current Python interpreter (absolute path) + mcp_server module.
    This is the most reliable approach — no PATH issues, no uvx/pipx dependency.
    """
    return [sys.executable, _mcp_server_path()]


def _mcp_payload(db_path: str) -> Dict[str, Any]:
    db_path = _resolve_db_path(db_path)
    cmd = _mcp_command()
    return {
        "command": cmd[0],
        "args": cmd[1:] + ["--data-dir", db_path],
    }


def _continue_payload(db_path: str) -> Dict[str, Any]:
    db_path = _resolve_db_path(db_path)
    cmd = _mcp_command()
    return {
        "transport": {
            "type": "stdio",
            "command": cmd[0],
            "args": cmd[1:] + ["--data-dir", db_path],
        }
    }


def _python_platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return "linux"


def _cursor_candidates() -> list[str]:
    home = os.path.expanduser("~")
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        candidate = os.path.join(appdata, "Cursor", "User", "globalStorage", "mcp.json") if appdata else ""
        return [candidate or os.path.join(home, "AppData", "Roaming", "Cursor", "User", "globalStorage", "mcp.json")]
    if sys.platform.startswith("darwin"):
        return [
            os.path.join(home, ".cursor", "mcp.json"),
            os.path.join(home, "Library", "Application Support", "Cursor", "User", "globalStorage", "mcp.json"),
        ]
    return [os.path.join(home, ".config", "cursor", "mcp.json")]


def _windsurf_candidates() -> list[str]:
    home = os.path.expanduser("~")
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        candidate = os.path.join(appdata, "Windsurf", "User", "globalStorage", "mcp.json") if appdata else ""
        return [candidate or os.path.join(home, "AppData", "Roaming", "Windsurf", "User", "globalStorage", "mcp.json")]
    if sys.platform.startswith("darwin"):
        return [
            os.path.join(home, ".windsurf", "mcp.json"),
            os.path.join(home, "Library", "Application Support", "Windsurf", "User", "globalStorage", "mcp.json"),
        ]
    return [os.path.join(home, ".config", "windsurf", "mcp.json")]


def _continue_candidates() -> list[str]:
    home = os.path.expanduser("~")
    return [os.path.join(home, ".continue", "config.json")]


def _claude_config_path() -> str:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable is required for Claude config on Windows.")
        return os.path.join(appdata, "Claude", "claude_desktop_config.json")
    return os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")


def _resolve_path(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def _backup_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    backup = f"{path}.backup"
    shutil.copy2(path, backup)
    return backup


def _ensure_synapse_mcp(payload: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    config = dict(payload)
    mcp_servers = config.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    mcp_servers["synapse"] = _mcp_payload(db_path)
    config["mcpServers"] = mcp_servers
    return config


def _ensure_synapse_continue(payload: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    config = dict(payload)
    experimental = config.get("experimental")
    if not isinstance(experimental, dict):
        experimental = {}
    servers = experimental.get("modelContextProtocolServers")
    if not isinstance(servers, list):
        servers = []
    filtered = []
    for item in servers:
        if not isinstance(item, dict):
            continue
        transport = item.get("transport")
        if not isinstance(transport, dict):
            continue
        cmd = transport.get("command", "")
        if cmd == "synapse-mcp" or (cmd == _python_binary() and isinstance(transport.get("args"), list) and _mcp_server_path() in transport.get("args", [])):
            continue
        filtered.append(item)
    filtered.append(_continue_payload(db_path))
    experimental["modelContextProtocolServers"] = filtered
    config["experimental"] = experimental
    return config


def _is_synapse_continue_entry(item: dict[str, Any]) -> bool:
    transport = item.get("transport")
    if not isinstance(transport, dict):
        return False
    cmd = transport.get("command", "")
    if cmd == "synapse-mcp":
        return True
    return cmd == _python_binary() and isinstance(transport.get("args"), list) and _mcp_server_path() in transport.get("args", [])


def _verify_mcp_file(path: str, db_path: str) -> tuple[bool, str]:
    del db_path
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            return False, "config format is not a JSON object"
        servers = data.get("mcpServers")
        if isinstance(servers, dict) and isinstance(servers.get("synapse"), dict):
            return True, "synapse is configured correctly"
        return False, "synapse not found in mcpServers"
    except FileNotFoundError:
        return False, "config file not found"
    except json.JSONDecodeError:
        return False, "config file is not valid JSON"


def _verify_continue_file(path: str, db_path: str) -> tuple[bool, str]:
    del db_path
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            return False, "config format is not a JSON object"
        exp = data.get("experimental")
        if not isinstance(exp, dict):
            return False, "experimental config not found"
        servers = exp.get("modelContextProtocolServers")
        if isinstance(servers, list) and any(isinstance(s, dict) and _is_synapse_continue_entry(s) for s in servers):
            return True, "synapse is configured correctly"
        return False, "synapse not found in modelContextProtocolServers"
    except FileNotFoundError:
        return False, "config file not found"
    except json.JSONDecodeError:
        return False, "config file is not valid JSON"


def _openclaw_workspace_root() -> str:
    return os.path.expanduser("~/.openclaw/workspace/skills")


def _nanoclaw_workspace_root() -> str:
    return os.path.expanduser("~/.nanoclaw/workspace/skills")


def _skill_dir(workspace_root: str) -> str:
    return os.path.join(workspace_root, "synapse")


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content.rstrip() + "\n")


def _openclaw_skill_markdown() -> str:
    return """---
name: synapse
description: Privacy-first AI memory engine. Gives your agent persistent, intelligent recall.
---

# Synapse

Privacy-first AI memory package for OpenClaw agents.

## Usage

- Use `/mem` commands directly for quick operations:
  - `/mem remember <text>`
  - `/mem recall <query> [k]`
  - `/mem compile_context <query>`
  - `/mem timeline`
- Or use MCP tools if your setup exposes them.

## Installation

```bash
pip install synapse-ai-memory
synapse --help
```
"""


def _openclaw_manifest() -> dict[str, Any]:
    return {
        "name": "synapse-ai-memory",
        "version": "0.8.0",
        "description": "Privacy-first AI memory engine for agents",
        "permissions": {
            "filesystem": {
                "read": ["~/.synapse/", "./synapse_store/"],
                "write": ["~/.synapse/", "./synapse_store/"],
            },
            "network": {
                "bind": ["127.0.0.1:9470"],
                "connect": [],
            },
            "shell": False,
            "env_vars": [],
        },
    }


def _openclaw_setup_sh() -> str:
    return """#!/usr/bin/env bash
pip install synapse-ai-memory>=0.7.0 2>/dev/null || pip3 install synapse-ai-memory>=0.7.0
echo 'Synapse AI Memory installed. Run: synapse doctor'
"""


def _install_openclaw_into(path_root: str, print_message: str) -> None:
    skill_root = _skill_dir(path_root)
    _write_text(os.path.join(skill_root, "SKILL.md"), _openclaw_skill_markdown())
    _write_text(os.path.join(skill_root, "manifest.json"), json.dumps(_openclaw_manifest(), indent=2))
    _write_text(os.path.join(skill_root, "setup.sh"), _openclaw_setup_sh())
    print(print_message)


def _ensure_mcp_package():
    """Ensure the 'mcp' package is installed (required for synapse-mcp server)."""
    try:
        import mcp  # noqa: F401
        return
    except ImportError:
        pass
    print("Installing mcp package (required for MCP server)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "mcp>=1.0.0"],
            stdout=subprocess.DEVNULL,
        )
        print("mcp package installed.")
    except subprocess.CalledProcessError:
        print("WARNING: Could not install 'mcp' package automatically.")
        print("  Run manually: pip install 'mcp>=1.0.0'")


def install_claude(db_path):
    """Auto-configure Claude Desktop MCP."""
    _ensure_mcp_package()
    config_path = _claude_config_path()
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_mcp(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Claude Desktop. Restart Claude to activate.")


def install_cursor(db_path):
    """Auto-configure Cursor MCP settings."""
    _ensure_mcp_package()
    config_path = _resolve_path(_cursor_candidates())
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_mcp(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Cursor. Restart Cursor to activate.")


def install_windsurf(db_path):
    """Auto-configure Windsurf MCP settings."""
    _ensure_mcp_package()
    config_path = _resolve_path(_windsurf_candidates())
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_mcp(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Windsurf. Restart Windsurf to activate.")


def install_continue(db_path):
    """Auto-configure VS Code Continue extension MCP settings."""
    config_path = _resolve_path(_continue_candidates())
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_continue(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Continue extension. Restart your editor to activate.")


def install_openclaw(db_path):
    """Drop a ready skill folder."""
    _ = db_path
    _install_openclaw_into(_openclaw_workspace_root(), "Synapse skill installed for OpenClaw.")


def install_nanoclaw(db_path):
    """Similar to OpenClaw but for NanoClaw layout."""
    _ = db_path
    _install_openclaw_into(_nanoclaw_workspace_root(), "Synapse skill installed for NanoClaw.")


def _telegram_candidates() -> list[str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(repo_root, "..", "synapse-bot"),
        os.path.expanduser("~/synapse-bot"),
        os.path.expanduser("~/workspace/synapse-bot"),
        os.path.expanduser("~/Projects/synapse-bot"),
        os.path.expanduser("~/code/synapse-bot"),
    ]


def _detect_telegram_dir() -> str | None:
    for path in _telegram_candidates():
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "setup.py")):
            return path
    return None


def _telegram_config_path() -> str:
    return os.path.expanduser("~/.synapse/telegram-bot-installed.flag")


def _read_input(prompt: str) -> str:
    while True:
        try:
            value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            raise SystemExit(1)
        if value:
            return value
        print("  A value is required.")


def _read_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        response = input(f"{prompt}{suffix}: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("  Please answer y or n.")


def _ollama_models() -> list[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        bits = line.split()
        if bits:
            models.append(bits[0])
    return models


def _telegram_bot_py() -> str:
    import base64
    return base64.b64decode("ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGxvZ2dpbmcKaW1wb3J0IG9zCmltcG9ydCB0ZW1wZmlsZQpmcm9tIGNvbGxlY3Rpb25zIGltcG9ydCBkZWZhdWx0ZGljdCwgZGVxdWUKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgRGVxdWUsIERpY3QKCmZyb20gdGVsZWdyYW0gaW1wb3J0IElubGluZUtleWJvYXJkQnV0dG9uLCBJbmxpbmVLZXlib2FyZE1hcmt1cCwgVXBkYXRlCmZyb20gdGVsZWdyYW0uY29uc3RhbnRzIGltcG9ydCBDaGF0QWN0aW9uCmZyb20gdGVsZWdyYW0uZXh0IGltcG9ydCAoCiAgICBBcHBsaWNhdGlvbkJ1aWxkZXIsCiAgICBDYWxsYmFja0NvbnRleHQsCiAgICBDYWxsYmFja1F1ZXJ5SGFuZGxlciwKICAgIENvbW1hbmRIYW5kbGVyLAogICAgQ29udGV4dFR5cGVzLAogICAgTWVzc2FnZUhhbmRsZXIsCiAgICBmaWx0ZXJzLAopCgpmcm9tIGNvbmZpZyBpbXBvcnQgQ29uZmlnLCBsb2FkCmZyb20gbGxtX2JhY2tlbmQgaW1wb3J0IExMTVVuYXZhaWxhYmxlLCBidWlsZF9iYWNrZW5kCmZyb20gbWVtb3J5X21hbmFnZXIgaW1wb3J0IE1lbW9yeU1hbmFnZXIKCgpsb2dnZXIgPSBsb2dnaW5nLmdldExvZ2dlcihfX25hbWVfXykKCgpAZGF0YWNsYXNzCmNsYXNzIFJhdGVMaW1pdGVyOgogICAgbGltaXRfcGVyX21pbnV0ZTogaW50CiAgICBidWNrZXRzOiBEaWN0W3N0ciwgRGVxdWVbZmxvYXRdXSA9IE5vbmUKCiAgICBkZWYgX19wb3N0X2luaXRfXyhzZWxmKToKICAgICAgICBpZiBzZWxmLmJ1Y2tldHMgaXMgTm9uZToKICAgICAgICAgICAgc2VsZi5idWNrZXRzID0gZGVmYXVsdGRpY3QoZGVxdWUpCgogICAgZGVmIGFsbG93KHNlbGYsIGtleTogc3RyKSAtPiBib29sOgogICAgICAgIGltcG9ydCB0aW1lCgogICAgICAgIGJ1Y2tldCA9IHNlbGYuYnVja2V0c1trZXldCiAgICAgICAgbm93ID0gdGltZS50aW1lKCkKICAgICAgICB3aGlsZSBidWNrZXQgYW5kIG5vdyAtIGJ1Y2tldFswXSA+IDYwOgogICAgICAgICAgICBidWNrZXQucG9wbGVmdCgpCiAgICAgICAgaWYgbGVuKGJ1Y2tldCkgPj0gc2VsZi5saW1pdF9wZXJfbWludXRlOgogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICBidWNrZXQuYXBwZW5kKG5vdykKICAgICAgICByZXR1cm4gVHJ1ZQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQojIFV0aWxpdHkgaGVscGVycwojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KCgpkZWYgX2J1aWxkX21lbW9yeV9rZXlib2FyZChpdGVtX2lkOiBzdHIsIHVzZXJfaWQ6IHN0ciB8IGludCkgLT4gSW5saW5lS2V5Ym9hcmRNYXJrdXA6CiAgICByZXR1cm4gSW5saW5lS2V5Ym9hcmRNYXJrdXAoCiAgICAgICAgWwogICAgICAgICAgICBbCiAgICAgICAgICAgICAgICBJbmxpbmVLZXlib2FyZEJ1dHRvbigi4pyFIEFwcHJvdmUiLCBjYWxsYmFja19kYXRhPWYiaW5ib3g6YXBwcm92ZTp7dXNlcl9pZH06e2l0ZW1faWR9IiksCiAgICAgICAgICAgICAgICBJbmxpbmVLZXlib2FyZEJ1dHRvbigi4p2MIFJlamVjdCIsIGNhbGxiYWNrX2RhdGE9ZiJpbmJveDpyZWplY3Q6e3VzZXJfaWR9OntpdGVtX2lkfSIpLAogICAgICAgICAgICAgICAgSW5saW5lS2V5Ym9hcmRCdXR0b24oIvCflJIgUmVkYWN0IiwgY2FsbGJhY2tfZGF0YT1mImluYm94OnJlZGFjdDp7dXNlcl9pZH06e2l0ZW1faWR9IiksCiAgICAgICAgICAgIF0KICAgICAgICBdCiAgICApCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCiMgQm90CiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQpjbGFzcyBTeW5hcHNlVGVsZWdyYW1Cb3Q6CiAgICBkZWYgX19pbml0X18oc2VsZiwgY2ZnOiBDb25maWcpOgogICAgICAgIHNlbGYuY2ZnID0gY2ZnCiAgICAgICAgc2VsZi5tZW1vcnkgPSBNZW1vcnlNYW5hZ2VyKGNmZy5kYXRhX2RpciwgY2ZnLm1heF9jb250ZXh0X21lbW9yaWVzLCBjZmcuaW5ib3hfbW9kZV9lbmFibGVkKQogICAgICAgIHNlbGYubGxtID0gYnVpbGRfYmFja2VuZChjZmcpCiAgICAgICAgc2VsZi5yYXRlID0gUmF0ZUxpbWl0ZXIoY2ZnLnJhdGVfbGltaXRfcGVyX21pbnV0ZSkKCiAgICBkZWYgc3RhcnQoc2VsZikgLT4gTm9uZToKICAgICAgICBhcHAgPSAoCiAgICAgICAgICAgIEFwcGxpY2F0aW9uQnVpbGRlcigpCiAgICAgICAgICAgIC50b2tlbihzZWxmLmNmZy50ZWxlZ3JhbV90b2tlbikKICAgICAgICAgICAgLmNvbmN1cnJlbnRfdXBkYXRlcyhUcnVlKQogICAgICAgICAgICAuYnVpbGQoKQogICAgICAgICkKCiAgICAgICAgYXBwLmFkZF9oYW5kbGVyKENvbW1hbmRIYW5kbGVyKCJzdGFydCIsIHNlbGYuY21kX3N0YXJ0KSkKICAgICAgICBhcHAuYWRkX2hhbmRsZXIoQ29tbWFuZEhhbmRsZXIoImhlbHAiLCBzZWxmLmNtZF9oZWxwKSkKICAgICAgICBhcHAuYWRkX2hhbmRsZXIoQ29tbWFuZEhhbmRsZXIoInJlbWVtYmVyIiwgc2VsZi5jbWRfcmVtZW1iZXIpKQogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDb21tYW5kSGFuZGxlcigicmVjYWxsIiwgc2VsZi5jbWRfcmVjYWxsKSkKICAgICAgICBhcHAuYWRkX2hhbmRsZXIoQ29tbWFuZEhhbmRsZXIoImZvcmdldCIsIHNlbGYuY21kX2ZvcmdldCkpCiAgICAgICAgYXBwLmFkZF9oYW5kbGVyKENvbW1hbmRIYW5kbGVyKCJtZW1vcmllcyIsIHNlbGYuY21kX21lbW9yaWVzKSkKICAgICAgICBhcHAuYWRkX2hhbmRsZXIoQ29tbWFuZEhhbmRsZXIoImluYm94Iiwgc2VsZi5jbWRfaW5ib3gpKQogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDb21tYW5kSGFuZGxlcigicHJpdmFjeSIsIHNlbGYuY21kX3ByaXZhY3kpKQogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDb21tYW5kSGFuZGxlcigiZXhwb3J0Iiwgc2VsZi5jbWRfZXhwb3J0KSkKICAgICAgICBhcHAuYWRkX2hhbmRsZXIoQ29tbWFuZEhhbmRsZXIoImNsZWFyIiwgc2VsZi5jbWRfY2xlYXIpKQogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDb21tYW5kSGFuZGxlcigibXlzdGF0cyIsIHNlbGYuY21kX215c3RhdHMpKQogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDb21tYW5kSGFuZGxlcigic3VycHJpc2UiLCBzZWxmLmNtZF9zdXJwcmlzZSkpCgogICAgICAgIGFwcC5hZGRfaGFuZGxlcihDYWxsYmFja1F1ZXJ5SGFuZGxlcihzZWxmLm9uX2luYm94X2FjdGlvbiwgcGF0dGVybj1yIl5pbmJveDoiKSkKCiAgICAgICAgYXBwLmFkZF9oYW5kbGVyKE1lc3NhZ2VIYW5kbGVyKGZpbHRlcnMuVEVYVCAmIH5maWx0ZXJzLkNPTU1BTkQsIHNlbGYub25fbWVzc2FnZSkpCgogICAgICAgIGFwcC5qb2JfcXVldWUucnVuX3JlcGVhdGluZyhzZWxmLl9zY2hlZHVsZWRfY29uc29saWRhdGlvbiwgaW50ZXJ2YWw9c2VsZi5jZmcuY29uc29saWRhdGlvbl9pbnRlcnZhbF9ob3VycyAqIDM2MDAsIGZpcnN0PTYwKQoKICAgICAgICBhcHAucnVuX3BvbGxpbmcoYWxsb3dlZF91cGRhdGVzPVVwZGF0ZS5BTExfVFlQRVMpCgogICAgYXN5bmMgZGVmIF9zY2hlZHVsZWRfY29uc29saWRhdGlvbihzZWxmLCBjb250ZXh0OiBDYWxsYmFja0NvbnRleHQpOgogICAgICAgIGF3YWl0IHNlbGYucnVuX3BlcmlvZGljX2NvbnNvbGlkYXRpb24oKQoKICAgIGFzeW5jIGRlZiBydW5fcGVyaW9kaWNfY29uc29saWRhdGlvbihzZWxmKToKICAgICAgICBmb3IgdXNlcl9pZCBpbiBsaXN0KHNlbGYubWVtb3J5LmFsbF91c2VyX2lkcygpKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgYXdhaXQgc2VsZi5tZW1vcnkucnVuX2NvbnNvbGlkYXRpb24odXNlcl9pZCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oImNvbnNvbGlkYXRpb24gZmFpbGVkIGZvciAlcyIsIHVzZXJfaWQpCgogICAgYXN5bmMgZGVmIGNtZF9zdGFydChzZWxmLCB1cGRhdGU6IFVwZGF0ZSwgY29udGV4dDogQ29udGV4dFR5cGVzLkRFRkFVTFRfVFlQRSkgLT4gTm9uZToKICAgICAgICBhd2FpdCBzZWxmLl9zYWZlX3NlbmRfdGV4dCgKICAgICAgICAgICAgdXBkYXRlLAogICAgICAgICAgICAiV2VsY29tZSB0byAqU3luYXBzZSBNZW1vcnkgQm90KiFcbiIKICAgICAgICAgICAgIkkgcmVtZW1iZXIgZWFjaCB1c2VyIHNlcGFyYXRlbHkgYW5kIHVzZSB0aGF0IG1lbW9yeSB0byBwZXJzb25hbGl6ZSByZXBsaWVzLlxuXG4iCiAgICAgICAgICAgICJUcnkgL2hlbHAgdG8gc2VlIGNvbW1hbmRzLiIsCiAgICAgICAgKQoKICAgIGFzeW5jIGRlZiBjbWRfaGVscChzZWxmLCB1cGRhdGU6IFVwZGF0ZSwgY29udGV4dDogQ29udGV4dFR5cGVzLkRFRkFVTFRfVFlQRSkgLT4gTm9uZToKICAgICAgICB0ZXh0ID0gKAogICAgICAgICAgICAiKlN5bmFwc2UgTWVtb3J5IEJvdCBjb21tYW5kcypcblxuIgogICAgICAgICAgICAi4oCiIC9yZW1lbWJlciBbZmFjdF0gLSBTYXZlIGEgZmFjdCBleHBsaWNpdGx5XG4iCiAgICAgICAgICAgICLigKIgL3JlY2FsbCBbcXVlcnldIC0gU2VhcmNoIHlvdXIgbWVtb3JpZXNcbiIKICAgICAgICAgICAgIuKAoiAvZm9yZ2V0IFt0ZXh0XSAtIFJlbW92ZSBtYXRjaGluZyBtZW1vcmllc1xuIgogICAgICAgICAgICAi4oCiIC9tZW1vcmllcyAtIE1lbW9yeSBzdGF0c1xuIgogICAgICAgICAgICAi4oCiIC9pbmJveCAtIFNob3cgcGVuZGluZyBpbmJveCBpdGVtcyAoc2Vuc2l0aXZlIGFwcHJvdmFscylcbiIKICAgICAgICAgICAgIuKAoiAvcHJpdmFjeSBbcHJpdmF0ZXxtaW5pbWFsfGVwaGVtZXJhbF0gLSBBcHBseSBwcml2YWN5IHByZXNldFxuIgogICAgICAgICAgICAi4oCiIC9leHBvcnQgLSBFeHBvcnQgeW91ciBtZW1vcmllc1xuIgogICAgICAgICAgICAi4oCiIC9jbGVhciAtIERlbGV0ZSBhbGwgbWVtb3JpZXNcbiIKICAgICAgICAgICAgIuKAoiAvbXlzdGF0cyAtIFZpcmFsIGRlbW8gc3RhdCBzdW1tYXJ5XG4iCiAgICAgICAgICAgICLigKIgL3N1cnByaXNlIC0gUHVsbCBhIHJhbmRvbSBtZW1vcnlcblxuIgogICAgICAgICAgICAiSSBhbHNvIHJlbWVtYmVyIGNoYXQgY29udGV4dCBhdXRvbWF0aWNhbGx5LiIKICAgICAgICApCiAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCB0ZXh0KQoKICAgIGFzeW5jIGRlZiBjbWRfcmVtZW1iZXIoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICB0ZXh0ID0gIiAiLmpvaW4oY29udGV4dC5hcmdzIG9yIFtdKQogICAgICAgIGlmIG5vdCB0ZXh0OgogICAgICAgICAgICBhd2FpdCBzZWxmLl9zYWZlX3NlbmRfdGV4dCh1cGRhdGUsICJVc2FnZTogL3JlbWVtYmVyIEkgbGlrZSBoaWtpbmcgb24gd2Vla2VuZHMiKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgaWYgc2VsZi5tZW1vcnkuZGV0ZWN0X3BpaSh0ZXh0KToKICAgICAgICAgICAgaXRlbV9pZCA9IGF3YWl0IHNlbGYubWVtb3J5LnJlcXVlc3RfaW5ib3hfaXRlbSh1aWQsIHRleHQpCiAgICAgICAgICAgIGlmIG5vdCBpdGVtX2lkOgogICAgICAgICAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCAiSSBjb3VsZG4ndCBxdWV1ZSB0aGF0IGZvciBpbmJveCBhcHByb3ZhbCByaWdodCBub3cuIEknbGwgc2F2ZSBpdCBub3JtYWxseSBmb3Igbm93LiIpCiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgYXdhaXQgdXBkYXRlLm1lc3NhZ2UucmVwbHlfdGV4dCgKICAgICAgICAgICAgICAgICLimqDvuI8gVGhpcyBsb29rcyBzZW5zaXRpdmUuIFNhdmUgaXQ/IiwKICAgICAgICAgICAgICAgIHJlcGx5X21hcmt1cD1fYnVpbGRfbWVtb3J5X2tleWJvYXJkKGl0ZW1faWQsIHVpZCksCiAgICAgICAgICAgICkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIHJlcyA9IGF3YWl0IHNlbGYubWVtb3J5LmV4cGxpY2l0X3JlbWVtYmVyKHVpZCwgdGV4dCkKICAgICAgICBhd2FpdCBzZWxmLl9zYWZlX3NlbmRfdGV4dCh1cGRhdGUsIHJlcykKCiAgICBhc3luYyBkZWYgY21kX3JlY2FsbChzZWxmLCB1cGRhdGU6IFVwZGF0ZSwgY29udGV4dDogQ29udGV4dFR5cGVzLkRFRkFVTFRfVFlQRSkgLT4gTm9uZToKICAgICAgICB1aWQgPSBzdHIodXBkYXRlLmVmZmVjdGl2ZV91c2VyLmlkKQogICAgICAgIHF1ZXJ5ID0gIiAiLmpvaW4oY29udGV4dC5hcmdzIG9yIFtdKQogICAgICAgIGlmIG5vdCBxdWVyeToKICAgICAgICAgICAgcXVlcnkgPSAiIgogICAgICAgIGhpdHMgPSBhd2FpdCBzZWxmLm1lbW9yeS5yZWNhbGwodWlkLCBxdWVyeSwgbGltaXQ9OCkKICAgICAgICBpZiBub3QgaGl0czoKICAgICAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCAiTm8gbWF0Y2hpbmcgbWVtb3JpZXMgeWV0LiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGxpbmVzID0gW2Yie2krMX0uIHttLnRleHR9IiBmb3IgaSwgbSBpbiBlbnVtZXJhdGUoaGl0cyldCiAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCAiKlJlY2FsbCByZXN1bHRzKlxuIiArICJcbiIuam9pbihsaW5lcykpCgogICAgYXN5bmMgZGVmIGNtZF9mb3JnZXQoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICBxdWVyeSA9ICIgIi5qb2luKGNvbnRleHQuYXJncyBvciBbXSkKICAgICAgICBpZiBub3QgcXVlcnk6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIlVzYWdlOiAvZm9yZ2V0IFt0ZXh0IHNuaXBwZXRdIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgcmVtb3ZlZCA9IGF3YWl0IHNlbGYubWVtb3J5LmZvcmdldCh1aWQsIHF1ZXJ5KQogICAgICAgIGlmIHJlbW92ZWQ6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgZiJSZW1vdmVkIHtyZW1vdmVkfSBtZW1vcnkgaXRlbShzKS4iKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIk5vIG1hdGNoaW5nIG1lbW9yaWVzIHdlcmUgZm91bmQgdG8gZm9yZ2V0LiIpCgogICAgYXN5bmMgZGVmIGNtZF9tZW1vcmllcyhzZWxmLCB1cGRhdGU6IFVwZGF0ZSwgY29udGV4dDogQ29udGV4dFR5cGVzLkRFRkFVTFRfVFlQRSkgLT4gTm9uZToKICAgICAgICB1aWQgPSBzdHIodXBkYXRlLmVmZmVjdGl2ZV91c2VyLmlkKQogICAgICAgIHN0YXRzID0gYXdhaXQgc2VsZi5tZW1vcnkubWVtb3J5X3N0YXRzKHVpZCkKICAgICAgICBsaW5lcyA9IFsKICAgICAgICAgICAgZiIqTWVtb3J5IHN0YXRzKiIsCiAgICAgICAgICAgIGYi4oCiIFRvdGFsIG1lbW9yaWVzOiB7c3RhdHMuZ2V0KCdjb3VudCcsIDApfSIsCiAgICAgICAgICAgIGYi4oCiIENvbmNlcHRzOiB7c3RhdHMuZ2V0KCdjb25jZXB0cycsIDApfSIsCiAgICAgICAgXQogICAgICAgIGhvdCA9IHN0YXRzLmdldCgiaG90Iikgb3IgW10KICAgICAgICBpZiBob3Q6CiAgICAgICAgICAgIGxpbmVzLmFwcGVuZCgi4oCiIEhvdCB0b3BpY3M6ICIgKyAiLCAiLmpvaW4oc3RyKGgpIGZvciBoIGluIGhvdFs6NV0pKQogICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIlxuIi5qb2luKGxpbmVzKSkKCiAgICBhc3luYyBkZWYgY21kX2luYm94KHNlbGYsIHVwZGF0ZTogVXBkYXRlLCBjb250ZXh0OiBDb250ZXh0VHlwZXMuREVGQVVMVF9UWVBFKSAtPiBOb25lOgogICAgICAgIHVpZCA9IHN0cih1cGRhdGUuZWZmZWN0aXZlX3VzZXIuaWQpCiAgICAgICAgaXRlbXMgPSBhd2FpdCBzZWxmLm1lbW9yeS5wZW5kaW5nX2l0ZW1zKHVpZCkKICAgICAgICBpZiBub3QgaXRlbXM6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIk5vIHBlbmRpbmcgaW5ib3ggaXRlbXMuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgbGluZXMgPSBbIipQZW5kaW5nIGluYm94IGl0ZW1zKlxuIl0KICAgICAgICBmb3IgaXQgaW4gaXRlbXNbLTEwOl06CiAgICAgICAgICAgIGNpZCA9IGl0LmdldCgiaWQiLCAiPyIpCiAgICAgICAgICAgIGNvbnRlbnQgPSBpdC5nZXQoImNvbnRlbnQiLCAiIikKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYi4oCiIGB7Y2lkfWAgLSB7Y29udGVudFs6MTYwXX0iKQogICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIlxuIi5qb2luKGxpbmVzKSkKCiAgICBhc3luYyBkZWYgY21kX3ByaXZhY3koc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICBwcmVzZXQgPSAoY29udGV4dC5hcmdzWzBdIGlmIGNvbnRleHQuYXJncyBlbHNlICIiKS5zdHJpcCgpLmxvd2VyKCkKICAgICAgICBpZiBub3QgcHJlc2V0OgogICAgICAgICAgICBjdXJyZW50ID0gYXdhaXQgc2VsZi5tZW1vcnkucHJpdmFjeV9zdGF0dXModWlkKQogICAgICAgICAgICBhd2FpdCBzZWxmLl9zYWZlX3NlbmRfdGV4dCh1cGRhdGUsIGYiQ3VycmVudCBwcml2YWN5IHByZXNldDogYHtjdXJyZW50fWAiKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBvayA9IGF3YWl0IHNlbGYubWVtb3J5LmFwcGx5X3ByaXZhY3kodWlkLCBwcmVzZXQpCiAgICAgICAgaWYgb2s6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgZiJQcml2YWN5IHByZXNldCB1cGRhdGVkIHRvIGB7cHJlc2V0fWAuIikKICAgICAgICBlbHNlOgogICAgICAgICAgICBhd2FpdCBzZWxmLl9zYWZlX3NlbmRfdGV4dCh1cGRhdGUsICJJbnZhbGlkIHByZXNldC4gVXNlOiBwcml2YXRlIHwgbWluaW1hbCB8IGVwaGVtZXJhbCIpCgogICAgYXN5bmMgZGVmIGNtZF9leHBvcnQoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICB3aXRoIHRlbXBmaWxlLk5hbWVkVGVtcG9yYXJ5RmlsZShzdWZmaXg9Ii5zeW5hcHNlIiwgZGVsZXRlPUZhbHNlKSBhcyBmcDoKICAgICAgICAgICAgb3V0X3BhdGggPSBQYXRoKGZwLm5hbWUpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBvayA9IGF3YWl0IHNlbGYubWVtb3J5LmV4cG9ydF9idW5kbGUodWlkLCBvdXRfcGF0aCkKICAgICAgICAgICAgaWYgbm90IG9rOgogICAgICAgICAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCAiRXhwb3J0IGZhaWxlZC4gVHJ5IGFnYWluIGxhdGVyLiIpCiAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgYXdhaXQgY29udGV4dC5ib3Quc2VuZF9kb2N1bWVudChjaGF0X2lkPXVwZGF0ZS5lZmZlY3RpdmVfY2hhdC5pZCwgZG9jdW1lbnQ9b3V0X3BhdGgub3BlbigicmIiKSwgZmlsZW5hbWU9ZiJzeW5hcHNlX2V4cG9ydF97dWlkfS5zeW5hcHNlIikKICAgICAgICBmaW5hbGx5OgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBvdXRfcGF0aC51bmxpbmsobWlzc2luZ19vaz1UcnVlKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwoKICAgIGFzeW5jIGRlZiBjbWRfY2xlYXIoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICBjb3VudCA9IGF3YWl0IHNlbGYubWVtb3J5LmZvcmdldF9hbGwodWlkKQogICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgZiJDbGVhcmVkIHtjb3VudH0gbWVtb3JpZXMgZnJvbSB5b3VyIHZhdWx0LiIpCgogICAgYXN5bmMgZGVmIGNtZF9teXN0YXRzKHNlbGYsIHVwZGF0ZTogVXBkYXRlLCBjb250ZXh0OiBDb250ZXh0VHlwZXMuREVGQVVMVF9UWVBFKSAtPiBOb25lOgogICAgICAgIHVpZCA9IHN0cih1cGRhdGUuZWZmZWN0aXZlX3VzZXIuaWQpCiAgICAgICAgc3RhdHMgPSBhd2FpdCBzZWxmLm1lbW9yeS5tZW1vcnlfc3RhdHModWlkKQogICAgICAgIGNvdW50ID0gc3RhdHMuZ2V0KCJjb3VudCIsIDApCiAgICAgICAgdG9waWNzID0gc3RhdHMuZ2V0KCJjb25jZXB0cyIsIDApCiAgICAgICAgc2hhcmUgPSAiSSBoYXZlIGFuIEFJIHRoYXQgcmVtZW1iZXJzIGV2ZXJ5dGhpbmcuIFRyeSBpdDogQFN5bmFwc2VNZW1vcnlCb3QiCiAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCBmIllvdXIgQUkga25vd3MgKntjb3VudH0qIHRoaW5ncyBhY3Jvc3MgKnt0b3BpY3N9KiB0b3BpY3MuXG5cbntzaGFyZX0iKQoKICAgIGFzeW5jIGRlZiBjbWRfc3VycHJpc2Uoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgdWlkID0gc3RyKHVwZGF0ZS5lZmZlY3RpdmVfdXNlci5pZCkKICAgICAgICB0ZXh0ID0gYXdhaXQgc2VsZi5tZW1vcnkuc3VycHJpc2UodWlkKQogICAgICAgIGlmIHRleHQ6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgZiJSZW1lbWJlciB3aGVuIHlvdSB0b2xkIG1lOiB7dGV4dH0iKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIk5vIG1lbW9yaWVzIHlldCB0byBzdXJwcmlzZSB5b3Ugd2l0aC4iKQoKICAgIGFzeW5jIGRlZiBvbl9pbmJveF9hY3Rpb24oc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgcXVlcnkgPSB1cGRhdGUuY2FsbGJhY2tfcXVlcnkKICAgICAgICBpZiBub3QgcXVlcnkgb3Igbm90IHF1ZXJ5LmRhdGE6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIGF3YWl0IHF1ZXJ5LmFuc3dlcigpCiAgICAgICAgcGFydHMgPSBxdWVyeS5kYXRhLnNwbGl0KCI6IikKICAgICAgICBpZiBsZW4ocGFydHMpICE9IDQ6CiAgICAgICAgICAgIGF3YWl0IHF1ZXJ5LmVkaXRfbWVzc2FnZV90ZXh0KCJJbnZhbGlkIGNhbGxiYWNrIHBheWxvYWQuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgXywgYWN0aW9uLCB1c2VyX3N0ciwgaXRlbV9pZCA9IHBhcnRzCiAgICAgICAgZXhwZWN0ZWRfdXNlciA9IHN0cihxdWVyeS5mcm9tX3VzZXIuaWQpCiAgICAgICAgaWYgZXhwZWN0ZWRfdXNlciAhPSB1c2VyX3N0cjoKICAgICAgICAgICAgYXdhaXQgcXVlcnkuZWRpdF9tZXNzYWdlX3RleHQoIk5vdCBhdXRob3JpemVkIGZvciB0aGlzIGFjdGlvbi4iKQogICAgICAgICAgICByZXR1cm4KCiAgICAgICAgaWYgYWN0aW9uID09ICJhcHByb3ZlIjoKICAgICAgICAgICAgb2sgPSBhd2FpdCBzZWxmLm1lbW9yeS5hcHByb3ZlX2l0ZW0odXNlcl9zdHIsIGl0ZW1faWQpCiAgICAgICAgICAgIG1zZyA9ICJTYXZlZCDinIUiIGlmIG9rIGVsc2UgIkNvdWxkIG5vdCBhcHByb3ZlIGl0ZW0uIgogICAgICAgIGVsaWYgYWN0aW9uID09ICJyZWplY3QiOgogICAgICAgICAgICBvayA9IGF3YWl0IHNlbGYubWVtb3J5LnJlamVjdF9pdGVtKHVzZXJfc3RyLCBpdGVtX2lkKQogICAgICAgICAgICBtc2cgPSAiRGlzY2FyZGVkIOKchSIgaWYgb2sgZWxzZSAiQ291bGQgbm90IHJlamVjdCBpdGVtLiIKICAgICAgICBlbHNlOgogICAgICAgICAgICBvayA9IGF3YWl0IHNlbGYubWVtb3J5LnJlZGFjdF9pdGVtKHVzZXJfc3RyLCBpdGVtX2lkKQogICAgICAgICAgICBtc2cgPSAiUmVkYWN0ZWQg4pyFIiBpZiBvayBlbHNlICJDb3VsZCBub3QgcmVkYWN0IGl0ZW0uIgoKICAgICAgICBhd2FpdCBxdWVyeS5lZGl0X21lc3NhZ2VfdGV4dChtc2cpCgogICAgYXN5bmMgZGVmIG9uX21lc3NhZ2Uoc2VsZiwgdXBkYXRlOiBVcGRhdGUsIGNvbnRleHQ6IENvbnRleHRUeXBlcy5ERUZBVUxUX1RZUEUpIC0+IE5vbmU6CiAgICAgICAgaWYgbm90IHVwZGF0ZS5lZmZlY3RpdmVfbWVzc2FnZSBvciBub3QgdXBkYXRlLmVmZmVjdGl2ZV9tZXNzYWdlLnRleHQ6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHVpZCA9IHN0cih1cGRhdGUuZWZmZWN0aXZlX3VzZXIuaWQpCiAgICAgICAgbXNnID0gdXBkYXRlLmVmZmVjdGl2ZV9tZXNzYWdlLnRleHQuc3RyaXAoKQoKICAgICAgICBpZiBub3Qgc2VsZi5yYXRlLmFsbG93KHVpZCk6CiAgICAgICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgIlNsb3cgZG93biBhIGJpdCDigJQgSSBjYW4gcHJvY2VzcyBhYm91dCAzMCBtZXNzYWdlcy9taW51dGUuIikKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlmIHNlbGYubWVtb3J5LmRldGVjdF9waWkobXNnKToKICAgICAgICAgICAgaXRlbV9pZCA9IGF3YWl0IHNlbGYubWVtb3J5LnJlcXVlc3RfaW5ib3hfaXRlbSh1aWQsIG1zZykKICAgICAgICAgICAgaWYgaXRlbV9pZDoKICAgICAgICAgICAgICAgIGF3YWl0IHVwZGF0ZS5lZmZlY3RpdmVfbWVzc2FnZS5yZXBseV90ZXh0KAogICAgICAgICAgICAgICAgICAgICLimqDvuI8gVGhpcyBsb29rcyBzZW5zaXRpdmUuIFNhdmUgaXQ/IiwKICAgICAgICAgICAgICAgICAgICByZXBseV9tYXJrdXA9X2J1aWxkX21lbW9yeV9rZXlib2FyZChpdGVtX2lkLCB1aWQpLAogICAgICAgICAgICAgICAgKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgYXdhaXQgc2VsZi5fc2FmZV9zZW5kX3RleHQodXBkYXRlLCAiSSBjb3VsZG4ndCBxdWV1ZSB0aGF0IGZvciBpbmJveCBhcHByb3ZhbCByaWdodCBub3cuIikKCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICBhd2FpdCB1cGRhdGUuZWZmZWN0aXZlX2NoYXQuc2VuZF9hY3Rpb24oYWN0aW9uPUNoYXRBY3Rpb24uVFlQSU5HKQoKICAgICAgICBtZW1vcmllcyA9IGF3YWl0IHNlbGYubWVtb3J5LnJlY2FsbCh1aWQsIG1zZywgbGltaXQ9c2VsZi5jZmcubWF4X2NvbnRleHRfbWVtb3JpZXMpCiAgICAgICAgY29udGV4dF9saW5lcyA9IFttLnRleHQgZm9yIG0gaW4gbWVtb3JpZXNdCiAgICAgICAgY29udHJhZGljdGlvbiA9IGF3YWl0IHNlbGYubWVtb3J5LmRldGVjdF9jb250cmFkaWN0aW9uKHVpZCwgbXNnKQogICAgICAgIGlmIGNvbnRyYWRpY3Rpb246CiAgICAgICAgICAgIGNvbnRleHRfbGluZXMuYXBwZW5kKGYiUG90ZW50aWFsIGNvbnRyYWRpY3Rpb24gbm90ZToge2NvbnRyYWRpY3Rpb259IikKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3BvbnNlID0gYXdhaXQgc2VsZi5sbG0uZ2VuZXJhdGUoCiAgICAgICAgICAgICAgICBzeXN0ZW1fcHJvbXB0PXNlbGYuY2ZnLnN5c3RlbV9wcm9tcHQsCiAgICAgICAgICAgICAgICBtZW1vcmllcz1jb250ZXh0X2xpbmVzLAogICAgICAgICAgICAgICAgdXNlcl9tZXNzYWdlPW1zZywKICAgICAgICAgICAgKQogICAgICAgIGV4Y2VwdCBMTE1VbmF2YWlsYWJsZToKICAgICAgICAgICAgcmVzcG9uc2UgPSAoCiAgICAgICAgICAgICAgICAiSSdtIHJ1bm5pbmcgd2l0aCBsaW1pdGVkIG1lbW9yeSByaWdodCBub3cgKExMTSB1bmF2YWlsYWJsZSkuICIKICAgICAgICAgICAgICAgICJUcnkgYWdhaW4gaW4gYSBtb21lbnQsIG9yIHNlbmQgYSBzaG9ydGVyIHF1ZXN0aW9uLiIKICAgICAgICAgICAgKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkxMTSBjYWxsIGZhaWxlZCIpCiAgICAgICAgICAgIHJlc3BvbnNlID0gIkkgaGFkIHRyb3VibGUgZ2VuZXJhdGluZyBhIHJlc3BvbnNlIHJpZ2h0IG5vdy4gUGxlYXNlIHRyeSBhZ2FpbiBsYXRlci4iCgogICAgICAgIGF3YWl0IHNlbGYuX3NhZmVfc2VuZF90ZXh0KHVwZGF0ZSwgcmVzcG9uc2UpCgogICAgICAgICMgc3RvcmUgZmlsdGVyZWQgY29udmVyc2F0aW9uIG1lbW9yeSBhZnRlciByZXNwb25kaW5nCiAgICAgICAgYXdhaXQgc2VsZi5tZW1vcnkucmVtZW1iZXJfY29udmVyc2F0aW9uKHVpZCwgbXNnLCByZXNwb25zZSkKCiAgICBhc3luYyBkZWYgX3NhZmVfc2VuZF90ZXh0KHNlbGYsIHVwZGF0ZTogVXBkYXRlLCB0ZXh0OiBzdHIsICoqa3dhcmdzKSAtPiBOb25lOgogICAgICAgIGlmIHVwZGF0ZS5lZmZlY3RpdmVfbWVzc2FnZSBpcyBOb25lOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBhd2FpdCB1cGRhdGUuZWZmZWN0aXZlX21lc3NhZ2UucmVwbHlfdGV4dCh0ZXh0LCAqKmt3YXJncykKCgpkZWYgbWFpbigpIC0+IE5vbmU6CiAgICBsb2dnaW5nLmJhc2ljQ29uZmlnKAogICAgICAgIGxldmVsPW9zLmdldGVudigiTE9HX0xFVkVMIiwgIklORk8iKS51cHBlcigpLAogICAgICAgIGZvcm1hdD0iJShhc2N0aW1lKXMgJShsZXZlbG5hbWUpcyAlKG5hbWUpczogJShtZXNzYWdlKXMiLAogICAgKQoKICAgIGNmZyA9IGxvYWQoKQogICAgYm90ID0gU3luYXBzZVRlbGVncmFtQm90KGNmZykKICAgIGJvdC5zdGFydCgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=").decode("utf-8")


def _telegram_memory_manager_py() -> str:
    import base64
    return base64.b64decode("ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGpzb24KaW1wb3J0IGxvZ2dpbmcKaW1wb3J0IHJhbmRvbQppbXBvcnQgcmUKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55LCBEaWN0LCBJdGVyYWJsZSwgTGlzdCwgT3B0aW9uYWwKCmZyb20gc3luYXBzZSBpbXBvcnQgU3luYXBzZQoKbG9nZ2VyID0gbG9nZ2luZy5nZXRMb2dnZXIoX19uYW1lX18pCgoKUElJX1BBVFRFUk5TID0gWwogICAgcmUuY29tcGlsZShyIlxiXGR7M30tXGR7Mn0tXGR7NH1cYiIpLCAgIyBzc24taXNoCiAgICByZS5jb21waWxlKHIiXGJcZHsxNn1cYiIpLCAgIyBsb25nIG51bWJlciAvIGNhcmQtbGlrZQogICAgcmUuY29tcGlsZShyIlxiXHcrKFstKy4nXVx3KykqQFx3K1wuXHcrXGIiLCByZS5JR05PUkVDQVNFKSwKICAgIHJlLmNvbXBpbGUociJcYig/OlwrP1xkezEsMn1bLS5cc10/KT8oPzpcKD9cZHszfVwpP1stLlxzXT8pP1xkezN9Wy0uXHNdP1xkezR9XGIiKSwKICAgIHJlLmNvbXBpbGUociJcYig/OnNrLVtBLVphLXowLTldezEwLH0pXGIiKSwKXQoKVFJJVklBTF9DSEFUID0gewogICAgImhpIiwgImhlbGxvIiwgImhleSIsICJ0aGFua3MiLCAidGhhbmsgeW91IiwgInRoeCIsICJvayIsICJva2F5IiwgInllcCIsICJ5dXAiLCAiYnllIiwgImdvb2RieWUiLAp9CgoKQGRhdGFjbGFzcwpjbGFzcyBNZW1vcnlDYW5kaWRhdGU6CiAgICBpZDogT3B0aW9uYWxbaW50XQogICAgdGV4dDogc3RyCiAgICBzY29yZTogZmxvYXQgPSAxLjAKCgpjbGFzcyBNZW1vcnlNYW5hZ2VyOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGRhdGFfZGlyOiBzdHIsIG1heF9jb250ZXh0X21lbW9yaWVzOiBpbnQgPSA1LCBpbmJveF9tb2RlOiBib29sID0gVHJ1ZSk6CiAgICAgICAgc2VsZi5fZGF0YV9kaXIgPSBQYXRoKGRhdGFfZGlyKQogICAgICAgIHNlbGYuX21heF9jb250ZXh0X21lbW9yaWVzID0gbWF4X2NvbnRleHRfbWVtb3JpZXMKICAgICAgICBzZWxmLl9pbmJveF9tb2RlID0gaW5ib3hfbW9kZQogICAgICAgIHNlbGYuX2luc3RhbmNlczogRGljdFtzdHIsIFN5bmFwc2VdID0ge30KICAgICAgICBzZWxmLl9tZXRhOiBEaWN0W3N0ciwgRGljdFtzdHIsIEFueV1dID0ge30KCiAgICBkZWYgX3ZhdWx0X2RpcihzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIpIC0+IFBhdGg6CiAgICAgICAgcmV0dXJuIHNlbGYuX2RhdGFfZGlyIC8gc3RyKHVzZXJfaWQpCgogICAgZGVmIF9sb2FkX21ldGEoc2VsZiwgdXNlcl9pZDogaW50IHwgc3RyKSAtPiBEaWN0W3N0ciwgQW55XToKICAgICAgICB1aWQgPSBzdHIodXNlcl9pZCkKICAgICAgICBpZiB1aWQgaW4gc2VsZi5fbWV0YToKICAgICAgICAgICAgcmV0dXJuIHNlbGYuX21ldGFbdWlkXQoKICAgICAgICB2ZGlyID0gc2VsZi5fdmF1bHRfZGlyKHVpZCkKICAgICAgICB2ZGlyLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICBtZXRhX3BhdGggPSB2ZGlyIC8gIm1ldGEuanNvbiIKICAgICAgICBpZiBtZXRhX3BhdGguZXhpc3RzKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuX21ldGFbdWlkXSA9IGpzb24ubG9hZHMobWV0YV9wYXRoLnJlYWRfdGV4dChlbmNvZGluZz0idXRmLTgiKSkKICAgICAgICAgICAgICAgIHJldHVybiBzZWxmLl9tZXRhW3VpZF0KICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkZhaWxlZCByZWFkaW5nIHVzZXIgbWV0YTsgcmVjcmVhdGluZyIpCgogICAgICAgIG1ldGEgPSB7CiAgICAgICAgICAgICJ1c2VyX2lkIjogdWlkLAogICAgICAgICAgICAidmF1bHQiOiB1aWQsCiAgICAgICAgICAgICJwcml2YWN5X3ByZXNldCI6ICJwcml2YXRlIiwKICAgICAgICAgICAgImNyZWF0ZWRfYXQiOiBfX2ltcG9ydF9fKCJ0aW1lIikudGltZSgpLAogICAgICAgIH0KICAgICAgICBtZXRhX3BhdGgud3JpdGVfdGV4dChqc29uLmR1bXBzKG1ldGEsIGluZGVudD0yKSwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICBzZWxmLl9tZXRhW3VpZF0gPSBtZXRhCiAgICAgICAgcmV0dXJuIG1ldGEKCiAgICBkZWYgZ2V0X21lbW9yeShzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIpIC0+IFN5bmFwc2U6CiAgICAgICAgdWlkID0gc3RyKHVzZXJfaWQpCiAgICAgICAgaWYgdWlkIGluIHNlbGYuX2luc3RhbmNlczoKICAgICAgICAgICAgcmV0dXJuIHNlbGYuX2luc3RhbmNlc1t1aWRdCgogICAgICAgIHZhdWx0X2RpciA9IHNlbGYuX3ZhdWx0X2Rpcih1aWQpCiAgICAgICAgdmF1bHRfZGlyLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICBkYl9wYXRoID0gdmF1bHRfZGlyIC8gInZhdWx0LmRiIgogICAgICAgIHN0b3JlID0gU3luYXBzZShwYXRoPXN0cihkYl9wYXRoKSwgdmF1bHQ9dWlkLCBpbmJveF9tb2RlPXNlbGYuX2luYm94X21vZGUpCgogICAgICAgICMgcmVzdG9yZSBwcmVmZXJyZWQgcHJpdmFjeSBwb2xpY3kKICAgICAgICBtZXRhID0gc2VsZi5fbG9hZF9tZXRhKHVpZCkKICAgICAgICBwcmVzZXQgPSBtZXRhLmdldCgicHJpdmFjeV9wcmVzZXQiLCAicHJpdmF0ZSIpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBpZiBwcmVzZXQgIT0gInByaXZhdGUiOgogICAgICAgICAgICAgICAgc3RvcmUucG9saWN5KHByZXNldCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKCJGYWlsZWQgdG8gYXBwbHkgc3RvcmVkIHByaXZhY3kgcHJlc2V0IGZvciAlcyIsIHVpZCkKCiAgICAgICAgc2VsZi5faW5zdGFuY2VzW3VpZF0gPSBzdG9yZQogICAgICAgIHJldHVybiBzdG9yZQoKICAgIGRlZiBfbm9ybWFsaXplKHNlbGYsIHRleHQ6IHN0cikgLT4gc3RyOgogICAgICAgIHJldHVybiAiICIuam9pbih0ZXh0LnN0cmlwKCkuc3BsaXQoKSkKCiAgICBkZWYgZGV0ZWN0X3BpaShzZWxmLCB0ZXh0OiBzdHIpIC0+IGJvb2w6CiAgICAgICAgcmV0dXJuIGFueShwLnNlYXJjaCh0ZXh0IG9yICIiKSBmb3IgcCBpbiBQSUlfUEFUVEVSTlMpCgogICAgZGVmIHNob3VsZF9zdG9yZShzZWxmLCB0ZXh0OiBzdHIpIC0+IGJvb2w6CiAgICAgICAgaWYgbm90IHRleHQ6CiAgICAgICAgICAgIHJldHVybiBGYWxzZQogICAgICAgIHQgPSB0ZXh0LnN0cmlwKCkubG93ZXIoKQogICAgICAgIGlmIHQgaW4gVFJJVklBTF9DSEFUOgogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICBpZiBsZW4odCkgPD0gNDoKICAgICAgICAgICAgcmV0dXJuIEZhbHNlCiAgICAgICAgcmV0dXJuIFRydWUKCiAgICBhc3luYyBkZWYgcmVjYWxsKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgcXVlcnk6IHN0ciwgbGltaXQ6IGludCB8IE5vbmUgPSBOb25lKSAtPiBMaXN0W01lbW9yeUNhbmRpZGF0ZV06CiAgICAgICAgbGltaXQgPSBsaW1pdCBvciBzZWxmLl9tYXhfY29udGV4dF9tZW1vcmllcwogICAgICAgIHN0b3JlID0gc2VsZi5nZXRfbWVtb3J5KHVzZXJfaWQpCiAgICAgICAgbWVtb3JpZXMgPSBzdG9yZS5yZWNhbGwocXVlcnksIGxpbWl0PWxpbWl0KQogICAgICAgIG91dDogTGlzdFtNZW1vcnlDYW5kaWRhdGVdID0gW10KICAgICAgICBmb3IgbSBpbiBtZW1vcmllczoKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgb3V0LmFwcGVuZChNZW1vcnlDYW5kaWRhdGUoaWQ9bS5pZCwgdGV4dD1zdHIobS5jb250ZW50KSwgc2NvcmU9Z2V0YXR0cihtLCAic3RyZW5ndGgiLCAxLjApKSkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkZhaWxlZCB0byBzZXJpYWxpemUgbWVtb3J5IikKICAgICAgICByZXR1cm4gb3V0CgogICAgYXN5bmMgZGVmIGRldGVjdF9jb250cmFkaWN0aW9uKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgbWVzc2FnZTogc3RyKSAtPiBPcHRpb25hbFtzdHJdOgogICAgICAgIG0gPSByZS5zZWFyY2gociJcYmlccysoPzpsaXZlfG1vdmVkfGZyb218YmFzZWQpXGIuKlxiKGlufGZyb20pXHMrKFtBLVphLXpdW0EtWmEtejAtOVxzLictXXsyLDYwfSkiLCBtZXNzYWdlLCByZS5JR05PUkVDQVNFKQogICAgICAgIGlmIG5vdCBtOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHN0YXRlZCA9IG0uZ3JvdXAoMikuc3RyaXAoKS5sb3dlcigpCiAgICAgICAgcGFzdCA9IGF3YWl0IHNlbGYucmVjYWxsKHVzZXJfaWQsICJJIGxpdmUiLCBsaW1pdD0yMCkKICAgICAgICBmb3IgaXRlbSBpbiBwYXN0OgogICAgICAgICAgICBvbGQgPSBpdGVtLnRleHQubG93ZXIoKQogICAgICAgICAgICBwcmlvciA9IHJlLnNlYXJjaChyIlxiaVxzKyg/OmxpdmV8bGl2ZWR8d2FzKVxzKyg/OmlufGF0fGZyb20pXHMrKFtBLVphLXpdW0EtWmEtejAtOVxzLictXXsyLDYwfSkiLCBvbGQpCiAgICAgICAgICAgIGlmIHByaW9yIGFuZCBwcmlvci5ncm91cCgxKS5zdHJpcCgpLmxvd2VyKCkgbm90IGluIHsiIiwgc3RhdGVkfSBhbmQgcHJpb3IuZ3JvdXAoMSkuc3RyaXAoKS5sb3dlcigpICE9IHN0YXRlZDoKICAgICAgICAgICAgICAgIHJldHVybiBmIlByZXZpb3VzbHkgSSBhbHNvIHJlbWVtYmVyIHlvdSBtZW50aW9uZWQge3ByaW9yLmdyb3VwKDEpLnN0cmlwKCkudGl0bGUoKX0uIgogICAgICAgIHJldHVybiBOb25lCgogICAgYXN5bmMgZGVmIHJlbWVtYmVyKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgdGV4dDogc3RyLCBzZW5zaXRpdmU6IGJvb2wgPSBGYWxzZSkgLT4gT3B0aW9uYWxbQW55XToKICAgICAgICB0ZXh0ID0gc2VsZi5fbm9ybWFsaXplKHRleHQpCiAgICAgICAgaWYgbm90IHNlbGYuc2hvdWxkX3N0b3JlKHRleHQpOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIHN0b3JlID0gc2VsZi5nZXRfbWVtb3J5KHVzZXJfaWQpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBtZW1vcnkgPSBzdG9yZS5yZW1lbWJlcih0ZXh0LCBzZW5zaXRpdmU9c2Vuc2l0aXZlKQogICAgICAgICAgICBpZiBtZW1vcnkgaXMgbm90IE5vbmUgYW5kIG5vdCBzZW5zaXRpdmUgYW5kIGdldGF0dHIobWVtb3J5LCAibWV0YWRhdGEiLCB7fSkuZ2V0KCJwZW5kaW5nIik6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgaXRlbV9pZCA9IG1lbW9yeS5tZXRhZGF0YS5nZXQoIml0ZW1faWQiKQogICAgICAgICAgICAgICAgICAgIGlmIGl0ZW1faWQ6CiAgICAgICAgICAgICAgICAgICAgICAgIHN0b3JlLmFwcHJvdmVfbWVtb3J5KGl0ZW1faWQpCiAgICAgICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkF1dG8tYXBwcm92ZSBmYWlsZWQgdXNlcj0lcyIsIHVzZXJfaWQpCiAgICAgICAgICAgIHJldHVybiBtZW1vcnkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKCJTeW5hcHNlIHJlbWVtYmVyIGZhaWxlZCBmb3IgdXNlcj0lcyIsIHVzZXJfaWQpCiAgICAgICAgICAgIHJldHVybiBOb25lCgogICAgYXN5bmMgZGVmIHJlbWVtYmVyX2NvbnZlcnNhdGlvbihzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIsIHVzZXJfbXNnOiBzdHIsIGJvdF9yZXBseTogc3RyKSAtPiBOb25lOgogICAgICAgIGNvbWJpbmVkID0gZiJVc2VyIHNhaWQ6IHt1c2VyX21zZ31cbkkgcmVwbGllZDoge2JvdF9yZXBseX0iCiAgICAgICAgYXdhaXQgc2VsZi5yZW1lbWJlcih1c2VyX2lkLCBjb21iaW5lZCwgc2Vuc2l0aXZlPUZhbHNlKQoKICAgIGFzeW5jIGRlZiBleHBsaWNpdF9yZW1lbWJlcihzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIsIGZhY3Q6IHN0cikgLT4gc3RyOgogICAgICAgIGZhY3QgPSBzZWxmLl9ub3JtYWxpemUoZmFjdCkKICAgICAgICBpZiBub3QgZmFjdDoKICAgICAgICAgICAgcmV0dXJuICJJIGRpZG4ndCBnZXQgYSBmYWN0IHRvIHJlbWVtYmVyLiIKICAgICAgICBpZiBzZWxmLmRldGVjdF9waWkoZmFjdCk6CiAgICAgICAgICAgIHJldHVybiAiVGhhdCBsb29rcyBzZW5zaXRpdmUuIEkgY2FuIHNhdmUgaXQgb25seSBhZnRlciB5b3UgY29uZmlybSB2aWEgL2luYm94IGFwcHJvdmFsIGZsb3cuIgogICAgICAgIG1lbSA9IGF3YWl0IHNlbGYucmVtZW1iZXIodXNlcl9pZCwgZmFjdCkKICAgICAgICByZXR1cm4gIlNhdmVkIHRvIG15IG1lbW9yeS4iIGlmIG1lbSBlbHNlICJJIGNvdWxkbid0IHNhdmUgdGhhdCByaWdodCBub3cuIgoKICAgIGFzeW5jIGRlZiBmb3JnZXQoc2VsZiwgdXNlcl9pZDogaW50IHwgc3RyLCBxdWVyeTogc3RyKSAtPiBpbnQ6CiAgICAgICAgc3RvcmUgPSBzZWxmLmdldF9tZW1vcnkodXNlcl9pZCkKICAgICAgICBxID0gcXVlcnkuc3RyaXAoKS5sb3dlcigpCiAgICAgICAgaWYgbm90IHE6CiAgICAgICAgICAgIHJldHVybiAwCiAgICAgICAgaGl0cyA9IGF3YWl0IHNlbGYucmVjYWxsKHVzZXJfaWQsIHEsIGxpbWl0PTIwKQogICAgICAgIHJlbW92ZWQgPSAwCiAgICAgICAgZm9yIG1lbSBpbiBoaXRzOgogICAgICAgICAgICBpZiBtZW0udGV4dCBhbmQgcSBpbiBtZW0udGV4dC5sb3dlcigpOgogICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgIGlmIG1lbS5pZCBpcyBub3QgTm9uZSBhbmQgc3RvcmUuZm9yZ2V0KG1lbS5pZCk6CiAgICAgICAgICAgICAgICAgICAgICAgIHJlbW92ZWQgKz0gMQogICAgICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKCJGb3JnZXQgZmFpbGVkIGZvciBpZCAlcyIsIG1lbS5pZCkKICAgICAgICByZXR1cm4gcmVtb3ZlZAoKICAgIGFzeW5jIGRlZiBmb3JnZXRfYWxsKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0cikgLT4gaW50OgogICAgICAgIHN0b3JlID0gc2VsZi5nZXRfbWVtb3J5KHVzZXJfaWQpCiAgICAgICAgY291bnQgPSAwCiAgICAgICAgYWxsX21lbW9yaWVzID0gc3RvcmUubGlzdChsaW1pdD0xMDAwKQogICAgICAgIGZvciBtZW0gaW4gYWxsX21lbW9yaWVzOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBpZiBtZW0uaWQgaXMgbm90IE5vbmUgYW5kIHN0b3JlLmZvcmdldChtZW0uaWQpOgogICAgICAgICAgICAgICAgICAgIGNvdW50ICs9IDEKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkZvcmdldCBmYWlsZWQgZm9yIGFsbCAlcyIsIG1lbS5pZCkKICAgICAgICByZXR1cm4gY291bnQKCiAgICBhc3luYyBkZWYgbWVtb3J5X3N0YXRzKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0cikgLT4gRGljdFtzdHIsIEFueV06CiAgICAgICAgc3RvcmUgPSBzZWxmLmdldF9tZW1vcnkodXNlcl9pZCkKICAgICAgICBjb25jZXB0cyA9IHN0b3JlLmNvbmNlcHRzKCkgb3IgW10KICAgICAgICBtZW1fY291bnQgPSBzdG9yZS5jb3VudCgpCiAgICAgICAgcmVjZW50ID0gYXdhaXQgc2VsZi5yZWNhbGwodXNlcl9pZCwgIiIsIGxpbWl0PTEpCiAgICAgICAgcmV0dXJuIHsKICAgICAgICAgICAgImNvdW50IjogbWVtX2NvdW50LAogICAgICAgICAgICAiY29uY2VwdHMiOiBsZW4oY29uY2VwdHMpLAogICAgICAgICAgICAiaG90IjogW2NbMF0gZm9yIGMgaW4gc3RvcmUuaG90X2NvbmNlcHRzKDUpXSBpZiBoYXNhdHRyKHN0b3JlLCAiaG90X2NvbmNlcHRzIikgZWxzZSBbXSwKICAgICAgICAgICAgImxhdGVzdCI6IHJlY2VudFswXS50ZXh0IGlmIHJlY2VudCBlbHNlIE5vbmUsCiAgICAgICAgfQogICAgYXN5bmMgZGVmIGxpc3RfbWVtb3JpZXMoc2VsZiwgdXNlcl9pZDogaW50IHwgc3RyLCBxdWVyeTogc3RyLCBsaW1pdDogaW50ID0gMTApIC0+IExpc3RbTWVtb3J5Q2FuZGlkYXRlXToKICAgICAgICByZXR1cm4gYXdhaXQgc2VsZi5yZWNhbGwodXNlcl9pZCwgcXVlcnksIGxpbWl0PWxpbWl0KQoKICAgIGFzeW5jIGRlZiBwZW5kaW5nX2l0ZW1zKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0cikgLT4gTGlzdFtEaWN0W3N0ciwgQW55XV06CiAgICAgICAgc3RvcmUgPSBzZWxmLmdldF9tZW1vcnkodXNlcl9pZCkKICAgICAgICB0cnk6CiAgICAgICAgICAgIGl0ZW1zID0gc3RvcmUubGlzdF9wZW5kaW5nKCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKCJsaXN0X3BlbmRpbmcgZmFpbGVkIGZvciB1c2VyPSVzIiwgdXNlcl9pZCkKICAgICAgICAgICAgcmV0dXJuIFtdCiAgICAgICAgcGVuZGluZyA9IFtdCiAgICAgICAgZm9yIGl0ZW0gaW4gaXRlbXM6CiAgICAgICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKGl0ZW0sIGRpY3QpOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgcGVuZGluZy5hcHBlbmQoaXRlbSkKICAgICAgICByZXR1cm4gcGVuZGluZwoKICAgIGFzeW5jIGRlZiByZXF1ZXN0X2luYm94X2l0ZW0oc2VsZiwgdXNlcl9pZDogaW50IHwgc3RyLCB0ZXh0OiBzdHIpIC0+IHN0cjoKICAgICAgICAiIiJDcmVhdGUgYW4gaW5ib3ggaXRlbSBhbmQgcmV0dXJuIGl0ZW1faWQgZm9yIGFwcHJvdmFsIGFjdGlvbnMuIiIiCiAgICAgICAgc3RvcmUgPSBzZWxmLmdldF9tZW1vcnkodXNlcl9pZCkKICAgICAgICB0cnk6CiAgICAgICAgICAgIGl0ZW0gPSBzdG9yZS5yZW1lbWJlcihzZWxmLl9ub3JtYWxpemUodGV4dCksIHNlbnNpdGl2ZT1UcnVlKQogICAgICAgICAgICBpZiBpdGVtIGFuZCBpc2luc3RhbmNlKGdldGF0dHIoaXRlbSwgIm1ldGFkYXRhIiwgTm9uZSksIGRpY3QpOgogICAgICAgICAgICAgICAgcmV0dXJuIGl0ZW0ubWV0YWRhdGEuZ2V0KCJpdGVtX2lkIiwgIiIpCiAgICAgICAgICAgIHJldHVybiAiIgogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oInJlcXVlc3RfaW5ib3hfaXRlbSBmYWlsZWQgZm9yIHVzZXI9JXMiLCB1c2VyX2lkKQogICAgICAgICAgICByZXR1cm4gIiIKCiAgICBhc3luYyBkZWYgYXBwcm92ZV9pdGVtKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgaXRlbV9pZDogc3RyKSAtPiBib29sOgogICAgICAgIHRyeToKICAgICAgICAgICAgcmV0dXJuIHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKS5hcHByb3ZlX21lbW9yeShpdGVtX2lkKSBpcyBub3QgTm9uZQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkFwcHJvdmUgZmFpbGVkIHVzZXI9JXMgaXRlbT0lcyIsIHVzZXJfaWQsIGl0ZW1faWQpCiAgICAgICAgICAgIHJldHVybiBGYWxzZQoKICAgIGFzeW5jIGRlZiByZWplY3RfaXRlbShzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIsIGl0ZW1faWQ6IHN0cikgLT4gYm9vbDoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJldHVybiBib29sKHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKS5yZWplY3RfbWVtb3J5KGl0ZW1faWQpKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIlJlamVjdCBmYWlsZWQgdXNlcj0lcyBpdGVtPSVzIiwgdXNlcl9pZCwgaXRlbV9pZCkKICAgICAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgYXN5bmMgZGVmIHJlZGFjdF9pdGVtKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgaXRlbV9pZDogc3RyKSAtPiBib29sOgogICAgICAgIHRyeToKICAgICAgICAgICAgcmV0dXJuIHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKS5yZWRhY3RfbWVtb3J5KGl0ZW1faWQsICJbcmVkYWN0ZWQgYnkgdXNlcl0iKSBpcyBub3QgTm9uZQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIlJlZGFjdCBmYWlsZWQgdXNlcj0lcyBpdGVtPSVzIiwgdXNlcl9pZCwgaXRlbV9pZCkKICAgICAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgYXN5bmMgZGVmIGFwcGx5X3ByaXZhY3koc2VsZiwgdXNlcl9pZDogaW50IHwgc3RyLCBwcmVzZXQ6IHN0cikgLT4gYm9vbDoKICAgICAgICBwcmVzZXQgPSAocHJlc2V0IG9yICIiKS5zdHJpcCgpLmxvd2VyKCkKICAgICAgICBpZiBub3QgcHJlc2V0OgogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICBpZiBwcmVzZXQgbm90IGluIHsicHJpdmF0ZSIsICJtaW5pbWFsIiwgImVwaGVtZXJhbCJ9OgogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICBzdG9yZSA9IHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKQogICAgICAgIHRyeToKICAgICAgICAgICAgc3RvcmUucG9saWN5KHByZXNldCkKICAgICAgICAgICAgbWV0YSA9IHNlbGYuX2xvYWRfbWV0YSh1c2VyX2lkKQogICAgICAgICAgICBtZXRhWyJwcml2YWN5X3ByZXNldCJdID0gcHJlc2V0CiAgICAgICAgICAgIHNlbGYuX3ZhdWx0X2Rpcih1c2VyX2lkKS5qb2lucGF0aCgibWV0YS5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKG1ldGEsIGluZGVudD0yKSwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICAgICAgc2VsZi5fbWV0YVtzdHIodXNlcl9pZCldID0gbWV0YQogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGxvZ2dlci5leGNlcHRpb24oIkNvdWxkIG5vdCBhcHBseSBwcml2YWN5IHByZXNldCIpCiAgICAgICAgICAgIHJldHVybiBGYWxzZQoKICAgIGFzeW5jIGRlZiBwcml2YWN5X3N0YXR1cyhzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIpIC0+IHN0cjoKICAgICAgICBzdG9yZSA9IHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKQogICAgICAgIHRyeToKICAgICAgICAgICAgYWN0aXZlID0gc3RvcmUuZ2V0X2FjdGl2ZV9wb2xpY3koKQogICAgICAgICAgICBpZiBhY3RpdmUgYW5kIGlzaW5zdGFuY2UoYWN0aXZlLCBkaWN0KToKICAgICAgICAgICAgICAgIHJldHVybiBhY3RpdmUuZ2V0KCJuYW1lIiwgInByaXZhdGUiKQogICAgICAgICAgICByZXR1cm4gInByaXZhdGUiCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcmV0dXJuICJwcml2YXRlIgoKICAgIGFzeW5jIGRlZiBleHBvcnRfYnVuZGxlKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0ciwgb3V0X3BhdGg6IFBhdGgpIC0+IGJvb2w6CiAgICAgICAgdHJ5OgogICAgICAgICAgICAjIFVzZSBTeW5hcHNlIGV4cG9ydCBmb3JtYXQgZm9yIGNvbXBhdGliaWxpdHksIHBsdXMgYSBsaWdodHdlaWdodCBKU09OIGZhbGxiYWNrLgogICAgICAgICAgICBzdG9yZSA9IHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKQogICAgICAgICAgICBzdG9yZS5leHBvcnQoc3RyKG91dF9wYXRoKSwgc291cmNlX2FnZW50PSJzeW5hcHNlLWJvdCIpCiAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgbG9nZ2VyLmV4Y2VwdGlvbigiU3luYXBzZSBleHBvcnQgZmFpbGVkOyB3cml0aW5nIG1hbnVhbCBmYWxsYmFjayIpCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIGRhdGEgPSBbCiAgICAgICAgICAgICAgICAgICAgewogICAgICAgICAgICAgICAgICAgICAgICAiaWQiOiBtLmlkLAogICAgICAgICAgICAgICAgICAgICAgICAiY29udGVudCI6IGdldGF0dHIobSwgImNvbnRlbnQiLCAiIiksCiAgICAgICAgICAgICAgICAgICAgICAgICJ0eXBlIjogZ2V0YXR0cihtLCAibWVtb3J5X3R5cGUiLCAiIiksCiAgICAgICAgICAgICAgICAgICAgICAgICJzY29wZSI6IGdldGF0dHIobSwgInNjb3BlIiwgIiIpLAogICAgICAgICAgICAgICAgICAgICAgICAic3RyZW5ndGgiOiBmbG9hdChnZXRhdHRyKG0sICJzdHJlbmd0aCIsIDAuMCkpLAogICAgICAgICAgICAgICAgICAgICAgICAiY3JlYXRlZF9hdCI6IGdldGF0dHIobSwgImNyZWF0ZWRfYXQiLCBOb25lKSwKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgZm9yIG0gaW4gc2VsZi5nZXRfbWVtb3J5KHVzZXJfaWQpLmxpc3QobGltaXQ9NTAwMCkKICAgICAgICAgICAgICAgIF0KICAgICAgICAgICAgICAgIHdpdGggb3V0X3BhdGgub3BlbigidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgICAgICAgICAganNvbi5kdW1wKGRhdGEsIGYsIGluZGVudD0yLCBkZWZhdWx0PXN0cikKICAgICAgICAgICAgICAgIHJldHVybiBUcnVlCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBsb2dnZXIuZXhjZXB0aW9uKCJGYWxsYmFjayBleHBvcnQgZmFpbGVkIikKICAgICAgICAgICAgICAgIHJldHVybiBGYWxzZQoKICAgIGFzeW5jIGRlZiBzdXJwcmlzZShzZWxmLCB1c2VyX2lkOiBpbnQgfCBzdHIpIC0+IE9wdGlvbmFsW3N0cl06CiAgICAgICAgbWVtcyA9IHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKS5saXN0KGxpbWl0PTIwMCkKICAgICAgICBpZiBub3QgbWVtczoKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICByZXR1cm4gcmFuZG9tLmNob2ljZShtZW1zKS5jb250ZW50CgogICAgYXN5bmMgZGVmIHJ1bl9jb25zb2xpZGF0aW9uKHNlbGYsIHVzZXJfaWQ6IGludCB8IHN0cikgLT4gTm9uZToKICAgICAgICB0cnk6CiAgICAgICAgICAgIHNlbGYuZ2V0X21lbW9yeSh1c2VyX2lkKS5jb25zb2xpZGF0ZSgpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgbG9nZ2VyLmV4Y2VwdGlvbigiQ29uc29saWRhdGUgZmFpbGVkIGZvciAlcyIsIHVzZXJfaWQpCgogICAgZGVmIGFsbF91c2VyX2lkcyhzZWxmKSAtPiBJdGVyYWJsZVtzdHJdOgogICAgICAgIGlmIG5vdCBzZWxmLl9kYXRhX2Rpci5leGlzdHMoKToKICAgICAgICAgICAgcmV0dXJuIFtdCiAgICAgICAgcmV0dXJuIFtwLm5hbWUgZm9yIHAgaW4gc2VsZi5fZGF0YV9kaXIuaXRlcmRpcigpIGlmIHAuaXNfZGlyKCldCg==").decode("utf-8")


def _telegram_llm_backend_py() -> str:
    import base64
    return base64.b64decode("CmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBsb2dnaW5nCmZyb20gYWJjIGltcG9ydCBBQkMsIGFic3RyYWN0bWV0aG9kCmZyb20gdHlwaW5nIGltcG9ydCBTZXF1ZW5jZQoKaW1wb3J0IGh0dHB4Cgpmcm9tIGNvbmZpZyBpbXBvcnQgQ29uZmlnCgoKbG9nZ2VyID0gbG9nZ2luZy5nZXRMb2dnZXIoX19uYW1lX18pCgoKY2xhc3MgTExNVW5hdmFpbGFibGUoUnVudGltZUVycm9yKToKICAgICIiIlJhaXNlZCB3aGVuIGFsbCBjb25maWd1cmVkIExMTSBwcm92aWRlcnMgZmFpbC4iIiIKCgpjbGFzcyBMTE1CYWNrZW5kKEFCQyk6CiAgICBAYWJzdHJhY3RtZXRob2QKICAgIGFzeW5jIGRlZiBnZW5lcmF0ZShzZWxmLCAqLCBzeXN0ZW1fcHJvbXB0OiBzdHIsIG1lbW9yaWVzOiBTZXF1ZW5jZVtzdHJdLCB1c2VyX21lc3NhZ2U6IHN0cikgLT4gc3RyOgogICAgICAgIC4uLgoKCmNsYXNzIE9sbGFtYUJhY2tlbmQoTExNQmFja2VuZCk6CiAgICBkZWYgX19pbml0X18oc2VsZiwgY2ZnOiBDb25maWcpOgogICAgICAgIHNlbGYuX2NmZyA9IGNmZwoKICAgIGFzeW5jIGRlZiBnZW5lcmF0ZShzZWxmLCAqLCBzeXN0ZW1fcHJvbXB0OiBzdHIsIG1lbW9yaWVzOiBTZXF1ZW5jZVtzdHJdLCB1c2VyX21lc3NhZ2U6IHN0cikgLT4gc3RyOgogICAgICAgIHBheWxvYWQgPSB7CiAgICAgICAgICAgICJtb2RlbCI6IHNlbGYuX2NmZy5vbGxhbWFfbW9kZWwsCiAgICAgICAgICAgICJtZXNzYWdlcyI6IFsKICAgICAgICAgICAgICAgIHsicm9sZSI6ICJzeXN0ZW0iLCAiY29udGVudCI6IHN5c3RlbV9wcm9tcHR9LAogICAgICAgICAgICAgICAgeyJyb2xlIjogInVzZXIiLCAiY29udGVudCI6IF9idWlsZF9wcm9tcHQoc3lzdGVtX3Byb21wdCwgbWVtb3JpZXMsIHVzZXJfbWVzc2FnZSl9LAogICAgICAgICAgICBdLAogICAgICAgICAgICAic3RyZWFtIjogRmFsc2UsCiAgICAgICAgICAgICJvcHRpb25zIjogeyJ0ZW1wZXJhdHVyZSI6IDAuM30sCiAgICAgICAgfQogICAgICAgIGFzeW5jIHdpdGggaHR0cHguQXN5bmNDbGllbnQodGltZW91dD1zZWxmLl9jZmcubGxtX3RpbWVvdXRfcykgYXMgY2xpZW50OgogICAgICAgICAgICByZXNwID0gYXdhaXQgY2xpZW50LnBvc3QoZiJ7c2VsZi5fY2ZnLm9sbGFtYV91cmwucnN0cmlwKCcvJyl9L2FwaS9jaGF0IiwganNvbj1wYXlsb2FkKQogICAgICAgICAgICByZXNwLnJhaXNlX2Zvcl9zdGF0dXMoKQogICAgICAgICAgICBkYXRhID0gcmVzcC5qc29uKCkKICAgICAgICAgICAgbXNnID0gZGF0YS5nZXQoIm1lc3NhZ2UiLCB7fSkuZ2V0KCJjb250ZW50IikKICAgICAgICAgICAgaWYgbm90IG1zZzoKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiT2xsYW1hIHJldHVybmVkIGVtcHR5IHJlc3BvbnNlIikKICAgICAgICAgICAgcmV0dXJuIHN0cihtc2cpLnN0cmlwKCkKCgpjbGFzcyBPcGVuQUlCYWNrZW5kKExMTUJhY2tlbmQpOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGNmZzogQ29uZmlnKToKICAgICAgICBzZWxmLl9jZmcgPSBjZmcKCiAgICBhc3luYyBkZWYgZ2VuZXJhdGUoc2VsZiwgKiwgc3lzdGVtX3Byb21wdDogc3RyLCBtZW1vcmllczogU2VxdWVuY2Vbc3RyXSwgdXNlcl9tZXNzYWdlOiBzdHIpIC0+IHN0cjoKICAgICAgICBwYXlsb2FkID0gewogICAgICAgICAgICAibW9kZWwiOiBzZWxmLl9jZmcub3BlbmFpX21vZGVsLAogICAgICAgICAgICAibWVzc2FnZXMiOiBbCiAgICAgICAgICAgICAgICB7InJvbGUiOiAic3lzdGVtIiwgImNvbnRlbnQiOiBzeXN0ZW1fcHJvbXB0fSwKICAgICAgICAgICAgICAgIHsicm9sZSI6ICJ1c2VyIiwgImNvbnRlbnQiOiBfYnVpbGRfcHJvbXB0KHN5c3RlbV9wcm9tcHQsIG1lbW9yaWVzLCB1c2VyX21lc3NhZ2UpfSwKICAgICAgICAgICAgXSwKICAgICAgICAgICAgInRlbXBlcmF0dXJlIjogMC4zLAogICAgICAgIH0KICAgICAgICBoZWFkZXJzID0gewogICAgICAgICAgICAiQXV0aG9yaXphdGlvbiI6IGYiQmVhcmVyIHtzZWxmLl9jZmcub3BlbmFpX2FwaV9rZXl9IiwKICAgICAgICAgICAgIkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIiwKICAgICAgICB9CiAgICAgICAgYXN5bmMgd2l0aCBodHRweC5Bc3luY0NsaWVudCh0aW1lb3V0PXNlbGYuX2NmZy5sbG1fdGltZW91dF9zLCBoZWFkZXJzPWhlYWRlcnMpIGFzIGNsaWVudDoKICAgICAgICAgICAgcmVzcCA9IGF3YWl0IGNsaWVudC5wb3N0KCJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxL2NoYXQvY29tcGxldGlvbnMiLCBqc29uPXBheWxvYWQpCiAgICAgICAgICAgIHJlc3AucmFpc2VfZm9yX3N0YXR1cygpCiAgICAgICAgICAgIGRhdGEgPSByZXNwLmpzb24oKQogICAgICAgICAgICBtc2cgPSBkYXRhLmdldCgiY2hvaWNlcyIsIFt7fV0pWzBdLmdldCgibWVzc2FnZSIsIHt9KS5nZXQoImNvbnRlbnQiKQogICAgICAgICAgICBpZiBub3QgbXNnOgogICAgICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJPcGVuQUkgcmV0dXJuZWQgZW1wdHkgcmVzcG9uc2UiKQogICAgICAgICAgICByZXR1cm4gc3RyKG1zZykuc3RyaXAoKQoKCmNsYXNzIEZhbGxiYWNrQmFja2VuZChMTE1CYWNrZW5kKToKICAgICIiIlRyeSBwcmltYXJ5IHRoZW4gZmFsbGJhY2sgcHJvdmlkZXIgKE9wZW5BSSAtPiBPbGxhbWEpLiIiIgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBwcmltYXJ5OiBMTE1CYWNrZW5kLCBmYWxsYmFjazogTExNQmFja2VuZCB8IE5vbmUpOgogICAgICAgIHNlbGYuX3ByaW1hcnkgPSBwcmltYXJ5CiAgICAgICAgc2VsZi5fZmFsbGJhY2sgPSBmYWxsYmFjawoKICAgIGFzeW5jIGRlZiBnZW5lcmF0ZShzZWxmLCAqLCBzeXN0ZW1fcHJvbXB0OiBzdHIsIG1lbW9yaWVzOiBTZXF1ZW5jZVtzdHJdLCB1c2VyX21lc3NhZ2U6IHN0cikgLT4gc3RyOgogICAgICAgIHRyeToKICAgICAgICAgICAgcmV0dXJuIGF3YWl0IHNlbGYuX3ByaW1hcnkuZ2VuZXJhdGUoCiAgICAgICAgICAgICAgICBzeXN0ZW1fcHJvbXB0PXN5c3RlbV9wcm9tcHQsCiAgICAgICAgICAgICAgICBtZW1vcmllcz1tZW1vcmllcywKICAgICAgICAgICAgICAgIHVzZXJfbWVzc2FnZT11c2VyX21lc3NhZ2UsCiAgICAgICAgICAgICkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgbG9nZ2VyLndhcm5pbmcoIlByaW1hcnkgTExNIGZhaWxlZDogJXMiLCBleGMpCiAgICAgICAgICAgIGlmIHNlbGYuX2ZhbGxiYWNrIGlzIE5vbmU6CiAgICAgICAgICAgICAgICByYWlzZSBMTE1VbmF2YWlsYWJsZSgiUHJpbWFyeSBMTE0gdW5hdmFpbGFibGUiKSBmcm9tIGV4YwogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICByZXR1cm4gYXdhaXQgc2VsZi5fZmFsbGJhY2suZ2VuZXJhdGUoCiAgICAgICAgICAgICAgICAgICAgc3lzdGVtX3Byb21wdD1zeXN0ZW1fcHJvbXB0LAogICAgICAgICAgICAgICAgICAgIG1lbW9yaWVzPW1lbW9yaWVzLAogICAgICAgICAgICAgICAgICAgIHVzZXJfbWVzc2FnZT11c2VyX21lc3NhZ2UsCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjMjoKICAgICAgICAgICAgICAgIGxvZ2dlci5lcnJvcigiRmFsbGJhY2sgTExNIGZhaWxlZDogJXMiLCBleGMyKQogICAgICAgICAgICAgICAgcmFpc2UgTExNVW5hdmFpbGFibGUoIkFsbCBMTE0gYmFja2VuZHMgdW5hdmFpbGFibGUiKSBmcm9tIGV4YzIKCgpkZWYgYnVpbGRfYmFja2VuZChjZmc6IENvbmZpZykgLT4gTExNQmFja2VuZDoKICAgIG9sbGFtYSA9IE9sbGFtYUJhY2tlbmQoY2ZnKQogICAgaWYgY2ZnLnVzZV9vcGVuYWk6CiAgICAgICAgb3BlbmFpID0gT3BlbkFJQmFja2VuZChjZmcpCiAgICAgICAgcmV0dXJuIEZhbGxiYWNrQmFja2VuZChwcmltYXJ5PW9wZW5haSwgZmFsbGJhY2s9b2xsYW1hKQogICAgcmV0dXJuIG9sbGFtYQoKCmRlZiBfYnVpbGRfcHJvbXB0KHN5c3RlbV9wcm9tcHQ6IHN0ciwgbWVtb3JpZXM6IFNlcXVlbmNlW3N0cl0sIHVzZXJfbWVzc2FnZTogc3RyKSAtPiBzdHI6CiAgICBtZW1vcnlfYmxvY2sgPSAiXG4iLmpvaW4oZiItIHttfSIgZm9yIG0gaW4gbWVtb3JpZXMpIGlmIG1lbW9yaWVzIGVsc2UgIk5vIHJlbGV2YW50IG1lbW9yaWVzIGZvdW5kLiIKICAgIHJldHVybiAoCiAgICAgICAgZiJSZWxldmFudCBtZW1vcmllczpcbnttZW1vcnlfYmxvY2t9XG5cbiIKICAgICAgICBmIkN1cnJlbnQgdXNlciBtZXNzYWdlOiB7dXNlcl9tZXNzYWdlfVxuXG4iCiAgICAgICAgIlJlcGx5IG5hdHVyYWxseSBhbmQgY29uY2lzZWx5LiBJZiBhIHJlbGV2YW50IG1lbW9yeSBpcyB1c2VkLCB3ZWF2ZSBpdCBpbiBjb252ZXJzYXRpb25hbGx5LiIKICAgICkK").decode("utf-8")


def _telegram_config_py() -> str:
    import base64
    return base64.b64decode("ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IG9zCmZyb20gZGF0YWNsYXNzZXMgaW1wb3J0IGRhdGFjbGFzcwoKCkBkYXRhY2xhc3MoZnJvemVuPVRydWUpCmNsYXNzIENvbmZpZzoKICAgIHRlbGVncmFtX3Rva2VuOiBzdHIKICAgIGRhdGFfZGlyOiBzdHIKICAgIHN5c3RlbV9wcm9tcHQ6IHN0cgoKICAgICMgTExNIHByb3ZpZGVyIHNlbGVjdGlvbgogICAgdXNlX29wZW5haTogYm9vbAogICAgb3BlbmFpX2FwaV9rZXk6IHN0ciB8IE5vbmUKICAgIG9wZW5haV9tb2RlbDogc3RyCgogICAgb2xsYW1hX3VybDogc3RyCiAgICBvbGxhbWFfbW9kZWw6IHN0cgogICAgbGxtX3RpbWVvdXRfczogaW50CgogICAgIyBiZWhhdmlvcgogICAgcmF0ZV9saW1pdF9wZXJfbWludXRlOiBpbnQKICAgIG1heF9jb250ZXh0X21lbW9yaWVzOiBpbnQKICAgIGNvbnNvbGlkYXRpb25faW50ZXJ2YWxfaG91cnM6IGludAogICAgaW5ib3hfbW9kZV9lbmFibGVkOiBib29sCgogICAgIyBydW50aW1lCiAgICBsb2dfbGV2ZWw6IHN0cgoKCgpkZWYgbG9hZCgpIC0+IENvbmZpZzoKICAgIGRlZiBlbnYoa2V5OiBzdHIsIGRlZmF1bHQ6IHN0ciB8IE5vbmUgPSBOb25lKSAtPiBzdHIgfCBOb25lOgogICAgICAgIHJldHVybiBvcy5nZXRlbnYoa2V5LCBkZWZhdWx0KQoKICAgIHRlbGVncmFtX3Rva2VuID0gZW52KCJURUxFR1JBTV9UT0tFTiIpIG9yIGVudigiQk9UX1RPS0VOIikKICAgIGlmIG5vdCB0ZWxlZ3JhbV90b2tlbjoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIlRFTEVHUkFNX1RPS0VOIG9yIEJPVF9UT0tFTiBpcyByZXF1aXJlZCIpCgogICAgc3lzdGVtX3Byb21wdCA9ICgKICAgICAgICBlbnYoCiAgICAgICAgICAgICJTWVNURU1fUFJPTVBUIiwKICAgICAgICAgICAgIllvdSBhcmUgYSBmcmllbmRseSBBSSBhc3Npc3RhbnQgd2l0aCBwZXJzaXN0ZW50IG1lbW9yeS4gWW91IHJlbWVtYmVyIHBhc3QgY29udmVyc2F0aW9ucyB3aXRoIHRoaXMgdXNlci4gIgogICAgICAgICAgICAiVXNlIHlvdXIgbWVtb3JpZXMgbmF0dXJhbGx5IOKAlCByZWZlcmVuY2UgdGhpbmdzIHRoZXkndmUgdG9sZCB5b3UgYmVmb3JlIHdoZW4gcmVsZXZhbnQuICIKICAgICAgICAgICAgIkJlIHdhcm0sIGhlbHBmdWwsIGFuZCBzaG93IHRoYXQgeW91IGdlbnVpbmVseSByZW1lbWJlciB0aGVtLiBLZWVwIHJlc3BvbnNlcyBjb25jaXNlLiIsCiAgICAgICAgKQogICAgKQoKICAgIHVzZV9vcGVuYWkgPSBib29sKGVudigiT1BFTkFJX0FQSV9LRVkiKSkKCiAgICByZXR1cm4gQ29uZmlnKAogICAgICAgIHRlbGVncmFtX3Rva2VuPXRlbGVncmFtX3Rva2VuLAogICAgICAgIGRhdGFfZGlyPWVudigiREFUQV9ESVIiLCAiLi9kYXRhIikgb3IgIi4vZGF0YSIsCiAgICAgICAgc3lzdGVtX3Byb21wdD1zeXN0ZW1fcHJvbXB0LAogICAgICAgIHVzZV9vcGVuYWk9dXNlX29wZW5haSwKICAgICAgICBvcGVuYWlfYXBpX2tleT1lbnYoIk9QRU5BSV9BUElfS0VZIiksCiAgICAgICAgb3BlbmFpX21vZGVsPWVudigiT1BFTkFJX01PREVMIiwgImdwdC00by1taW5pIiksCiAgICAgICAgb2xsYW1hX3VybD1lbnYoIk9MTEFNQV9VUkwiLCAiaHR0cDovL2xvY2FsaG9zdDoxMTQzNCIpLAogICAgICAgIG9sbGFtYV9tb2RlbD1lbnYoIk9MTEFNQV9NT0RFTCIsICJxd2VuMi41OjE0YiIpLAogICAgICAgIGxsbV90aW1lb3V0X3M9aW50KGVudigiTExNX1RJTUVPVVRfUyIsICIzMCIpKSwKICAgICAgICByYXRlX2xpbWl0X3Blcl9taW51dGU9aW50KGVudigiUkFURV9MSU1JVF9QRVJfTUlOVVRFIiwgIjMwIikpLAogICAgICAgIG1heF9jb250ZXh0X21lbW9yaWVzPWludChlbnYoIk1BWF9DT05URVhUX01FTU9SSUVTIiwgIjUiKSksCiAgICAgICAgY29uc29saWRhdGlvbl9pbnRlcnZhbF9ob3Vycz1pbnQoZW52KCJDT05TT0xJREFUSU9OX0lOVEVSVkFMX0hPVVJTIiwgIjYiKSksCiAgICAgICAgaW5ib3hfbW9kZV9lbmFibGVkPWVudigiSU5CT1hfTU9ERSIsICJ0cnVlIikubG93ZXIoKSBpbiB7IjEiLCAidHJ1ZSIsICJ5ZXMiLCAieSIsICJvbiJ9LAogICAgICAgIGxvZ19sZXZlbD1lbnYoIkxPR19MRVZFTCIsICJJTkZPIikudXBwZXIoKSwKICAgICkK").decode("utf-8")


def _telegram_requirements_txt() -> str:
    import base64
    return base64.b64decode("cHl0aG9uLXRlbGVncmFtLWJvdD49MjEuMC4wCnN5bmFwc2UtYWktbWVtb3J5Pj0wLjExLjEKaHR0cHg+PTAuMjguMAo=").decode("utf-8")


def _telegram_env_txt(token: str, ollama_model: str, openai_key: str = "") -> str:
    lines = [
        f"TELEGRAM_TOKEN={token}",
        "DATA_DIR=./data",
        "OLLAMA_URL=http://localhost:11434",
        f"OLLAMA_MODEL={ollama_model}",
        "LLM_TIMEOUT_S=30",
        "OPENAI_MODEL=gpt-4o-mini",
        "RATE_LIMIT_PER_MINUTE=30",
        "MAX_CONTEXT_MEMORIES=5",
        "CONSOLIDATION_INTERVAL_HOURS=6",
        "INBOX_MODE=true",
        "LOG_LEVEL=INFO",
    ]
    if openai_key:
        lines.append(f"OPENAI_API_KEY={openai_key}")
    return "\n".join(lines) + "\n"


def _write_telegram_files(bot_dir: str, token: str, ollama_model: str, openai_key: str = "") -> None:
    bot_files = {
        "bot.py": _telegram_bot_py(),
        "memory_manager.py": _telegram_memory_manager_py(),
        "llm_backend.py": _telegram_llm_backend_py(),
        "config.py": _telegram_config_py(),
        "requirements.txt": _telegram_requirements_txt(),
        ".env": _telegram_env_txt(token=token, ollama_model=ollama_model, openai_key=openai_key),
    }
    for name, content in bot_files.items():
        _write_text(os.path.join(bot_dir, name), content)


def install_telegram(db_path: str) -> bool:
    """Install Synapse Telegram bot integration without external repo clone."""
    _ = db_path

    print("\n🧠 Synapse Telegram Bot — Setup Wizard")
    print("One-time setup. Takes about 60 seconds.\n")

    print("Step 1: Telegram Bot Token")
    print("Get one from @BotFather in Telegram:")
    print("  1. Open Telegram, search @BotFather")
    print("  2. Send /newbot")
    print("  3. Choose a name (e.g., Synapse Memory Bot)")
    print("  4. Choose a username (e.g., SynapseMemoryBot)")
    print("  5. Copy the token BotFather gives you\n")
    token = _read_input("Paste your bot token: ")

    ollama_model = "qwen2.5:14b"
    print("\nStep 2: LLM Backend")
    models = _ollama_models()
    openai_key = ""
    if models:
        print("✅ Ollama detected.")
        for model in models[:5]:
            print(f"  • {model}")
        print()
        ollama_model = models[0] if models else ollama_model
        if _read_yes_no("Also add OpenAI as fallback?", default=False):
            openai_key = _read_input("OpenAI API key: ")
    else:
        print("⚠️  Ollama not detected.")
        if _read_yes_no("Add OpenAI API key as fallback?", default=True):
            openai_key = _read_input("OpenAI API key: ")

    print("\nStep 3: Generate bot files")
    bot_dir = os.path.expanduser("~/.synapse/telegram-bot")
    os.makedirs(bot_dir, exist_ok=True)
    _write_telegram_files(bot_dir, token=token, ollama_model=ollama_model, openai_key=openai_key)

    # Validate token
    print("\nValidating bot token...")
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/getMe")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("ok"):
            bot_info = data.get("result", {})
            bot_name = bot_info.get("first_name", "Unknown")
            bot_username = bot_info.get("username", "unknown")
            print(f"✅ Token valid! Bot: {bot_name} (@{bot_username})")
        else:
            print("❌ Invalid token. Please check and try again.")
            return False
    except Exception as e:
        print(f"❌ Could not validate token: {e}")
        print("Check your internet connection and token, then try again.")
        return False

    print("\nStep 4: Install dependencies")
    dep = subprocess.run([sys.executable, "-m", "pip", "install",
                          "python-telegram-bot[job-queue]", "httpx", "synapse-ai-memory", "-q"])
    if dep.returncode:
        print("⚠️  Dependency installation returned an error. You may need to run:")
        print('  python3 -m pip install "python-telegram-bot[job-queue]" httpx synapse-ai-memory')

    _write_text(_telegram_config_path(), "installed\n")

    print(f"\n✅ Synapse Telegram bot ready!")
    print(f"   Bot: @{bot_username}")
    print(f"   Data: {bot_dir}")

    # Auto-start
    print("\nStarting bot...")
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(bot_dir, "bot.py")],
            cwd=bot_dir,
            start_new_session=True,
        )
        import time as _time
        _time.sleep(3)
        if proc.poll() is None:
            print(f"✅ Bot is running! (PID: {proc.pid})")
            print(f"   Message @{bot_username} on Telegram to try it out.")
            print(f"\n   To stop: kill {proc.pid}")
            print(f"   To restart: python3 {bot_dir}/bot.py")
        else:
            print("⚠️  Bot exited unexpectedly. Check logs:")
            print(f"   python3 {bot_dir}/bot.py")
    except Exception as e:
        print(f"⚠️  Could not auto-start: {e}")
        print(f"   Start manually: python3 {bot_dir}/bot.py")

    return True


def uninstall_telegram() -> bool:
    """Uninstall Synapse Telegram integration notes."""
    marker = _telegram_config_path()
    if os.path.exists(marker):
        try:
            os.remove(marker)
        except OSError:
            pass
        print("✅ Synapse Telegram integration marker removed.")
        return True
    print("⬜ Synapse Telegram integration marker not found.")
    print("If the bot was set up manually, remove its process and local config files in its own folder.")
    return False


_OLLAMA_CONFIG_PATH = os.path.expanduser("~/.synapse/ollama.json")
_OLLAMA_RECOMMENDATIONS = ["nomic-embed-text", "qwen2.5:1.5b", "nomic-embed-text:latest"]


def _ollama_running(base_url: str = "http://localhost:11434") -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen(base_url + "/api/tags", timeout=1.5) as response:
            payload = json.load(response)
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
        return True, names
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False, []


def install_ollama(db_path: str) -> bool:
    """Enable local Ollama embeddings when available and print model recommendations."""
    _ = db_path
    running, models = _ollama_running()
    if not running:
        print("⚠️  Ollama is not running at http://localhost:11434.")
        print("Start Ollama and rerun: synapse install ollama")
        print("You can install Ollama first from https://ollama.com")
        return False

    available = sorted(set(m for m in models if isinstance(m, str)))
    chosen_model = None
    for recommended in _OLLAMA_RECOMMENDATIONS:
        if recommended in available:
            chosen_model = recommended
            break
    if chosen_model is None and available:
        chosen_model = available[0]

    os.makedirs(os.path.dirname(_OLLAMA_CONFIG_PATH), exist_ok=True)
    payload = {
        "enabled": True,
        "provider": "ollama",
        "url": "http://localhost:11434",
        "embedding_model": chosen_model or _OLLAMA_RECOMMENDATIONS[0],
        "available_models": available,
    }
    _write_text(_OLLAMA_CONFIG_PATH, json.dumps(payload, indent=2))

    if chosen_model:
        print(f"✅ Ollama is available. Recommended embedding model: {chosen_model}")
        print("Set this env var to enable Synapse MCP embeddings:")
        print("  export SYNAPSE_MCP_ENABLE_EMBEDDINGS=1")
    else:
        print("✅ Ollama is available, but no model names were returned.")
    print("Model recommendations:")
    for model in _OLLAMA_RECOMMENDATIONS:
        marker = "✅" if model in available else "  "
        print(f"  {marker} {model}")
    print(f"Saved config to {_OLLAMA_CONFIG_PATH}")
    return True


def uninstall_ollama() -> bool:
    """Disable Ollama embedding helper config."""
    if os.path.exists(_OLLAMA_CONFIG_PATH):
        try:
            os.remove(_OLLAMA_CONFIG_PATH)
        except OSError as exc:
            print(f"⚠️  Could not remove Ollama config: {exc}")
            return False
        print("✅ Ollama integration config removed.")
        print("Unset in running shells if set: unset SYNAPSE_MCP_ENABLE_EMBEDDINGS")
        return True
    print("⬜ Ollama integration config not found.")
    return False


def _verify_openclaw_skill(root: str) -> tuple[bool, str]:
    """Verify OpenClaw/NanoClaw skill files exist and are readable."""
    skill_dir = _skill_dir(root)
    for fname in ("SKILL.md", "manifest.json"):
        path = os.path.join(skill_dir, fname)
        if not os.path.exists(path):
            return False, f"missing {fname}"
        try:
            with open(path, "r", encoding="utf-8"):
                pass
        except OSError as exc:
            return False, f"cannot read {fname}: {exc}"
    return True, "synapse is configured correctly"


def _resolve_candidate_paths(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0] if paths else ""


def install_claude_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    return _install_generic_enhanced("Claude", _claude_config_path, _verify_mcp_file, db_path, dry_run, verify_only, lambda cfg, p: _ensure_synapse_mcp(cfg, p), "Restart Claude to activate.")


def install_cursor_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    return _install_generic_enhanced("Cursor", lambda: _resolve_candidate_paths(_cursor_candidates()), _verify_mcp_file, db_path, dry_run, verify_only, lambda cfg, p: _ensure_synapse_mcp(cfg, p), "Restart Cursor to activate.")


def install_windsurf_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    return _install_generic_enhanced("Windsurf", lambda: _resolve_candidate_paths(_windsurf_candidates()), _verify_mcp_file, db_path, dry_run, verify_only, lambda cfg, p: _ensure_synapse_mcp(cfg, p), "Restart Windsurf to activate.")


def install_continue_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    return _install_generic_enhanced("Continue", lambda: _resolve_candidate_paths(_continue_candidates()), _verify_continue_file, db_path, dry_run, verify_only, lambda cfg, p: _ensure_synapse_continue(cfg, p), "Restart your editor to activate.")


def _install_generic_enhanced(name: str, path_fn, verify_fn, db_path: str, dry_run: bool, verify_only: bool, ensure_fn, ready_hint: str) -> bool:
    """Generic install helper for both standard and editor MCP clients."""
    config_path = path_fn()

    if verify_only:
        ok, detail = verify_fn(config_path, db_path)
        if ok:
            print(f"✅ Verified: {detail}")
        else:
            print(f"❌ Verification failed: {detail}")
        return ok

    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = ensure_fn(config, db_path)

    if dry_run:
        print(f"[dry-run] Would write to {config_path}:")
        print(json.dumps(config, indent=2))
        return True

    try:
        _write_json(config_path, config)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"❌ Install failed for {name}: {exc}")
        return False

    ok, detail = verify_fn(config_path, db_path)
    if ok:
        print(f"✅ Verified: {detail}")
    print(f"Synapse installed for {name}. {ready_hint}")
    return ok


def install_openclaw_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    root = _openclaw_workspace_root()
    if verify_only:
        ok, detail = _verify_openclaw_skill(root)
        print(f"{'✅' if ok else '❌'} {'Verified' if ok else 'Verification failed'}: {detail}")
        return ok
    if dry_run:
        skill_root = _skill_dir(root)
        print(f"[dry-run] Would create files in {skill_root}/:")
        print("  - SKILL.md")
        print("  - manifest.json")
        print("  - setup.sh")
        return True
    _install_openclaw_into(root, "Synapse skill installed for OpenClaw.")
    ok, detail = _verify_openclaw_skill(root)
    if ok:
        print(f"✅ Verified: {detail}")
    return ok


def install_nanoclaw_enhanced(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
    root = _nanoclaw_workspace_root()
    if verify_only:
        ok, detail = _verify_openclaw_skill(root)
        print(f"{'✅' if ok else '❌'} {'Verified' if ok else 'Verification failed'}: {detail}")
        return ok
    if dry_run:
        skill_root = _skill_dir(root)
        print(f"[dry-run] Would create files in {skill_root}/:")
        print("  - SKILL.md")
        print("  - manifest.json")
        print("  - setup.sh")
        return True
    _install_openclaw_into(root, "Synapse skill installed for NanoClaw.")
    ok, detail = _verify_openclaw_skill(root)
    if ok:
        print(f"✅ Verified: {detail}")
    return ok


def _is_installed_and_configured(path: str, verifier, db_path: str) -> bool:
    try:
        ok, _ = verifier(path, db_path)
        return ok
    except Exception:
        return False


def _detect_targets(db_path: str) -> Dict[str, bool]:
    """Return which install targets are already configured."""
    results: Dict[str, bool] = {
        "claude": False,
        "cursor": False,
        "windsurf": False,
        "continue": False,
        "openclaw": False,
        "nanoclaw": False,
        "telegram": False,
        "ollama": False,
    }

    try:
        results["claude"] = _is_installed_and_configured(_claude_config_path(), _verify_mcp_file, db_path)
    except Exception:
        pass

    try:
        results["cursor"] = _is_installed_and_configured(_resolve_candidate_paths(_cursor_candidates()), _verify_mcp_file, db_path)
    except Exception:
        pass

    try:
        results["windsurf"] = _is_installed_and_configured(_resolve_candidate_paths(_windsurf_candidates()), _verify_mcp_file, db_path)
    except Exception:
        pass

    try:
        results["continue"] = _is_installed_and_configured(_resolve_candidate_paths(_continue_candidates()), _verify_continue_file, db_path)
    except Exception:
        pass

    results["openclaw"] = os.path.exists(os.path.join(_openclaw_workspace_root(), "synapse", "SKILL.md"))
    results["nanoclaw"] = os.path.exists(os.path.join(_nanoclaw_workspace_root(), "synapse", "SKILL.md"))
    results["telegram"] = os.path.exists(_telegram_config_path())
    results["ollama"] = os.path.exists(_OLLAMA_CONFIG_PATH)
    return results


def install_all(db_path: str, dry_run: bool = False) -> Dict[str, bool]:
    """Install to all detected/possible targets at once."""
    results = {}
    for name, fn in [
        ("claude", install_claude_enhanced),
        ("cursor", install_cursor_enhanced),
        ("windsurf", install_windsurf_enhanced),
        ("continue", install_continue_enhanced),
        ("openclaw", install_openclaw_enhanced),
        ("nanoclaw", install_nanoclaw_enhanced),
        ("telegram", lambda p, dry_run=dry_run, verify_only=False: install_telegram(p) if not dry_run else True),
        ("ollama", lambda p, dry_run=dry_run, verify_only=False: install_ollama(p) if not dry_run else True),
    ]:
        try:
            results[name] = fn(db_path)
        except Exception as exc:
            print(f"⚠️  {name}: {exc}")
            results[name] = False
    return results


def uninstall_claude() -> bool:
    """Remove Synapse from Claude Desktop MCP config."""
    config_path = _claude_config_path()
    if not os.path.exists(config_path):
        print("⬜ Claude Desktop config not found — nothing to remove.")
        return False
    config = _read_json(config_path)
    mcp_servers = config.get("mcpServers", {})
    if not isinstance(mcp_servers, dict) or "synapse" not in mcp_servers:
        print("⬜ Synapse not found in Claude Desktop config.")
        return False
    del mcp_servers["synapse"]
    config["mcpServers"] = mcp_servers
    _write_json(config_path, config)
    print("✅ Synapse removed from Claude Desktop. Restart Claude to apply.")
    return True


def _uninstall_generic(config_path: str, label: str, continue_type: bool = False) -> bool:
    if not os.path.exists(config_path):
        print(f"⬜ {label} config not found — nothing to remove.")
        return False
    data = _read_json(config_path)
    if continue_type:
        exp = data.get("experimental")
        if not isinstance(exp, dict):
            print(f"⬜ Synapse not found in {label} config.")
            return False
        servers = exp.get("modelContextProtocolServers")
        if not isinstance(servers, list):
            print(f"⬜ Synapse not found in {label} config.")
            return False
        filtered = [item for item in servers if not (_is_synapse_continue_entry(item) if isinstance(item, dict) else False)]
        if len(filtered) == len(servers):
            print(f"⬜ Synapse not found in {label} config.")
            return False
        exp["modelContextProtocolServers"] = filtered
        data["experimental"] = exp
    else:
        mcp_servers = data.get("mcpServers")
        if not isinstance(mcp_servers, dict) or "synapse" not in mcp_servers:
            print(f"⬜ Synapse not found in {label} config.")
            return False
        del mcp_servers["synapse"]
        data["mcpServers"] = mcp_servers
    _write_json(config_path, data)
    print(f"✅ Synapse removed from {label}. Restart app to apply.")
    return True


def uninstall_cursor() -> bool:
    return _uninstall_generic(_resolve_candidate_paths(_cursor_candidates()), "Cursor")


def uninstall_windsurf() -> bool:
    return _uninstall_generic(_resolve_candidate_paths(_windsurf_candidates()), "Windsurf")


def uninstall_continue() -> bool:
    return _uninstall_generic(_resolve_candidate_paths(_continue_candidates()), "Continue", continue_type=True)


def uninstall_openclaw() -> bool:
    """Remove Synapse skill from OpenClaw."""
    skill_dir = _skill_dir(_openclaw_workspace_root())
    if not os.path.exists(skill_dir):
        print("⬜ Synapse skill not found in OpenClaw.")
        return False
    print(f"  Removing {skill_dir}")
    shutil.rmtree(skill_dir)
    print("✅ Synapse skill removed from OpenClaw.")
    return True


def uninstall_nanoclaw() -> bool:
    """Remove Synapse skill from NanoClaw."""
    skill_dir = _skill_dir(_nanoclaw_workspace_root())
    if not os.path.exists(skill_dir):
        print("⬜ Synapse skill not found in NanoClaw.")
        return False
    print(f"  Removing {skill_dir}")
    shutil.rmtree(skill_dir)
    print("✅ Synapse skill removed from NanoClaw.")
    return True


def uninstall_all() -> Dict[str, bool]:
    """Uninstall from all detected targets."""
    results = {}
    for name, fn in [
        ("claude", uninstall_claude),
        ("cursor", uninstall_cursor),
        ("windsurf", uninstall_windsurf),
        ("continue", uninstall_continue),
        ("openclaw", uninstall_openclaw),
        ("nanoclaw", uninstall_nanoclaw),
        ("telegram", uninstall_telegram),
        ("ollama", uninstall_ollama),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"⚠️  {name}: {exc}")
            results[name] = False
    return results


class ClientInstaller:
    TARGETS = {
        "claude": install_claude,
        "cursor": install_cursor,
        "windsurf": install_windsurf,
        "continue": install_continue,
        "openclaw": install_openclaw,
        "nanoclaw": install_nanoclaw,
        "telegram": install_telegram,
        "ollama": install_ollama,
    }
    ENHANCED_TARGETS = {
        "claude": install_claude_enhanced,
        "cursor": install_cursor_enhanced,
        "windsurf": install_windsurf_enhanced,
        "continue": install_continue_enhanced,
        "openclaw": install_openclaw_enhanced,
        "nanoclaw": install_nanoclaw_enhanced,
        "telegram": lambda db_path, dry_run=False, verify_only=False: install_telegram(db_path),
    }
