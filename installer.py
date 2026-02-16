#!/usr/bin/env python3
"""Cross-client installation helpers for Synapse."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict


def _python_binary() -> str:
    return "python3"


def _mcp_server_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))


def _claude_config_path() -> str:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable is required for Claude config on Windows.")
        return os.path.join(appdata, "Claude", "claude_desktop_config.json")
    return os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")


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


def _skill_dir(workspace_root: str) -> str:
    return os.path.join(workspace_root, "synapse")


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content.rstrip() + "\n")


def _openclaw_workspace_root() -> str:
    return os.path.expanduser("~/.openclaw/workspace/skills")


def _nanoclaw_workspace_root() -> str:
    return os.path.expanduser("~/.nanoclaw/workspace/skills")


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
        "version": "0.7.0",
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
    _write_text(
        os.path.join(skill_root, "manifest.json"),
        json.dumps(_openclaw_manifest(), indent=2),
    )
    _write_text(os.path.join(skill_root, "setup.sh"), _openclaw_setup_sh())
    print(print_message)


def install_claude(db_path):
    """Auto-configure Claude Desktop MCP:
    1. Find Claude config: ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
       or %APPDATA%/Claude/claude_desktop_config.json (Windows)
    2. Read existing config (or create new)
    3. Add/update mcpServers.synapse entry with:
       - command: path to python3
       - args: [path to mcp_server.py, --db, db_path]
    4. Write config back
    5. Print: 'Synapse installed for Claude Desktop. Restart Claude to activate.'
    Handle: config doesn't exist yet, synapse already configured (update), backup existing config.
    """

    config_path = _claude_config_path()
    config = _read_json(config_path)

    if os.path.exists(config_path):
        _backup_file(config_path)
    mcp_servers = config.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        config["mcpServers"] = mcp_servers

    mcp_servers["synapse"] = {
        "command": _python_binary(),
        "args": [_mcp_server_path(), "--db", db_path],
    }
    _write_json(config_path, config)
    print("Synapse installed for Claude Desktop. Restart Claude to activate.")


def install_openclaw(db_path):
    """Drop a ready skill folder:
    1. Find OpenClaw workspace: ~/.openclaw/workspace/skills/
    2. Create synapse/ folder with SKILL.md + manifest.json + setup.sh
    3. SKILL.md instructs the agent to use /mem commands or MCP
    4. manifest.json has permission declarations
    5. Print: 'Synapse skill installed for OpenClaw.'
    """

    _ = db_path
    _install_openclaw_into(_openclaw_workspace_root(), "Synapse skill installed for OpenClaw.")


def install_nanoclaw(db_path):
    """Similar to OpenClaw but for NanoClaw layout."""

    _ = db_path
    _install_openclaw_into(_nanoclaw_workspace_root(), "Synapse skill installed for NanoClaw.")


class ClientInstaller:
    TARGETS = {
        "claude": install_claude,
        "openclaw": install_openclaw,
        "nanoclaw": install_nanoclaw,
    }
