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


def _mcp_payload(db_path: str) -> Dict[str, Any]:
    return {
        "command": _python_binary(),
        "args": [_mcp_server_path(), "--db", db_path],
    }


def _continue_payload(db_path: str) -> Dict[str, Any]:
    return {
        "transport": {
            "type": "stdio",
            "command": _python_binary(),
            "args": [_mcp_server_path(), "--db", db_path],
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
        if transport.get("command") == _python_binary() and isinstance(transport.get("args"), list):
            transport_args = transport.get("args")
            if transport_args and _mcp_server_path() in transport_args:
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
    return transport.get("command") == _python_binary() and isinstance(transport.get("args"), list) and _mcp_server_path() in transport.get("args", [])


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


def install_claude(db_path):
    """Auto-configure Claude Desktop MCP."""
    config_path = _claude_config_path()
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_mcp(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Claude Desktop. Restart Claude to activate.")


def install_cursor(db_path):
    """Auto-configure Cursor MCP settings."""
    config_path = _resolve_path(_cursor_candidates())
    config = _read_json(config_path)
    if os.path.exists(config_path):
        _backup_file(config_path)
    config = _ensure_synapse_mcp(config, db_path)
    _write_json(config_path, config)
    print("Synapse installed for Cursor. Restart Cursor to activate.")


def install_windsurf(db_path):
    """Auto-configure Windsurf MCP settings."""
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


def install_telegram(db_path: str) -> bool:
    """Install Synapse Telegram bot integration."""
    _ = db_path
    bot_dir = _detect_telegram_dir()
    if not bot_dir:
        print("❌ Synapse bot files not found.")
        print("Please clone or download the bot repository first:")
        print("  git clone https://github.com/your-org/synapse-bot.git")
        print("  # or place it at one of these paths:")
        for candidate in _telegram_candidates():
            print(f"  - {candidate}")
        print("Then run: synapse install telegram")
        return False

    setup_py = os.path.join(bot_dir, "setup.py")
    if not os.path.exists(setup_py):
        print("⚠️  Found synapse-bot folder but setup.py is missing.")
        print(f"  Location: {bot_dir}")
        print("Please run manual setup steps provided by the repository.")
        return False

    print(f"Launching Telegram bot setup wizard from: {bot_dir}")
    try:
        subprocess.run([sys.executable, setup_py], cwd=bot_dir, check=True)
        _write_text(_telegram_config_path(), "installed\n")
        print("✅ Synapse Telegram bot setup complete.")
        return True
    except FileNotFoundError:
        print("❌ Python interpreter not found. Run manually with: python setup.py")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"❌ Telegram bot setup failed (exit code {exc.returncode}).")
        print("Manual setup fallback:")
        print(f"  cd {bot_dir}")
        print(f"  {sys.executable} setup.py")
        return False
    except OSError as exc:
        print(f"❌ Unable to launch Telegram setup: {exc}")
        return False


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
    }
