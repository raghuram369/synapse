# Compatibility Matrix

_Last updated: 2026-02-16_

This matrix tracks officially supported combinations for the current distribution and installer surface.

## MCP + Installer Compatibility

| Platform | Python package (`pip install synapse-ai-memory`) | One-line bootstrap (`curl ... | bash`) | MCP auto-install (`synapse install <client>`) | Known limitations |
|---|---:|---:|---:|---|
| **macOS 11+ (arm64/x86_64)** | ✅ 3.10–3.13 | ✅ | ✅ Claude Desktop, Cursor, Windsurf, Continue | Claude Desktop config may require app restart to pick up changes |
| **Linux (x86_64/aarch64)** | ✅ 3.10–3.13 | ✅ | ✅ Cursor, Windsurf, Continue | Claude/Cursor auto-config flows are best-effort on desktop-managed runtimes |
| **Windows 10/11** | ✅ 3.10–3.13 | ⚠️ (manual Python bootstrap recommended) | ✅ Cursor, Windsurf, Continue, OpenClaw/NanoClaw (`install`) | Claude Desktop installer path detection is best-effort; verify `~\\AppData\\...` config path |
| **CI/containers (Linux)** | ✅ 3.10–3.13 (python-only) | ✅ (`--no-onboard --non-interactive`) | ✅ via `synapse onboard --non-interactive ...` and JSON-safe payload generation | Headless runs may require writable `$HOME/.synapse` |

## Runtime Support Notes

- The managed launcher is `~/.synapse/bin/synapse-mcp` and is used by integration installers.
- Python runtime shim uses a stable `~/.synapse/runtime/python` symlink/copy where possible.
- If symlink permissions are restricted, installer falls back to binary copy semantics.

## Framework Integration Support (Python)

| Integration | Package | CLI install flow | Notes |
|---|---|---|---|
| **OpenAI** | `openai` | ✅ In-repo example package | Docs + examples available in `integrations/openai/` |
| **LangGraph** | `langgraph` | ✅ In-repo example package | Supports both Checkpointer + Store adapters |
| **LangChain** | `langchain` | ✅ In-repo example package | See `integrations/langchain/` |
| **CrewAI** | `crewai` | ✅ In-repo example package | See `integrations/crewai/` |
| **Claude (API)** | `anthropic` | ✅ In-repo example package | Not an MCP host auto-config target |

## Known Runtime Limitations

- **Node bootstrap** (`npx @synapse-memory/setup@latest`) is optional and skipped automatically when `npx` is missing.
- **Windsurf path resolution** may vary by install package; installer tests primary candidate paths and writes best-effort.
- **Claude Desktop auto-config** currently expects standard user config location per platform.
