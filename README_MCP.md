# Synapse MCP Guide

Synapse exposes memory tools via MCP for Claude Desktop, Cursor, Windsurf, Continue (VS Code), OpenClaw, and other MCP-compatible clients.

## Install

```bash
pip install synapse-ai-memory
# or
curl -fsSL https://synapse.ai/install.sh | bash
```

## Recommended setup

```bash
synapse onboard --flow quickstart
synapse onboard --flow quickstart --non-interactive --json   # CI/non-interactive
```

## Managed MCP launcher

Synapse configures clients to use a stable managed launcher:

- `~/.synapse/bin/synapse-mcp`
- runtime root: `~/.synapse/runtime`
- data root: `~/.synapse`

Direct run:

```bash
~/.synapse/bin/synapse-mcp --data-dir ~/.synapse
```

## Integration lifecycle commands

```bash
synapse integrations list
synapse integrations install <claude|cursor|windsurf|continue|openclaw>
synapse integrations test <name>
synapse integrations repair <name>
synapse integrations open <name>
```

`synapse integrations list --json` includes connector metadata (`type`, `tier`, `commands`, `capabilities`, `doctor_checks`, `example_prompt`).

## MCP config examples

```json
{
  "mcpServers": {
    "synapse-memory": {
      "command": "/absolute/path/to/.synapse/bin/synapse-mcp",
      "args": ["--data-dir", "/Users/you/.synapse"]
    }
  }
}
```

Config file locations:

- Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Cursor: `~/.cursor/mcp.json`
- Windsurf: `~/.windsurf/mcp.json`
- Continue (VS Code): `~/.continue/config.json`

## Policy receipts

```bash
synapse permit receipts --last 3
synapse permit receipts --last 3 --json
```

Receipts are scaffolded in **Phase-1** form and include allow/deny decisions, matched rules, scope, actor/app IDs, and memory-count deltas where available.

## Exposed tools

Launcher modes:

- `~/.synapse/bin/synapse-mcp --data-dir ...` → appliance surface (default)
- `synapse-mcp --data-dir ... --mode full` → extended surface

Common families:

- Memory capture and recall (`remember`, `recall`, `ingest`)
- Inbox review (`inbox_list`, `inbox_approve`, `inbox_reject`, `inbox_redact`, `inbox_pin`)
- Vaults (`vault_list`, `vault_create`, `vault_switch`)
- Forget/privacy (`forget`, `forget_topic`, `redact`, `gdpr_delete`, `natural_forget`)
- Context + inspection (`compile_context`, `beliefs`, `contradictions`, `fact_history`, `timeline`, `communities`, `count`, `concepts`, `browse`)
- Maintenance (`sleep`, `consolidate`, `prune`)

Use `list_tools` from your running server to verify exact tool names and schemas.

## Troubleshooting

- `synapse doctor` validates MCP and integration health.
- `synapse doctor --fix` applies safe integration remediation where supported.
