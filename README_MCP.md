# Synapse AI Memory MCP Server

Exposes Synapse AI Memory as MCP tools over stdio. Works with **Claude Desktop, Cursor, Windsurf, VS Code (Continue)**, and any MCP-compatible client.

## Quick Setup (recommended)

```bash
pip install synapse-ai-memory
synapse install claude      # auto-configures Claude Desktop
synapse install cursor      # auto-configures Cursor
synapse install windsurf    # auto-configures Windsurf
synapse install continue    # auto-configures VS Code Continue
```

The installer also creates a stable, absolute MCP launcher at:

- `~/.synapse/bin/synapse-mcp`
- managed runtime path: `~/.synapse/runtime/python` (symlink to the Python used to run the installer; fallback to a local copy if needed).

No `PATH` changes are required.

Or use the interactive wizard that auto-detects everything:

```bash
synapse setup
```

## Manual Setup

1. Install the MCP SDK:

```bash
python3 -m pip install mcp
```

2. The installer creates a managed stable wrapper that should be used by clients:

```bash
~/.synapse/bin/synapse-mcp --data-dir ~/.synapse
```

3. Add to your client's MCP config:

```json
{
  "mcpServers": {
    "synapse-memory": {
      "command": "/absolute/path/to/.synapse/bin/synapse-mcp",
      "args": [
        "--data-dir",
        "/Users/you/.synapse"
      ]
    }
  }
}
```

Config locations:
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Cursor**: `~/.cursor/mcp.json`
- **Windsurf**: `~/.windsurf/mcp.json`
- **Continue (VS Code)**: `~/.continue/config.json`

## Exposed Tools

- `remember`: store a memory (`content`, `memory_type`, `metadata`, `episode`, optional `user_id` for vault routing)
- `recall`: search memories (`query`, `limit`, `memory_type`, `explain`, `show_disputes`, `exclude_conflicted`, optional `user_id` for vault routing)
- `vault_list`: list all vaults
- `vault_create`: create a vault (`vault_id`, optional `user_id`)
- `vault_switch`: switch active vault (`vault_id`)
- `inbox_list`: list pending memory-review items
- `inbox_approve`: approve a pending item (`item_id`)
- `inbox_reject`: reject a pending item (`item_id`)
- `inbox_redact`: redact and approve a pending item (`item_id`, `redacted_content`)
- `inbox_pin`: pin and approve a pending item (`item_id`)
- `natural_forget`: forget by natural language (`command`, optional `dry_run`)
- `list`: list memories (`limit`, `offset`, `sort`)
- `browse`: browse by concept (`concept`, `limit`, `offset`)
- `count`: count stored (non-consolidated) memories
- `concepts`: list concepts in the graph
- `link`: create an edge (`source_id`, `target_id`, `edge_type`, `weight`)
- `forget`: delete by ID (`memory_id`)

Forgetting + privacy:

- `forget_topic`: forget all memories related to a topic/concept (`topic`)
- `redact`: redact fields while preserving metadata and graph links (`memory_id`, `fields`)
- `gdpr_delete`: full delete by user tag or concept (`user_id`, `concept`)

Truth maintenance + inspection:

- `contradictions`: list unresolved contradictions
- `beliefs`: list current belief versions (worldview)

Maintenance:

- `sleep`: run the full maintenance cycle (consolidate/promote/pattern-mine/prune/cleanup/communities)
- `communities`: list detected concept communities

LLM integration:

- `compile_context`: compile an LLM-ready ContextPack (`query`, `budget`, `policy`)

Maintenance + temporal helpers:

- `consolidate`: consolidate similar memories (`min_cluster_size`, `similarity_threshold`)
- `fact_history`: show the evolution chain for a fact (`query`)
- `timeline`: timeline-style view for a fact (`query`)
- `hot_concepts`: show most-activated concepts (`k`)
- `prune`: prune old/weak memories (`max_age_days`, `min_strength`)

## Notes

- The server writes JSON-RPC only to stdout; logs go to stderr.
- Writes (`remember`, `forget`, `link`) call `flush()` to persist immediately.
- Embeddings are disabled by default to avoid latency from optional local Ollama calls. To enable:
  - set `SYNAPSE_MCP_ENABLE_EMBEDDINGS=1` in the environment for the server process.
