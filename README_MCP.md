# Synapse AI Memory MCP Server (Claude Desktop)

This repository includes an MCP (Model Context Protocol) server that exposes Synapse AI Memory as Claude Desktop tools over stdio.

## Files

- `mcp_server.py`: MCP server (stdio transport) that imports `synapse.py` from this directory.
- `claude_desktop_config.json`: example Claude Desktop MCP configuration.

## Install

1. Install the MCP SDK:

```bash
python3 -m pip install mcp
```

2. From this repo directory, you can sanity-check the server starts (it will wait for JSON-RPC on stdin):

```bash
python3 mcp_server.py --data-dir ~/.synapse
```

Synapse AI Memory will persist to:

- `~/.synapse/synapse.log`
- `~/.synapse/synapse.snapshot`

## Configure Claude Desktop

1. Locate Claude Desktop's config file for MCP servers.
2. Add an entry like `claude_desktop_config.json`, replacing paths with absolute paths on your machine.

Key fields:

- command: `python3`
- args: absolute path to `mcp_server.py`, plus optional `--data-dir`

Example:

```json
{
  "mcpServers": {
    "synapse-memory": {
      "command": "python3",
      "args": [
        "/Users/you/path/to/synapse-mcp/mcp_server.py",
        "--data-dir",
        "/Users/you/.synapse"
      ]
    }
  }
}
```

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
