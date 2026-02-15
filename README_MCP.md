# Synapse MCP Server (Claude Desktop)

This repository includes an MCP (Model Context Protocol) server that exposes Synapse as Claude Desktop tools over stdio.

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

Synapse will persist to:

- `~/.synapse/synapse.log`
- `~/.synapse/synapse.snapshot`

## Configure Claude Desktop

1. Locate Claude Desktop's config file for MCP servers.
2. Add an entry like `claude_desktop_config.json`, replacing paths with absolute paths on your machine.

Key fields:

- `command`: `python3`
- `args`: absolute path to `mcp_server.py`, plus optional `--data-dir`

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

- `remember`: store a memory (`content`, `memory_type`, `metadata`, `episode`)
- `recall`: search memories (`query`, `limit`, `memory_type`)
- `forget`: delete by ID (`memory_id`)
- `count`: count stored (non-consolidated) memories
- `link`: create an edge (`source_id`, `target_id`, `edge_type`, `weight`)
- `concepts`: list concepts in the graph

## Notes

- The server writes JSON-RPC only to stdout; logs go to stderr.
- Writes (`remember`, `forget`, `link`) call `flush()` to persist immediately.
- Embeddings are disabled by default to avoid latency from optional local Ollama calls. To enable:
  - set `SYNAPSE_MCP_ENABLE_EMBEDDINGS=1` in the environment for the server process.
