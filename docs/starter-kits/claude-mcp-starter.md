# Claude + MCP Starter Kit

## Repo template pointers

- Repo scaffold (starter): _placeholder_ `https://github.com/synapse-ai-memory/claude-mcp-starter`
- In-repo implementation example: `integrations/claude/example.py`
- MCP tools + handlers: `integrations/claude/tool.py`
- MCP/Claude guide source: `integrations/claude/README.md`

## Required files

- `claude_desktop_config.json` template with `synapse` MCP server entry
- `app.py` that initializes `SynapseClaudeMemory` **or** tool-based memory handlers
- `requirements.txt` with `synapse-ai-memory`, `anthropic`

## Setup flow

```bash
synapse install claude
# or write a direct MCP entry to Claude config and point it to ~/.synapse/bin/synapse-mcp

# In app runtime
export ANTHROPIC_API_KEY=...
python app.py
```

## Compatibility note

For Claude Desktop, prefer the managed launcher (`~/.synapse/bin/synapse-mcp`) so Python runtime isolation remains stable across updates.
