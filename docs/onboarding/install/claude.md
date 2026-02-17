# Install Synapse for Claude Desktop

Use this page to configure `synapse-mcp` for Claude Desktop.

## Canonical command flow

1. Bootstrap Synapse:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

2. Install MCP integration for Claude:

```bash
# Use a shared DB path for all installers/tools
synapse install claude --db "$HOME/.synapse"
```

3. Validate and open settings:

```bash
synapse integrations test claude --db "$HOME/.synapse"
synapse integrations open claude --db "$HOME/.synapse"
```

4. Restart Claude Desktop after configuration is written.

## Canonical docs / quick links

- [Installer flow and CI-safe defaults](../INSTALLER_GUIDES.md)
- [Claude integration reference](../../integrations/claude/README.md)

## Notes

- `synapse install claude` writes `synapse` under your Claude MCP config (`mcpServers.synapse`).
- For interactive onboarding of multiple clients at once, run:

```bash
synapse onboard --flow quickstart --db "$HOME/.synapse"
```
