# Install Synapse for Cursor

Use this page to configure Synapse MCP for Cursor.

## Canonical command flow

1. Bootstrap Synapse:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

2. Install MCP integration:

```bash
synapse install cursor --db "$HOME/.synapse"
```

3. Validate and open settings:

```bash
synapse integrations test cursor --db "$HOME/.synapse"
synapse integrations open cursor --db "$HOME/.synapse"
```

4. Restart Cursor after configuration is written.

## Canonical docs / quick links

- [Installer flow and CI-safe defaults](../INSTALLER_GUIDES.md)
- [Cursor MCP path details in installer code](../../installer.py)

## Notes

- Cursor installs to `~/.cursor/mcp.json` (or platform equivalent).
- For one-step setup of installed clients, run:

```bash
synapse onboard --flow quickstart --db "$HOME/.synapse"
```
