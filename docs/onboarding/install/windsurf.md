# Install Synapse for Windsurf

Use this page to configure Synapse MCP for Windsurf.

## Canonical command flow

1. Bootstrap Synapse:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

2. Install MCP integration:

```bash
synapse install windsurf --db "$HOME/.synapse"
```

3. Validate and open settings:

```bash
synapse integrations test windsurf --db "$HOME/.synapse"
synapse integrations open windsurf --db "$HOME/.synapse"
```

4. Restart Windsurf after configuration is written.

## Canonical docs / quick links

- [Installer flow and CI-safe defaults](../INSTALLER_GUIDES.md)
- [Windsurf MCP path details in installer code](../../installer.py)

## Notes

- Windsurf installs to `~/.windsurf/mcp.json` (or platform equivalent).
- If you prefer non-interactive multi-client onboarding, run:

```bash
synapse onboard --flow quickstart --non-interactive --db "$HOME/.synapse"
```
