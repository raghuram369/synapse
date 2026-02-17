# Install Synapse for OpenClaw

Use this page to install the Synapse OpenClaw skill.

## Canonical command flow

1. Bootstrap Synapse:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

2. Install OpenClaw integration:

```bash
synapse install openclaw --db "$HOME/.synapse"
```

3. Validate and open skill workspace:

```bash
synapse integrations test openclaw --db "$HOME/.synapse"
synapse integrations open openclaw --db "$HOME/.synapse"
```

`synapse integrations open openclaw` opens:

- `~/.openclaw/workspace/skills/synapse`

## Canonical docs / quick links

- [Installer flow and CI-safe defaults](../INSTALLER_GUIDES.md)
- [OpenClaw skill schema and template](../../installer.py)

## Notes

- OpenClaw install drops `SKILL.md`, `manifest.json`, and `setup.sh` into the skills folder.
- OpenClaw is a channel connector (not an MCP JSON client file).
