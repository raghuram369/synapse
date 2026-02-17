# Install Synapse for Continue

Use this page to configure Synapse MCP for the Continue extension.

## Canonical command flow

1. Bootstrap Synapse:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

2. Install Continue integration:

```bash
synapse install continue --db "$HOME/.synapse"
```

3. Validate and open settings:

```bash
synapse integrations test continue --db "$HOME/.synapse"
synapse integrations open continue --db "$HOME/.synapse"
```

4. Restart your editor after configuration is written.

## Canonical docs / quick links

- [Installer flow and CI-safe defaults](../INSTALLER_GUIDES.md)
- [Continue integration wiring in installer logic](../../installer.py)

## Notes

- Continue stores MCP transport config in `~/.continue/config.json`.
- Run quickstart onboarding to wire multiple clients:

```bash
synapse onboard --flow quickstart --db "$HOME/.synapse"
```
