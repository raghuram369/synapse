# Installer & Onboarding Guide

## CI / non-interactive installs

### 1) One-shot bootstrap in CI

```bash
curl -fsSL https://synapse.ai/install.sh | bash -s -- --no-onboard --non-interactive
```

Then run deterministic onboarding with defaults:

```bash
synapse onboard \
  --flow quickstart \
  --non-interactive \
  --json \
  --policy-template private \
  --default-scope private \
  --default-sensitive on \
  --db ~/.synapse
```

### 2) CI with explicit client installation (no prompts)

```bash
synapse install claude --db ~/.synapse
synapse install cursor --db ~/.synapse
synapse integrations list --json --db ~/.synapse
```

### 3) Common CI checks

```bash
synapse doctor --non-interactive --json --db ~/.synapse
synapse permit receipts --last 1 --json --db ~/.synapse
```

## Python onboarding example (non-interactive)

```python
import subprocess

subprocess.run([
    "synapse",
    "onboard",
    "--flow",
    "advanced",
    "--non-interactive",
    "--json",
    "--policy-template",
    "private",
    "--default-scope",
    "private",
    "--default-sensitive",
    "on",
], check=True)
```

Use this pattern in test harnesses and automated provisioning jobs. The JSON output is stable for assertions.

## TypeScript onboarding example (non-interactive)

```ts
import { execSync } from 'node:child_process';

const cmd = [
  'synapse',
  'onboard',
  '--flow',
  'quickstart',
  '--non-interactive',
  '--json',
  '--policy-template', 'private',
  '--default-scope', 'private',
  '--default-sensitive', 'on',
].join(' ');

execSync(cmd, {
  stdio: 'inherit',
  env: {
    ...process.env,
    SYNAPSE_DB: '/workspace/.synapse',
  },
});
```

## Deep-link install page placeholders

Use these deep-link placeholders once destination pages are published.

- `https://synapse.ai/install/claude` (Claude Desktop install + MCP payload)
- `https://synapse.ai/install/cursor` (Cursor MCP install)
- `https://synapse.ai/install/windsurf` (Windsurf MCP install)
- `https://synapse.ai/install/continue` (VS Code Continue install)
- `https://synapse.ai/install/openai-starter` (OpenAI starter kit bootstrap)
- `https://synapse.ai/install/langgraph-starter` (LangGraph starter kit bootstrap)
- `https://synapse.ai/install/claude-mcp-starter` (Claude-MCP starter kit bootstrap)

These placeholders should map to static landing pages with copy/paste-ready install snippets.
