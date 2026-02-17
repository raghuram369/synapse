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

## Deep-link install pages

Use these live install pages (copy/paste-ready snippets are embedded):

- https://synapse.ai/install/claude
- https://synapse.ai/install/cursor
- https://synapse.ai/install/windsurf
- https://synapse.ai/install/continue
- https://synapse.ai/install/openclaw

These pages point at in-repo landing docs under `docs/onboarding/install/`.
