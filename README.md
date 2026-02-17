# Synapse AI Memory

A local-first memory engine for AI agents. Store facts, context, and conversation history in one place, then retrieve them as structured context with policy + privacy controls.

![PyPI](https://img.shields.io/pypi/v/synapse-ai-memory.svg)  ![Python](https://img.shields.io/pypi/pyversions/synapse-ai-memory.svg)  ![License](https://img.shields.io/badge/license-MIT-blue.svg)

Current release: **0.13.0**

---

## Why this release

Recent work focuses on **onboarding + runtime + integrations + policy receipts**. The repo now provides a cleaner zero-prompt setup path, explicit runtime mode controls, and practical auditability for policy enforcement decisions.

## Install

```bash
pip install synapse-ai-memory
```

Optional one-step installer still available:

```bash
curl -fsSL https://synapse.ai/install.sh | bash
```

## Quick start (recommended)

```bash
# interactive quickstart (flow does onboarding + integration probes)
synapse onboard --flow quickstart

# CI/non-interactive default setup
synapse onboard --flow quickstart --non-interactive --json

# advanced flow with explicit defaults
synapse onboard --flow advanced \
  --policy-template private \
  --default-scope private \
  --default-sensitive on \
  --enable-service \
  --service-schedule daily
```

Python API is straightforward:

```python
from synapse import Synapse
s = Synapse()
s.remember("I prefer dark mode")
print(s.recall("what theme?"))
```

## Core flow model

- **Vault-aware memory**: isolate by `user_id` when enabled.
- **Review-first path** for sensitive/uncertain captures via the Inbox.
- **Natural language forgetting** for targeted cleanup.
- **Scope + sensitive controls** with conservative defaults.
- **Portable formats** (`.brain`, checkpoints, imports/exports).
- **Federation primitives** (`synapse federate`, `sync`, `push`, `pull`) available when configured.
- **Zero-LLM indexing** for core recall/ingest.

## Runtime and service management

- `synapse up --port <port>` starts the appliance daemon (MCP stdio server + HTTP JSON-RPC fallback)
- `synapse down` stops it.
- `synapse status` shows running status and basic store stats.
- `synapse service install|uninstall|status` manages autostart (`launchd` / `systemd`).

Runtime defaults are local-only; integrations may add network calls only when used.

## Integrations (client + contract metadata)

`synapse install` and `synapse integrations` are now split by use-case:

```bash
# quick probe + integration ops
synapse integrations list --db ~/.synapse
synapse integrations list --json

# install / verify / repair / open flows for detected clients
synapse integrations install claude
synapse integrations test cursor
synapse integrations repair windsurf
synapse integrations open continue
```

`synapse integrations list --json` includes connector contract metadata (`type`, `tier`, `commands`, `capabilities`, `doctor_checks`, `example_prompt`) for supported built-ins.

`synapse install` still handles broader install targets (for example `telegram`, `ollama`, `nanoclaw`) as a direct installer path.

## Policy receipts (Phase-1)

Audit trail for permit decisions is now available from policy hooks.

```bash
synapse permit receipts --last 5
synapse permit receipts --last 5 --json
```

Notes:
- Receipts are written to `<db>/receipts/permit_receipts.jsonl`.
- Schema is stable (`synapse.permit.receipt.v1`) for JSON output.
- This is a **Phase-1 implementation**: it records enforcement decisions and reasons, but log volume and retention policies are intentionally minimal.

## MCP server

The project exposes MCP tools via:

- `synapse serve` (default appliance mode)
- `synapse serve --http --port 8765` (HTTP JSON-RPC)
- `synapse-mcp` wrapper (managed launcher at `~/.synapse/bin/synapse-mcp`)

Use:

```bash
synapse serve                 # local stdio-compatible mode
synapse serve --http --port 8765
```

For client integrations, see `synapse install <client>` and the onboarding docs.

## Key CLI commands

```bash
# onboarding
synapse onboard --flow quickstart
synapse onboard --flow advanced --non-interactive

# integrations & connectors
synapse integrations list --json
synapse integrations install claude
synapse integrations test claude
synapse integrations repair claude

# policy and runtime
synapse permit receipts --last 10 --json
synapse up --port 8765
synapse down
synapse status
synapse service install

# runtime memory operations (examples)
synapse ingest "User prefers concise responses"
synapse clip "Meeting notes: ..."
synapse watch --clipboard
synapse inbox list
synapse nlforget "forget my old job" --dry-run
synapse pack --topic project-x --range 30d
```

## Data model highlights (for API users)

- **Contextual recall** (`Synapse.recall`) supports multiple retrieval paths and query options.
- **Structured triples + graph indexing** are available for explainability and consistency checks.
- **Contradiction detection and belief lineage** expose confidence/conflicts for recall.
- **Scope policy** can enforce conservative output behavior at runtime.

## Python examples

```python
from synapse import Synapse

s = Synapse("~/.synapse/synapse_store")

# memory with scope
s.remember("Team deadline is Friday", scope="shared", shared_with=["team:ops"])

# compile context for downstream LLM call
pack = s.compile_context("Prepare a concise status update", budget=1800, policy="balanced")
system_prompt = pack.to_system_prompt()

# maintenance + safety
report = s.sleep(verbose=True)
print(report.to_digest())
```

## Compatibility and links

- **Docs:** this README and `README_MCP.md`
- **Examples:** `examples/`, `docs/`, `integrations/`
- **Tests:** `tests/` and `docs/benchmarks/`
- **Installable on:** macOS / Linux (and compatible Python 3.10+)

## Security and licensing

See [SECURITY.md](SECURITY.md). Synapse remains MIT-licensed under [LICENSE](LICENSE).
