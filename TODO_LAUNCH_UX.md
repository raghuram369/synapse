# TODO — Synapse Launch UX + Connector Ecosystem

## Source
- `docs/SYNAPSE_LAUNCH_UX_CONNECTOR_SPEC.md`

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

## Install + Runtime
- [x] Add one-liner bootstrap flow (`install.sh`) for runtime + app + onboarding
- [x] Support optional Node bootstrap (`npx @synapse-memory/setup@latest`)
- [x] Ensure managed runtime under `~/.synapse/runtime`
- [x] Remove dependency on user Python for MCP startup
- [x] Add post-install automatic health checks

## Onboarding Wizard
- [x] Implement `synapse onboard --flow quickstart|advanced`
- [x] Add policy template selection step
- [x] Add integration selection step
- [x] Add storage configuration step
- [x] Add background service enablement step
- [x] Add live remember/recall verification step
- [x] Add fail-safe repair flow with rerun prompt

## Integrations Hub
- [x] Implement `synapse integrations` summary view
- [x] Implement `synapse integrations install <name>`
- [x] Implement `synapse integrations test <name>`
- [x] Implement `synapse integrations repair <name>`
- [x] Implement `synapse integrations open <name>`

## Doctor + Auto-fix
- [x] Implement `synapse doctor --fix` for env/runtime conflicts
- [x] Detect shadowed stdlib packages (e.g., asyncio)
- [x] Auto-heal by switching MCP launch to managed runtime
- [x] Print exact remediation steps on failure

## Policy Receipts UX
- [x] Implement `synapse permit receipts --last 3`
- [x] Add human-friendly receipt output format
- [x] Add JSON receipt output format
- [ ] Add explainable allow/deny receipts for memory reads/writes

## Memory Control Center (Inspector UX)
- [ ] Add overview cards (total, pending review, blocked, integrations healthy)
- [ ] Add memory browser filters + edit/redact/delete/scope actions
- [ ] Add policy center (template, rule toggles, simulate policy)
- [ ] Add receipts explorer with searchable decision traces
- [ ] Add portability panel (export/import preview + policy diff + signature verify)

## Connector Ecosystem
- [ ] Define connector metadata schema + validation
- [ ] Implement connector contract surface: install/verify/test/repair/uninstall/doctor_checks/example_prompt
- [ ] Tier 0: first-class support quality bar (Claude, Cursor, OpenAI, LangGraph, LangChain)
- [ ] Tier 1 launch-window support (Windsurf, Continue, CrewAI, OpenClaw)
- [ ] Tier 2 roadmap stubs (AutoGen, Google ADK, LlamaIndex)

## Ecosystem Assets + Distribution
- [ ] Publish MCP server metadata to MCP Registry
- [ ] Create starter repos (openai/langgraph/claude-mcp)
- [ ] Add compatibility matrix (OS/runtime/limitations)
- [ ] Add changelog section for breaking integration changes
- [ ] Add public benchmark safety/utility card
- [ ] Add installer docs for CI/non-interactive mode
- [ ] Add Python + TypeScript onboarding examples
- [ ] Add deep-link install pages for major clients

## Docs + QA
- [x] Keep README and README_MCP aligned with onboarding/runtime behavior
- [x] Add/update tests for installer payloads and integration repair flows
- [x] Add end-to-end onboarding smoke test to CI
- [x] Add UX quality-gate checklist automation skeleton (scriptable flows, one-command fixes, no manual JSON edits for top integrations path)
- [ ] Add explainable denials to quality checks
