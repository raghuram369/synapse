# MCP Registry Packaging & Publish Guide

This folder contains the metadata artifact used for MCP Registry submission.

## Artifact(s)

- `synapse-mcp.registry.json` — canonical registry metadata for MCP distribution channels.

## Packaging notes

1. Keep this artifact in source control so registry metadata and release version updates are auditable.
2. Update the `version_range`, `version` pins, and package URL fields whenever `synapse-ai-memory` is released.
3. Confirm `entrypoint.command` points to the stable wrapper command (`synapse-mcp`), not the raw `mcp_server.py` script.

## Publish helper

Use this local helper before opening the external MCP Registry portal:

```bash
python3 scripts/publish_mcp_registry.py --dry-run
```

This performs non-destructive validation and prints the final payload to stdout.

Optional output artifact for review/re-use:

```bash
python3 scripts/publish_mcp_registry.py --dry-run --out /tmp/synapse-mcp.registry.payload.json
```

Checks performed:

- Registry schema URL + required keys
- Name/license/description consistency with project metadata
- Entrypoint command exists and aligns with package scripts (`synapse-mcp`)
- Install commands align to current package version from `pyproject.toml`
- Version compatibility window (`version_range`) aligns with current package version
- URL fields are valid HTTPS targets

## Manual submission steps (external portal action)

Actual registry publishing remains external to this repo. When the helper passes:

1. Open the MCP Registry submission portal.
2. Submit `synapse-mcp.registry.json` (or the generated payload copy).
3. Continue through maintainer review/approval.
4. Verify visibility from at least one MCP host.

Exact portal endpoint may vary by org process; use the configured registry maintainer flow.
