# MCP Registry Packaging & Publish Guide

This folder contains the metadata artifact used for MCP Registry submission.

## Artifact(s)

- `synapse-mcp.registry.json` — canonical registry metadata for MCP distribution channels.

## Packaging notes

1. Keep this artifact in source control so registry metadata and release version updates are auditable.
2. Update the `version_range`, `version` pins, and package URL fields whenever `synapse-ai-memory` is released.
3. Confirm `entrypoint.command` points to the stable wrapper command (`synapse-mcp`), not the raw `mcp_server.py` script.

## Validate before publish (local)

```bash
cat docs/mcp-registry/synapse-mcp.registry.json | jq .
python -m json.tool docs/mcp-registry/synapse-mcp.registry.json >/tmp/manifest.pretty.json
```

## Publishing workflow (planned)

1. Publish a release of `synapse-ai-memory` first (PyPI + signed package artifacts).
2. Open MCP Registry submission PR with the exact JSON payload in this folder.
3. Confirm the submission references:
   - repository: `https://github.com/raghuram369/synapse`
   - docs: `README_MCP.md`
   - support: issue tracker
4. Verify registry ingestion from client installs and `synapse install ...` flows.

## External action required

Actual registry publishing is external to this repo. Remaining steps are:

- authenticate to the MCP Registry maintainer flow
- submit/approve `synapse-mcp.registry.json`
- validate visibility from at least one MCP host after approval
