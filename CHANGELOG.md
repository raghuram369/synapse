# Changelog

## Unreleased

### Added
- Added public safety/utility benchmark card and machine-readable reporting template at `docs/benchmarks/safety-utility-benchmark-card.md` and `docs/benchmarks/safety-utility-benchmark-metadata.json`.

### Changed
- _No entries yet._

### Fixed
- _No entries yet._

### Breaking Integration Changes

> Use this section when changing integration contracts, config schemas, installation payloads, or MCP host compatibility.
> Include migration notes + rollback command if behavior changes for existing users.

- **Template entry (copy/paste):**
  - **[{version}]** `[{date}]` — `integration-surface` — `breaking-change-title`
    - **Impact:** Which integrations and user flows are affected.
    - **Before:** Prior behavior / schema.
    - **After:** New behavior / schema.
    - **Migration:** How to migrate existing installs (commands, flag changes, config edits).
    - **Rollback:** How to revert safely (if possible).

- **Example:** `- [0.12.5] [2026-02-16] — MCP installation contract`
  - **Impact:** CLI and MCP installer behaviors.
  - **Before:** (fill)
  - **After:** (fill)
  - **Migration:** (fill)
  - **Rollback:** `synapse integrations repair <name>`
