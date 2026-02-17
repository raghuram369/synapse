# Changelog

## 0.13.0 - 2026-02-16

### Added
- Added onboarding enhancement: `synapse onboard` now supports quickstart + advanced flows with explicit `--non-interactive` and `--json` modes for scripted usage.
- Added runtime bootstrap options in onboarding (`--policy-template`, `--default-scope`, `--default-sensitive`, `--enable-service`, `--service-schedule`).
- Added integration contract metadata exposure via `synapse integrations list --json`, including connector type/tier/commands/capabilities and diagnostic checks.
- Added policy receipt inspection entrypoint (`synapse permit receipts`) with machine-readable output and human-readable formatting.
- Documented MCP launch/runtime behavior in a refreshed, launch-ready README and synchronized MCP-specific README.

### Changed
- Updated README and runtime docs to reflect the current onboarding/integrations/receipts implementation.
- Align documentation of MCP exposure around supported command modes and installable clients (`claude`, `cursor`, `windsurf`, `continue`, `openclaw`).
- Package metadata version bumped from 0.12.4 to 0.13.0 for release.

### Fixed
- Fixed consistency gaps between user-facing onboarding and integration claims in documentation.
- Clarified policy receipt availability caveat to avoid presenting it as fully complete before logging is enabled.

### Breaking Integration Changes

> Use this section when changing integration contracts, config schemas, installation payloads, or MCP host compatibility.
> Include migration notes + rollback command if behavior changes for existing users.

- **[0.13.0] [2026-02-16]** — onboarding/runtime doc surface and permit-receipt reporting
  - **Impact:** CLI onboarding + integration docs now reflect current runtime flows; no API-level compatibility break required for existing installs.
  - **Before:** README/README_MCP referenced mixed claims about onboarding, tool surfaces, and receipt coverage.
  - **After:** Release documentation now documents supported/explicit flows and marks scaffolded receipts behavior clearly.
  - **Migration:** No migration required.
  - **Rollback:** Re-run onboarding/installers with older docs if needed; existing data paths are unchanged.

## Unreleased

### Added
- _No entries yet._

### Changed
- _No entries yet._

### Fixed
- _No entries yet._
