#!/usr/bin/env python3
"""Prepare MCP registry submission payload and validate local manifest state.

Usage examples:
  python3 scripts/publish_mcp_registry.py
  python3 scripts/publish_mcp_registry.py --dry-run
  python3 scripts/publish_mcp_registry.py --out /tmp/synapse-mcp.registry.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.11+ in normal environments
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent

PYPROJECT_PATH = ROOT / "pyproject.toml"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as f:
        return str(tomllib.load(f)["project"]["version"])


def _load_project_name(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as f:
        return str(tomllib.load(f)["project"]["name"])


def _load_entrypoint(pyproject_path: Path, entrypoint: str) -> list[str]:
    with pyproject_path.open("rb") as f:
        scripts = tomllib.load(f)["project"].get("scripts", {})
    target = scripts.get(entrypoint)
    if not target:
        return []
    return [part.strip() for part in target.split(":", 1)]


def _is_valid_semver(text: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", text))


def _parse_range_min(expr: str) -> str | None:
    m = re.match(r"^\s*>?=\s*(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\s*$", expr)
    return m.group(1) if m else None


def _fail(msg: str, *, fatal: bool = False, results: list[str]) -> None:
    results.append(msg)
    if fatal:
        raise RuntimeError(msg)


def validate(manifest: dict[str, Any], pyproject_path: Path) -> list[str]:
    failures: list[str] = []

    required = [
        "$schema",
        "version",
        "name",
        "display_name",
        "description",
        "homepage",
        "repository",
        "license",
        "keywords",
        "version_range",
        "runtime",
        "install",
        "docs_url",
        "support_url",
    ]
    for key in required:
        if key not in manifest:
            failures.append(f"Missing required field: {key}")

    if failures:
        return failures

    # Basic schema and type checks
    if manifest["$schema"] != "https://modelcontextprotocol.io/schemas/registry-manifest.json":
        failures.append("$schema is not the MCP registry schema URL")

    if manifest["license"] != "MIT":
        failures.append("license should match current project license (MIT)")

    if manifest.get("description") == "":
        failures.append("description should not be empty")

    if not isinstance(manifest.get("keywords"), list) or not manifest["keywords"]:
        failures.append("keywords must be a non-empty list")

    # Runtime checks
    runtime = manifest["runtime"]
    entrypoint = runtime.get("entrypoint") if isinstance(runtime, dict) else None
    if not isinstance(entrypoint, dict):
        failures.append("runtime.entrypoint is required and must be an object")
    else:
        if entrypoint.get("command") != "synapse-mcp":
            failures.append("runtime.entrypoint.command should be synapse-mcp")

    runtime_env = runtime.get("environment") if isinstance(runtime, dict) else None
    if runtime_env is not None and not isinstance(runtime_env, dict):
        failures.append("runtime.environment must be an object when provided")

    # Installation checks
    install = manifest.get("install", {})
    if not all(k in install for k in ("linux", "darwin", "windows")):
        failures.append("install must include linux/darwin/windows entries")

    py_version = _load_project_version(pyproject_path)
    pkg_name = _load_project_name(pyproject_path)
    expected_pip = f"pip install {pkg_name}=={py_version}"
    expected_win = f"python -m pip install {pkg_name}=={py_version}"

    if install.get("linux", {}).get("command") != expected_pip:
        failures.append(f"install.linux.command should be '{expected_pip}'")

    if install.get("darwin", {}).get("command") != expected_pip:
        failures.append(f"install.darwin.command should be '{expected_pip}'")

    if install.get("windows", {}).get("command") != expected_win:
        failures.append(f"install.windows.command should be '{expected_win}'")

    if install.get("linux", {}).get("method") != "pip":
        failures.append("install.linux.method should be pip")
    if install.get("darwin", {}).get("method") != "pip":
        failures.append("install.darwin.method should be pip")
    if install.get("windows", {}).get("method") != "pip":
        failures.append("install.windows.method should be pip")

    # Sanity-check current package version compatibility window
    min_version = _parse_range_min(manifest.get("version_range", ""))
    if not min_version:
        failures.append("version_range should include a lower-bound like '>=0.12.4'")
    elif min_version != py_version:
        failures.append(
            f"version_range lower-bound ({min_version}) does not match pyproject version ({py_version})"
        )

    if not _is_valid_semver(py_version):
        failures.append(f"pyproject version is not semver-like: {py_version}")

    # Entrypoint script target should exist in package scripts
    ep = _load_entrypoint(pyproject_path, "synapse-mcp")
    if not ep:
        failures.append("pyproject[scripts].synapse-mcp must be defined")

    # Docs/support URL existence checks are intentionally light-touch; we validate only shape here.
    for field in ("homepage", "docs_url", "support_url"):
        if not isinstance(manifest.get(field), str) or "https://" not in manifest[field]:
            failures.append(f"{field} should be HTTPS URL")

    return failures


def print_report(
    manifest: dict[str, Any],
    failures: list[str],
    pyproject_path: Path,
    dry_run: bool,
    output: Path | None,
) -> int:
    if failures:
        print("MCP registry publish prep: FAILED")
        for item in failures:
            print(f" - {item}")
        print("\nPlease fix manifest issues before continuing.")
        return 1

    print("MCP registry publish prep: validation passed")
    print("- Product/version alignment: OK")
    print(
        f"- Installed package target: {_load_project_name(pyproject_path)}=={_load_project_version(pyproject_path)}"
    )

    payload = json.dumps(manifest, indent=2, sort_keys=True)
    print("\nFinal payload:\n")
    print(payload)

    if output:
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nPayload written to: {output}")

    print("\nDry-run mode:", "enabled" if dry_run else "disabled")
    if dry_run:
        print("No external submission was attempted (non-destructive).")
        print("Next step: copy the payload above into MCP Registry submit flow and continue with manual approval.")
    else:
        print("Manual final step required:")
        print("1) Open the MCP Registry submission portal and start a new package submission.")
        print("2) Paste/attach the payload shown above and submit for review.")
        print("3) Wait for maintainer approval and verify package visibility in MCP host clients.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and prepare MCP registry manifest for publication")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "docs/mcp-registry/synapse-mcp.registry.json"),
        help="Path to MCP registry JSON manifest",
    )
    parser.add_argument(
        "--pyproject",
        default=str(ROOT / "pyproject.toml"),
        help="Path to pyproject.toml used for validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare payload only; do not perform external submission",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Write the final payload to this file after validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    pyproject_path = Path(args.pyproject)
    out_path = Path(args.out) if args.out else None

    for path in (manifest_path, pyproject_path):
        if not path.exists():
            print(f"Missing required file: {path}")
            return 1

    try:
        manifest = _load_manifest(manifest_path)
        failures = validate(manifest, pyproject_path)
    except Exception as exc:  # pragma: no cover - surfaced explicitly to user
        print(f"Failed to load manifest: {exc}")
        return 1

    return print_report(
        manifest,
        failures,
        pyproject_path=pyproject_path,
        dry_run=args.dry_run,
        output=out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
