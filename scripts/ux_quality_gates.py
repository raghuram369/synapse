#!/usr/bin/env python3
"""UX quality-gate smoke checks for onboarding + integrations UX.

These are lightweight, offline, and scriptable checks intended to run in CI.
"""

from __future__ import annotations

import argparse
import argparse as argparse_module
import io
import json
import os
import tempfile
from pathlib import Path
import sys
from contextlib import redirect_stdout
from typing import Callable, Tuple
from unittest.mock import patch

# Ensure local workspace modules are imported when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli


GateResult = Tuple[bool, str]


def _capture_stdout(callable_obj: Callable[[], None]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        callable_obj()
    return buffer.getvalue()


def check_scriptable_onboarding_flow() -> GateResult:
    """Run a deterministic onboarding simulation and assert machine-readable output."""

    with tempfile.TemporaryDirectory(prefix="synapse-ux-gate-") as workdir:
        args = argparse.Namespace(
            db=os.path.join(workdir, "synapse-smoke.db"),
            flow="quickstart",
            non_interactive=True,
            json=True,
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
            ("openclaw", "OpenClaw", True, False),
        ]
        configured_calls: list[tuple[str, bool, bool]] = []

        def _fake_installer(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
            configured_calls.append((db_path, dry_run, verify_only))
            return True

        with patch.object(
            cli,
            "_onboard_state_path",
            return_value=os.path.join(workdir, "onboard_defaults.json"),
        ):
            with patch.object(cli, "_detect_client_installs", return_value=detect_result):
                with patch(
                    "installer.ClientInstaller.ENHANCED_TARGETS",
                    {"claude": _fake_installer, "openclaw": _fake_installer},
                ):
                    with patch.object(
                        cli,
                        "_run_onboard_probe",
                        return_value=(True, {"start_count": "3", "end_count": "4"}),
                    ):
                        output = _capture_stdout(lambda: cli.cmd_onboard(args))

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"Expected JSON output for scriptable onboarding, got: {exc}\n{output}"

    expected_clients = ["claude", "openclaw"]
    if payload.get("flow") != "quickstart":
        return False, "Flow did not report quickstart mode"
    if payload.get("non_interactive") is not True:
        return False, "non_interactive flag not reflected in JSON output"
    if payload.get("selected_integrations") != expected_clients:
        return False, f"Unexpected selected clients: {payload.get('selected_integrations')}"
    if payload.get("probe", {}).get("passed") is not True:
        return False, "Onboarding smoke probe did not report pass"
    expected_calls = [(args.db, False, False), (args.db, False, False)]
    if configured_calls != expected_calls:
        return False, f"Installer invocation mismatch: {configured_calls}"
    return True, "OK"


def check_one_command_fix_messaging() -> GateResult:
    """Failing onboarding should include a one-command remediation hint."""

    with tempfile.TemporaryDirectory(prefix="synapse-ux-gate-") as workdir:
        args = argparse.Namespace(
            db=os.path.join(workdir, "synapse-smoke.db"),
            flow="quickstart",
            non_interactive=True,
            json=False,
        )

        with patch.object(
            cli,
            "_onboard_state_path",
            return_value=os.path.join(workdir, "onboard_defaults.json"),
        ):
            with patch.object(cli, "_detect_client_installs", return_value=[]):
                with patch.object(
                    cli,
                    "_run_onboard_probe",
                    return_value=(False, {"error": "probe fail", "start_count": "0", "end_count": "0"}),
                ):
                    with patch("builtins.input", side_effect=AssertionError("input should not run")):
                        output = _capture_stdout(lambda: cli.cmd_onboard(args))

    if "synapse doctor --fix" not in output:
        return False, "Expected one-command fix hint 'synapse doctor --fix' in onboarding failure output"
    return True, "OK"


def check_json_flow_is_output_only_no_stdout_noise() -> GateResult:
    """Ensure JSON mode remains machine-readable with no ad-hoc stdout noise."""

    with tempfile.TemporaryDirectory(prefix="synapse-ux-gate-") as workdir:
        args = argparse.Namespace(
            db=os.path.join(workdir, "synapse-smoke.db"),
            flow="advanced",
            non_interactive=True,
            json=True,
            policy_template="private",
            default_scope="private",
            default_sensitive="on",
            service=True,
            service_schedule="daily",
        )

        with patch.object(
            cli,
            "_onboard_state_path",
            return_value=os.path.join(workdir, "onboard_defaults.json"),
        ):
            with patch.object(
                cli,
                "_detect_client_installs",
                return_value=[],
            ), \
                patch.object(cli, "_write_onboard_defaults", return_value=None), \
                patch("service.install_service", return_value="/tmp/synapse_service.plist"), \
                patch.object(
                    cli,
                    "_run_onboard_probe",
                    return_value=(True, {"start_count": "7", "end_count": "8"}),
                ), \
                patch.object(cli, "_maybe_repair_runtime_conflict", return_value=False):
                output = _capture_stdout(lambda: cli.cmd_onboard(args))

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"Expected pure JSON output with --json, parse failed: {exc}"

    service = payload.get("service", {})
    if service.get("requested") is not True:
        return False, "Service should be requested in this JSON-only smoke scenario"
    if service.get("installed") is not True:
        return False, "Service installation result missing in payload"
    if not output.strip().startswith('{'):
        return False, "Output is not JSON-only"

    return True, "OK"


def check_no_manual_json_edits_for_top_integrations() -> GateResult:
    """Top integrations should stay in scriptable command path (no manual file-edit guidance)."""

    with tempfile.TemporaryDirectory(prefix="synapse-ux-gate-") as workdir:
        args = argparse.Namespace(integrations_action="list", db=os.path.join(workdir, "synapse-store"), json=False)

        detect_result = [
            ("claude", "Claude Desktop", True, True),
            ("cursor", "Cursor", True, False),
            ("windsurf", "Windsurf", False, False),
            ("continue", "Continue", True, False),
            ("openclaw", "OpenClaw", True, False),
        ]

        with patch.object(cli, "_detect_client_installs", return_value=detect_result):
            output = _capture_stdout(lambda: cli.cmd_integrations(args))

    lowered = output.lower()
    forbidden = [
        "edit this file",
        "manual edit",
        "manually edit",
        "open the file",
        "copy this file",
    ]
    for needle in forbidden:
        if needle in lowered:
            return False, f"Found forbidden guidance for manual JSON edits: '{needle}'"

    required = {"claude", "cursor", "windsurf", "continue", "openclaw"}
    for key in required:
        if key not in lowered:
            return False, f"Missing integration in list output: {key}"

    return True, "OK"


def run_all() -> int:
    checks = [
        ("scriptable_onboarding_flow", check_scriptable_onboarding_flow),
        ("one_command_fix_messaging", check_one_command_fix_messaging),
        ("json_flow_no_stdout_noise", check_json_flow_is_output_only_no_stdout_noise),
        ("no_manual_json_edits_for_top_integrations", check_no_manual_json_edits_for_top_integrations),
    ]

    failures: list[str] = []
    for name, fn in checks:
        ok, detail = fn()
        status = "pass" if ok else "fail"
        print(f"[{status}] {name}")
        if not ok:
            failures.append(f"[{name}] {detail}")

    if failures:
        print("\nQuality gate failures:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("\nUX quality gates: all pass")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse_module.ArgumentParser(description="Run UX quality gates")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    args = parser.parse_args(argv)

    if args.json:
        payload = {
            "scriptable_onboarding_flow": check_scriptable_onboarding_flow()[0],
            "one_command_fix_messaging": check_one_command_fix_messaging()[0],
            "json_flow_no_stdout_noise": check_json_flow_is_output_only_no_stdout_noise()[0],
            "no_manual_json_edits_for_top_integrations": check_no_manual_json_edits_for_top_integrations()[0],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if all(payload.values()) else 1

    return run_all()


if __name__ == "__main__":
    sys.exit(main())
