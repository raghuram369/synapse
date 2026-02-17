import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import cli


def _run_cmd(cmd_fn, args):
    out = io.StringIO()
    with redirect_stdout(out):
        cmd_fn(args)
    return out.getvalue()


class TestOnboardBootstrap(unittest.TestCase):
    def test_onboard_non_interactive_auto_selects_pending_clients(self):
        args = argparse.Namespace(
            db="/tmp/synapse_db",
            flow="quickstart",
            non_interactive=True,
            json=False,
            policy_template=None,
            default_scope=None,
            default_sensitive=None,
            service=False,
            service_schedule="daily",
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
            ("windsurf", "Windsurf", False, False),
        ]

        fake_installer = MagicMock(return_value=True)

        with patch.object(cli, "_detect_client_installs", return_value=detect_result):
            with patch("installer.ClientInstaller.ENHANCED_TARGETS", {"claude": fake_installer}):
                with patch("builtins.input", side_effect=AssertionError("input should not run")):
                    with patch.object(
                        cli,
                        "_run_onboard_probe",
                        return_value=(True, {"start_count": "1", "end_count": "2"}),
                    ):
                        _run_cmd(cli.cmd_onboard, args)

        fake_installer.assert_called_once_with("/tmp/synapse_db", dry_run=False, verify_only=False)

    def test_doctor_non_interactive_exits_on_error(self):
        import doctor

        fake_checks = [
            {"name": "Synapse", "status": "ok", "message": "ok"},
            {"name": "Runtime", "status": "error", "message": "shadowed stdlib module"},
        ]

        args = argparse.Namespace(db="/tmp/store", json=False, fix=False, non_interactive=True)

        with patch.object(doctor, "gather_checks", return_value=fake_checks):
            with self.assertRaises(SystemExit) as exc:
                doctor.run_doctor(args)
            self.assertEqual(exc.exception.code, 2)

    def test_onboard_failure_has_one_command_fix(self):
        args = argparse.Namespace(db="/tmp/synapse_db", flow="quickstart", non_interactive=True, json=False)

        with patch.object(cli, "_detect_client_installs", return_value=[]):
            with patch.object(
                cli,
                "_run_onboard_probe",
                return_value=(False, {"error": "probe failed", "start_count": "0", "end_count": "0"}),
            ):
                with patch("builtins.input", side_effect=AssertionError("input should not run")):
                    output = _run_cmd(cli.cmd_onboard, args)

        self.assertIn("Fix steps:", output)

    def test_onboarding_smoke_json(self):
        args = argparse.Namespace(
            db="/tmp/synapse_smoke_store.db",
            flow="quickstart",
            non_interactive=True,
            json=True,
            policy_template=None,
            default_scope=None,
            default_sensitive=None,
            service=False,
            service_schedule="daily",
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
            ("openclaw", "OpenClaw", True, False),
        ]

        install_calls: list[tuple[str, bool, bool]] = []

        def _fake_installer(db_path: str, dry_run: bool = False, verify_only: bool = False):
            install_calls.append((db_path, dry_run, verify_only))
            return True

        with patch.object(cli, "_detect_client_installs", return_value=detect_result):
            with patch(
                "installer.ClientInstaller.ENHANCED_TARGETS",
                {"claude": _fake_installer, "openclaw": _fake_installer},
            ):
                with patch.object(
                    cli,
                    "_run_onboard_probe",
                    return_value=(True, {"start_count": "7", "end_count": "8"}),
                ):
                    with patch("builtins.input", side_effect=AssertionError("input should not run")):
                        output = _run_cmd(cli.cmd_onboard, args)

        payload = json.loads(output)
        self.assertEqual(payload["flow"], "quickstart")
        self.assertEqual(payload["selected_integrations"], ["claude", "openclaw"])
        self.assertTrue(payload["probe"]["passed"])
        self.assertIn("claude", [item["name"] for item in payload["configured"]])
        self.assertEqual(
            install_calls,
            [
                ("/tmp/synapse_smoke_store.db", False, False),
                ("/tmp/synapse_smoke_store.db", False, False),
            ],
        )


class TestOnboardDefaultsFlow(unittest.TestCase):
    def test_onboard_non_interactive_policy_and_defaults_in_json(self):
        args = argparse.Namespace(
            db="/tmp/synapse_db",
            flow="advanced",
            non_interactive=True,
            json=True,
            policy_template="work",
            default_scope="public",
            default_sensitive="off",
            service=False,
            service_schedule="daily",
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
        ]

        fake_installer = MagicMock(return_value=True)

        with patch.object(cli, "_detect_client_installs", return_value=detect_result), \
                patch.object(cli, "_write_onboard_defaults", return_value=None), \
                patch("installer.ClientInstaller.ENHANCED_TARGETS", {"claude": fake_installer}), \
                patch.object(cli, "_run_onboard_probe", return_value=(True, {"start_count": "1", "end_count": "2"})):
            out = _run_cmd(cli.cmd_onboard, args)

        payload = json.loads(out)
        self.assertEqual(payload["policy"], "work")
        self.assertEqual(payload["storage"]["default_scope"], "public")
        self.assertFalse(payload["storage"]["default_sensitive"])
        self.assertEqual(payload["selected_integrations"], ["claude"])

    def test_onboard_interactive_integration_prompt_selection(self):
        args = argparse.Namespace(
            db="/tmp/synapse_db",
            flow="advanced",
            non_interactive=False,
            json=False,
            policy_template=None,
            default_scope=None,
            default_sensitive=None,
            service=True,
            service_schedule="daily",
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
        ]

        fake_installer = MagicMock(return_value=True)

        with patch.object(cli, "_detect_client_installs", return_value=detect_result), \
                patch("installer.ClientInstaller.ENHANCED_TARGETS", {"claude": fake_installer, "cursor": fake_installer}), \
                patch("service.install_service", return_value="/tmp/synapse_service.plist"), \
                patch.object(cli, "_run_onboard_probe", return_value=(True, {"start_count": "1", "end_count": "2"}),) , \
                patch("builtins.input", side_effect=["2", "1", "", "", "", ""]):
            out = _run_cmd(cli.cmd_onboard, args)

        fake_installer.assert_called_once()
        self.assertIn("Step 7", out)

    def test_onboard_runtime_conflict_triggers_probe_retest(self):
        args = argparse.Namespace(
            db="/tmp/synapse_db",
            flow="advanced",
            non_interactive=False,
            json=True,
            policy_template="private",
            default_scope=None,
            default_sensitive=None,
            service=False,
            service_schedule="daily",
        )

        detect_result = [
            ("claude", "Claude Desktop", False, False),
        ]

        with patch.object(cli, "_detect_client_installs", return_value=detect_result), \
                patch.object(cli, "_write_onboard_defaults", return_value=None), \
                patch.object(
                    cli,
                    "_run_onboard_probe",
                    side_effect=[
                        (False, {"error": "probe failed"}),
                        (True, {"start_count": "1", "end_count": "2"}),
                    ],
                ), \
                patch.object(cli, "_maybe_repair_runtime_conflict", return_value=True), \
                patch("builtins.input", side_effect=["", "", "", ""]):
            out = _run_cmd(cli.cmd_onboard, args)

        payload = json.loads(out)
        self.assertEqual(payload["probe"]["passed"], True)
        self.assertTrue(payload["repair_attempted"])
        self.assertEqual(payload["probe"]["details"].get("start_count"), "1")


if __name__ == "__main__":
    unittest.main()
