import argparse
import io
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
        )

        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
            ("windsurf", "Windsurf", False, False),
        ]

        fake_installer = MagicMock(return_value=True)

        with patch.object(cli, "_detect_client_installs", return_value=detect_result), \
             patch("installer.ClientInstaller.ENHANCED_TARGETS", {"claude": fake_installer}), \
             patch("builtins.input", side_effect=AssertionError("input should not run")), \
             patch.object(cli, "_run_onboard_probe", return_value=(True, {"start_count": "1", "end_count": "2"})):
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


if __name__ == "__main__":
    unittest.main()
