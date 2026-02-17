import io
import os
import sys
import tempfile
import unittest
import json
from contextlib import redirect_stdout
from unittest.mock import patch

import cli


class TestOnboardingSmokeEndToEnd(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory(prefix="synapse-onboard-smoke-")
        self.db_path = os.path.join(self.workdir.name, "synapse_store")

    def tearDown(self):
        self.workdir.cleanup()

    def _run_main(self, argv):
        with patch.object(sys, "argv", ["synapse"] + argv):
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main()
        return out.getvalue()

    def test_onboarding_smoke_quickstart_json_path(self):
        detect_result = [
            ("claude", "Claude Desktop", True, False),
            ("cursor", "Cursor", True, True),
            ("openclaw", "OpenClaw", True, False),
        ]
        configured: list[tuple[str, bool, bool]] = []

        def _fake_installer(db_path: str, dry_run: bool = False, verify_only: bool = False) -> bool:
            configured.append((db_path, dry_run, verify_only))
            return True

        with patch.object(
            cli,
            "_onboard_state_path",
            return_value=os.path.join(self.workdir.name, "onboard_defaults.json"),
        ):
            with patch.object(
                cli,
                "_detect_client_installs",
                return_value=detect_result,
            ):
                with patch(
                    "installer.ClientInstaller.ENHANCED_TARGETS",
                    {"claude": _fake_installer, "openclaw": _fake_installer},
                ):
                    with patch.object(
                        cli,
                        "_run_onboard_probe",
                        return_value=(True, {"start_count": "1", "end_count": "2"}),
                    ):
                        out = self._run_main([
                            "onboard",
                            "--flow",
                            "quickstart",
                            "--non-interactive",
                            "--json",
                            "--db",
                            self.db_path,
                        ])

        payload = json.loads(out)
        self.assertEqual(payload["flow"], "quickstart")
        self.assertTrue(payload["non_interactive"])
        self.assertTrue(payload["probe"]["passed"])
        self.assertEqual(payload["selected_integrations"], ["claude", "openclaw"])
        self.assertEqual(
            configured,
            [(self.db_path, False, False), (self.db_path, False, False)],
        )

    def test_onboarding_smoke_fix_hint(self):
        with patch.object(
            cli,
            "_onboard_state_path",
            return_value=os.path.join(self.workdir.name, "onboard_defaults.json"),
        ), patch.object(
            cli,
            "_detect_client_installs",
            return_value=[],
        ), patch.object(
            cli,
            "_run_onboard_probe",
            return_value=(False, {"error": "probe failed", "start_count": "0", "end_count": "0"}),
        ):
            out = self._run_main([
                "onboard",
                "--flow",
                "quickstart",
                "--non-interactive",
                "--db",
                self.db_path,
            ])

        self.assertIn("Fix steps:", out)
        self.assertIn("synapse doctor --fix", out)


if __name__ == "__main__":
    unittest.main()
