import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import cli


class TestCliIntegrationsPermit(unittest.TestCase):
    def _capture(self, fn, *args, **kwargs):
        out = io.StringIO()
        with redirect_stdout(out):
            fn(*args, **kwargs)
        return out.getvalue()

    def test_integrations_list_human_output(self):
        with patch.object(cli, "_detect_client_installs", return_value=[
            ("claude", "Claude Desktop", True, True),
            ("cursor", "Cursor", True, False),
            ("windsurf", "Windsurf", False, False),
            ("continue", "Continue", False, False),
            ("openclaw", "OpenClaw", True, False),
        ]):
            out = self._capture(cli.cmd_integrations, argparse.Namespace(integrations_action="list", db="/tmp/db", json=False))

        self.assertIn("Synapse integrations", out)
        self.assertIn("Claude Desktop", out)
        self.assertIn("healthy", out)
        self.assertIn("detected/not configured", out)

    def test_integrations_list_json_output(self):
        with patch.object(cli, "_detect_client_installs", return_value=[
            ("claude", "Claude Desktop", True, True),
            ("cursor", "Cursor", False, False),
            ("windsurf", "Windsurf", False, False),
            ("continue", "Continue", False, False),
            ("openclaw", "OpenClaw", False, False),
        ]):
            out = self._capture(cli.cmd_integrations, argparse.Namespace(integrations_action="list", db="/tmp/db", json=True))

        payload = json.loads(out)
        self.assertEqual(payload["db_path"], "/tmp/db")
        self.assertEqual(payload["integrations"][0]["name"], "claude")

    def test_integrations_test_dispatches_verify_only(self):
        fake_installer = {"claude": lambda db, dry_run=False, verify_only=False: verify_only}
        with patch("installer.ClientInstaller.ENHANCED_TARGETS", fake_installer):
            cli.cmd_integrations(argparse.Namespace(integrations_action="test", name="claude", db="/tmp/db"))

    def test_integrations_open_uses_browser(self):
        with tempfile.TemporaryDirectory(prefix="synapse-open-") as tmpdir:
            cfg = os.path.join(tmpdir, "claude.json")
            with patch.object(cli, "_integration_open_target", return_value=("file", cfg)), patch("webbrowser.open", return_value=True) as open_mock:
                out = self._capture(cli.cmd_integrations, argparse.Namespace(integrations_action="open", name="claude", db="/tmp/db"))

            self.assertIn("Claude Desktop", out)
            open_mock.assert_called_once()

    def test_permit_receipts_json_empty(self):
        with patch.object(cli, "_load_policy_receipts", return_value=([], "")):
            out = self._capture(cli.cmd_permit, argparse.Namespace(permit_action="receipts", last=3, db="/tmp/db", json=True))

        payload = json.loads(out)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["receipts"], [])

    def test_permit_receipts_human_with_data(self):
        receipts = [{
            "receipt_id": "rec_123",
            "decision": "allow",
            "actor_id": "cursor-agent",
            "app_id": "cursor",
            "purpose": "code_assistance",
            "scope_requested": "shared",
            "scope_applied": "shared",
            "policy_id": "coding-agent-strict.v1",
            "matched_rules": ["rule.4"],
            "memory_counts": {"considered": 27, "returned": 5, "blocked": 3},
            "block_reasons": ["sensitive_memory_private_only"],
            "timestamp": "2026-02-17T09:31:22Z",
        }]
        with patch.object(cli, "_load_policy_receipts", return_value=(receipts, "/tmp/permit_receipts.jsonl")):
            out = self._capture(cli.cmd_permit, argparse.Namespace(permit_action="receipts", last=1, db="/tmp/db", json=False))

        self.assertIn("RECEIPT rec_123", out)
        self.assertIn("Decision: ALLOW", out)
        self.assertIn("Policy matched: coding-agent-strict.v1.rule.4", out)

    def test_main_dispatches_integrations_and_permit(self):
        with patch.object(cli, "cmd_integrations") as cmd_integrations:
            with patch.object(sys, "argv", ["synapse", "integrations", "list"]):
                cli.main()
            self.assertTrue(cmd_integrations.called)

        with patch.object(cli, "cmd_permit") as cmd_permit:
            with patch.object(sys, "argv", ["synapse", "permit", "receipts"]):
                cli.main()
            self.assertTrue(cmd_permit.called)


if __name__ == "__main__":
    unittest.main()
