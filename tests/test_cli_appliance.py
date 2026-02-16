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


class TestCliApplianceCommands(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="synapse-cli-appliance-")
        self.db = os.path.join(self.tmpdir.name, "synapse_store")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _capture(self, fn, *args, **kwargs):
        out = io.StringIO()
        with redirect_stdout(out):
            fn(*args, **kwargs)
        return out.getvalue()

    def _run_main(self, argv):
        with patch.object(sys, "argv", ["synapse"] + argv):
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main()
        return out.getvalue()

    def test_serve_prints_banner_and_tool_list_in_stdio_mode(self):
        tools = [{"name": "remember", "description": "Store a memory", "inputSchema": {}}]
        with patch.object(cli, "_collect_mcp_tools", return_value=tools) as collect_tools:
            with patch.object(cli, "_run_mcp_stdio_server") as run_stdio:
                output = self._capture(
                    cli.cmd_serve,
                    argparse.Namespace(db=self.db, http=False, port=9999),
                )

        self.assertEqual(output.strip().splitlines()[0], cli.APPLIANCE_BANNER)
        self.assertIn("MCP tools:", output)
        self.assertIn("remember", output)
        collect_tools.assert_called_once_with(self.db)
        run_stdio.assert_called_once_with(db_path=self.db)

    def test_serve_http_mode_starts_http_transport(self):
        tools = [{"name": "remember", "description": "Store a memory", "inputSchema": {}}]
        with patch.object(cli, "_collect_mcp_tools", return_value=tools) as collect_tools:
            with patch.object(cli, "_run_mcp_http_server") as run_http:
                output = self._capture(
                    cli.cmd_serve,
                    argparse.Namespace(db=self.db, http=True, port=9101),
                )

        self.assertIn("HTTP JSON-RPC on localhost:9101", output)
        collect_tools.assert_called_once_with(self.db)
        run_http.assert_called_once_with(db_path=self.db, port=9101)

    def test_inspect_text_output_shows_tool_catalog_and_store_summary(self):
        with patch.object(cli, "_collect_mcp_tools", return_value=[
            {"name": "remember", "description": "Store", "inputSchema": {"type": "object"}}
        ]):
            with patch.object(cli, "_collect_store_snapshot", return_value={
                "memory_count": 3,
                "concept_count": 4,
                "edge_count": 2,
                "belief_count": 1,
                "contradiction_count": 0,
                "last_sleep_at": None,
                "top_hot_concepts": [("task", 0.9)],
            }):
                output = self._capture(
                    cli.cmd_inspect,
                    argparse.Namespace(db=self.db, json=False),
                )

        self.assertIn("🧰 MCP tool catalog", output)
        self.assertIn("remember", output)
        self.assertIn("📦 Store summary", output)
        self.assertIn("Memories: 3", output)
        self.assertIn("Contradictions: 0", output)
        self.assertIn("- Last sleep: never", output)

    def test_inspect_json_output(self):
        snapshot = {
            "memory_count": 3,
            "concept_count": 4,
            "edge_count": 2,
            "belief_count": 1,
            "contradiction_count": 0,
            "last_sleep_at": None,
            "top_hot_concepts": [("task", 0.9)],
        }
        with patch.object(cli, "_collect_mcp_tools", return_value=[
            {"name": "recall", "description": "Recall", "inputSchema": {"type": "object"}},
        ]):
            with patch.object(cli, "_collect_store_snapshot", return_value=snapshot):
                output = self._capture(
                    cli.cmd_inspect,
                    argparse.Namespace(db=self.db, json=True),
                )

        payload = json.loads(output)
        self.assertEqual(payload["store_path"], self.db)
        self.assertEqual(payload["store"], {
            "memory_count": 3,
            "concept_count": 4,
            "edge_count": 2,
            "belief_count": 1,
            "contradictions": 0,
            "last_sleep_at": None,
            "top_hot_concepts": [["task", 0.9]],
        })
        self.assertEqual(payload["tools"], [{"name": "recall", "description": "Recall", "inputSchema": {"type": "object"}}])

    def test_doctor_reports_healthy_store_as_success(self):
        with patch.object(cli, "_check_store_files", return_value=[
            ("storage directory", "pass", "ok"),
            ("./synapse_store.log", "pass", "read/write"),
            ("./synapse_store.snapshot", "pass", "read/write"),
            ("snapshot json", "pass", "valid JSON"),
        ]), patch.object(
            cli,
            "_collect_store_snapshot",
            return_value={
                "memory_count": 2,
                "concept_count": 1,
                "edge_count": 0,
                "belief_count": 0,
                "contradiction_count": 0,
                "last_sleep_at": 0,
                "top_hot_concepts": [],
            },
        ), patch.object(cli, "_scan_portable_exports", return_value=[]), patch.object(
            cli,
            "_run_store_latency_probe",
            return_value=120.5,
        ):
            output = self._capture(
                cli.cmd_doctor,
                argparse.Namespace(db=self.db),
            )

        self.assertIn("Health check complete", output)
        self.assertIn("[PASS] Store counts", output)

    def test_doctor_exits_on_crc_failure(self):
        with patch.object(cli, "_check_store_files", return_value=[]), patch.object(
            cli,
            "_collect_store_snapshot",
            return_value={
                "memory_count": 2,
                "concept_count": 1,
                "edge_count": 0,
                "belief_count": 0,
                "contradiction_count": 0,
                "last_sleep_at": 0,
                "top_hot_concepts": [],
            },
        ), patch.object(cli, "_scan_portable_exports", return_value=[
            ("bad.synapse", False, "CRC invalid"),
        ]), patch.object(cli, "_run_store_latency_probe", return_value=50.0):
            out = io.StringIO()
            with self.assertRaises(SystemExit) as exc:
                with redirect_stdout(out):
                    cli.cmd_doctor(argparse.Namespace(db=self.db))
            output = out.getvalue()

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Portable: bad.synapse", output)

    def test_doctor_exits_on_performance_warning(self):
        with patch.object(cli, "_check_store_files", return_value=[("storage directory", "pass", "ok")]), patch.object(
            cli, "_collect_store_snapshot",
            return_value={
                "memory_count": 2,
                "concept_count": 1,
                "edge_count": 0,
                "belief_count": 0,
                "contradiction_count": 0,
                "last_sleep_at": 0,
                "top_hot_concepts": [],
            },
        ), patch.object(cli, "_scan_portable_exports", return_value=[]), patch.object(
            cli, "_run_store_latency_probe", return_value=750.5
        ):
            out = io.StringIO()
            with self.assertRaises(SystemExit) as exc:
                with redirect_stdout(out):
                    cli.cmd_doctor(argparse.Namespace(db=self.db))
            output = out.getvalue()

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Performance", output)

    def test_main_dispatches_new_serve_command(self):
        with patch.object(cli, "cmd_serve") as cmd:
            self._run_main(["serve", "--db", self.db])
            self.assertTrue(cmd.called)

    def test_main_dispatches_inspect_json_command(self):
        with patch.object(cli, "cmd_inspect") as cmd:
            self._run_main(["inspect", "--db", self.db, "--json"])
            self.assertTrue(cmd.called)


if __name__ == "__main__":
    unittest.main()
