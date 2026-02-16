#!/usr/bin/env python3
"""Tests for Synapse appliance lifecycle commands."""

import io
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import cli
import synapsed


class TestApplianceLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synapse-lifecycle-")
        self.pid_path = os.path.join(self.tmp.name, "synapse.pid")

    def tearDown(self):
        self.tmp.cleanup()

    def _capture(self, fn, *args, **kwargs) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            fn(*args, **kwargs)
        return out.getvalue()

    def test_up_prints_running_status(self):
        args = Namespace(db="/tmp/db", port=9470, mode="appliance")
        process = Mock()
        process.pid = 9012

        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=None), \
             patch.object(cli, "_start_synapse_daemon", return_value=process), \
             patch.object(cli, "_wait_for_pid_file", return_value=7777) as wait_pid:

            output = self._capture(cli.cmd_up, args)

        wait_pid.assert_called_once()
        self.assertIn("🧠 Synapse AI Memory running (pid 7777, port 9470)", output)
        self.assertIn("MCP stdio: synapse serve | HTTP: http://localhost:9470", output)

    def test_up_prevents_start_when_already_running(self):
        args = Namespace(db="/tmp/db", port=9470, mode="appliance")

        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=123), \
             patch.object(cli, "_pid_is_running", return_value=True), \
             patch.object(cli, "_start_synapse_daemon") as launch:

            output = self._capture(cli.cmd_up, args)

        launch.assert_not_called()
        self.assertIn("⚠️  Synapse already running (pid 123, port 9470)", output)

    def test_up_uses_selected_mode(self):
        args = Namespace(db="/tmp/db", port=9542, mode="full")
        process = Mock()
        process.pid = 1010

        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=None), \
             patch.object(cli, "_start_synapse_daemon", return_value=process), \
             patch.object(cli, "_wait_for_pid_file", return_value=5050), \
             patch.object(cli, "_daemon_command") as command_factory:

            self._capture(cli.cmd_up, args)

            command_factory.assert_called_once_with(port=9542, db_path="/tmp/db", mode="full")

    def test_down_sends_sigterm_and_confirms_stopped(self):
        args = Namespace()
        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=4242), \
             patch.object(cli, "_pid_is_running", side_effect=[True, False]), \
             patch.object(cli, "_send_signal") as send_signal, \
             patch.object(cli, "_clear_daemon_pid") as clear_pid:

            output = self._capture(cli.cmd_down, args)

        send_signal.assert_called_once_with(4242, unittest.mock.ANY)
        self.assertIn("Synapse stopped.", output)
        clear_pid.assert_called_once_with(self.pid_path)

    def test_down_without_pid_reports_stopped(self):
        args = Namespace()
        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=None), \
             patch.object(cli, "_clear_daemon_pid") as clear_pid:

            output = self._capture(cli.cmd_down, args)

        clear_pid.assert_called_once_with(self.pid_path)
        self.assertIn("Synapse stopped.", output)

    def test_status_running_shows_health_snapshot(self):
        args = Namespace(db="/tmp/db", port=9470)
        with open(self.pid_path, "w", encoding="utf-8") as fp:
            fp.write("7070")
        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=7070), \
             patch.object(cli, "_pid_is_running", return_value=True), \
             patch.object(cli, "_collect_store_snapshot", return_value={
                 "memory_count": 12,
                 "concept_count": 4,
                 "edge_count": 2,
                 "contradiction_count": 1,
                 "top_hot_concepts": [("task", 0.9), ("synapse", 0.8)],
                 "last_sleep_at": 1710000000.0,
             }), \
             patch.object(cli, "_collect_sleep_status", return_value={
                 "should_sleep": True,
                 "next_due_at": 1710003600.0,
             }), \
             patch.object(cli, "_store_file_size", return_value=2048):

            output = self._capture(cli.cmd_status, args)

        self.assertIn("State: running", output)
        self.assertIn("Port: 9470", output)
        self.assertIn("Count: 12", output)
        self.assertIn("Concepts: 4", output)
        self.assertIn("Top 5 hot concepts:", output)
        self.assertIn("task: 0.900", output)

    def test_status_stopped_when_not_running(self):
        args = Namespace(db="/tmp/db", port=9470)
        with patch.object(cli, "_appliance_pid_path", return_value=self.pid_path), \
             patch.object(cli, "_read_daemon_pid", return_value=None):

            output = self._capture(cli.cmd_status, args)

        self.assertIn("State: stopped", output)
        self.assertIn("Uptime:", output)

    def test_synapsed_resolves_db_path_without_fallback_to_parent_file(self):
        self.assertEqual(
            synapsed._resolve_data_dir("/tmp/store/synapse_store.db", None),
            "/tmp/store",
        )
        self.assertEqual(
            synapsed._resolve_data_dir("/tmp/store/synapse_store", None),
            "/tmp/store/synapse_store",
        )
