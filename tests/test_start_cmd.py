"""Tests for the synapse start command."""
import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import cmd_start, _appliance_state_dir


class TestCmdStartStatusDetection(unittest.TestCase):
    def test_start_detects_targets(self):
        """cmd_start should call _detect_targets and print status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                db=os.path.join(tmpdir, "test_store"),
                data=None,
                port=9470,
                inspect_flag=False,
                sleep_schedule=None,
            )
            with patch("cli._read_daemon_pid", return_value=None), \
                 patch("cli._start_synapse_daemon") as mock_daemon, \
                 patch("cli._wait_for_pid_file", return_value=12345), \
                 patch("cli._write_daemon_pid"), \
                 patch("cli._clear_daemon_pid"), \
                 patch("installer._detect_targets", return_value={"claude": True, "openclaw": False, "nanoclaw": False}):
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_daemon.return_value = mock_proc
                # Should not raise
                cmd_start(args)

    def test_start_already_running(self):
        """cmd_start should skip daemon launch if already running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                db=os.path.join(tmpdir, "test_store"),
                data=None,
                port=9470,
                inspect_flag=False,
                sleep_schedule=None,
            )
            with patch("cli._read_daemon_pid", return_value=999), \
                 patch("cli._pid_is_running", return_value=True), \
                 patch("cli._start_synapse_daemon") as mock_daemon, \
                 patch("installer._detect_targets", return_value={"claude": False, "openclaw": False, "nanoclaw": False}):
                cmd_start(args)
                mock_daemon.assert_not_called()

    def test_start_sleep_schedule(self):
        """cmd_start should write sleep schedule to config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, ".synapse")
            args = argparse.Namespace(
                db=os.path.join(tmpdir, "test_store"),
                data=None,
                port=9470,
                inspect_flag=False,
                sleep_schedule="daily",
            )
            with patch("cli._read_daemon_pid", return_value=None), \
                 patch("cli._start_synapse_daemon") as mock_daemon, \
                 patch("cli._wait_for_pid_file", return_value=100), \
                 patch("cli._write_daemon_pid"), \
                 patch("cli._clear_daemon_pid"), \
                 patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("installer._detect_targets", return_value={"claude": False, "openclaw": False, "nanoclaw": False}):
                mock_proc = MagicMock()
                mock_proc.pid = 100
                mock_daemon.return_value = mock_proc

                # Patch _appliance_pid_path to use our tmpdir
                pid_path = os.path.join(state_dir, "synapse.pid")
                with patch("cli._appliance_pid_path", return_value=pid_path):
                    cmd_start(args)

                config_path = os.path.join(state_dir, "config.json")
                self.assertTrue(os.path.exists(config_path))
                with open(config_path) as f:
                    cfg = json.load(f)
                self.assertEqual(cfg["sleep_schedule"], "daily")


class TestStartOutputFormat(unittest.TestCase):
    def test_output_contains_status_box(self):
        """Start command output should include the status box."""
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                db=os.path.join(tmpdir, "test_store"),
                data=None,
                port=9470,
                inspect_flag=False,
                sleep_schedule=None,
            )
            buf = io.StringIO()
            with redirect_stdout(buf), \
                 patch("cli._read_daemon_pid", return_value=999), \
                 patch("cli._pid_is_running", return_value=True), \
                 patch("installer._detect_targets", return_value={"claude": False, "openclaw": True, "nanoclaw": False}):
                cmd_start(args)

            output = buf.getvalue()
            self.assertIn("Synapse AI Memory", output)
            self.assertIn("Memory service", output)


if __name__ == "__main__":
    unittest.main()
