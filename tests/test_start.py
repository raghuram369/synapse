"""Tests for the synapse start golden path command."""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCmdStart(unittest.TestCase):
    """Test cmd_start logic."""

    @patch("cli._start_synapse_daemon")
    @patch("cli._wait_for_pid_file", return_value=12345)
    @patch("cli._read_daemon_pid", return_value=None)
    @patch("cli._pid_is_running", return_value=False)
    def test_start_prints_status_box(self, mock_running, mock_pid, mock_wait, mock_daemon):
        from cli import cmd_start
        import argparse

        with patch("installer._detect_targets", return_value={"claude": False, "openclaw": False, "nanoclaw": False}):
            args = argparse.Namespace(
                db=":memory:", port=9470, inspect_flag=False, sleep_schedule=None, data=None,
            )
            # Should not raise
            cmd_start(args)

    @patch("cli._start_synapse_daemon")
    @patch("cli._wait_for_pid_file", return_value=99)
    @patch("cli._read_daemon_pid", return_value=None)
    @patch("cli._pid_is_running", return_value=False)
    def test_start_with_sleep_schedule(self, mock_running, mock_pid, mock_wait, mock_daemon):
        import tempfile, argparse
        from cli import cmd_start, _appliance_state_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("cli._appliance_state_dir", return_value=tmpdir), \
                 patch("cli._appliance_pid_path", return_value=os.path.join(tmpdir, "synapse.pid")), \
                 patch("installer._detect_targets", return_value={"claude": False, "openclaw": False, "nanoclaw": False}):
                args = argparse.Namespace(
                    db=":memory:", port=9470, inspect_flag=False, sleep_schedule="daily", data=None,
                )
                cmd_start(args)

                config_path = os.path.join(tmpdir, "config.json")
                self.assertTrue(os.path.exists(config_path))
                with open(config_path) as f:
                    cfg = json.load(f)
                self.assertEqual(cfg["sleep_schedule"], "daily")

    @patch("cli._read_daemon_pid", return_value=999)
    @patch("cli._pid_is_running", return_value=True)
    def test_start_already_running(self, mock_running, mock_pid):
        import argparse
        from cli import cmd_start

        with patch("installer._detect_targets", return_value={"claude": True, "openclaw": True, "nanoclaw": False}):
            args = argparse.Namespace(
                db=":memory:", port=9470, inspect_flag=False, sleep_schedule=None, data=None,
            )
            # Should not start a new daemon
            cmd_start(args)


if __name__ == "__main__":
    unittest.main()
