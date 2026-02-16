"""Tests for enhanced doctor command."""
import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import cmd_doctor_enhanced


class TestDoctorEnhancedBasic(unittest.TestCase):
    def test_doctor_runs_without_error(self):
        """Doctor should run on a fresh store without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=False)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None), \
                 patch("cli._pid_is_running", return_value=False):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                output = buf.getvalue()
                self.assertIn("Synapse Health Check", output)

    def test_doctor_json_output(self):
        """Doctor with --json should output valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=True)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None), \
                 patch("cli._pid_is_running", return_value=False):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                output = buf.getvalue()
                data = json.loads(output)
                self.assertIsInstance(data, list)
                self.assertTrue(len(data) > 0)
                # Each check should have name, status, detail
                for check in data:
                    self.assertIn("name", check)
                    self.assertIn("status", check)

    def test_doctor_checks_storage(self):
        """Doctor should report on storage health."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=True)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                data = json.loads(buf.getvalue())
                names = [c["name"] for c in data]
                # Should have storage checks
                self.assertTrue(any("storage" in n for n in names))

    def test_doctor_checks_index_consistency(self):
        """Doctor should check BM25 index consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=True)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                data = json.loads(buf.getvalue())
                names = [c["name"] for c in data]
                self.assertTrue(any("bm25" in n for n in names))

    def test_doctor_checks_targets(self):
        """Doctor should check connected targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=True)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                data = json.loads(buf.getvalue())
                names = [c["name"] for c in data]
                self.assertTrue(any("target" in n for n in names))

    def test_doctor_checks_daemon(self):
        """Doctor should check daemon status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=True)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                data = json.loads(buf.getvalue())
                names = [c["name"] for c in data]
                self.assertTrue(any("daemon" in n for n in names))


class TestDoctorPrettyOutput(unittest.TestCase):
    def test_pretty_shows_icons(self):
        """Pretty output should include check/warning icons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store")
            args = argparse.Namespace(db=db_path, data=None, json=False)
            state_dir = os.path.join(tmpdir, ".synapse")
            with patch("cli._appliance_state_dir", return_value=state_dir), \
                 patch("cli.APPLIANCE_STATE_DIR", state_dir), \
                 patch("cli._read_daemon_pid", return_value=None):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cmd_doctor_enhanced(args)
                output = buf.getvalue()
                # Should have at least one status icon
                self.assertTrue("✅" in output or "⚠️" in output or "❌" in output)


if __name__ == "__main__":
    unittest.main()
