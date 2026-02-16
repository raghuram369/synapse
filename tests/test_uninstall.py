"""Tests for uninstall functions."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUninstallClaude(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "claude_desktop_config.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("installer._claude_config_path")
    def test_uninstall_claude_removes_synapse(self, mock_path):
        mock_path.return_value = self.config_path
        config = {"mcpServers": {"synapse": {"command": "python3"}, "other": {"command": "node"}}}
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        from installer import uninstall_claude
        result = uninstall_claude()
        self.assertTrue(result)

        with open(self.config_path) as f:
            updated = json.load(f)
        self.assertNotIn("synapse", updated["mcpServers"])
        self.assertIn("other", updated["mcpServers"])

    @patch("installer._claude_config_path")
    def test_uninstall_claude_not_found(self, mock_path):
        mock_path.return_value = os.path.join(self.tmpdir, "nonexistent.json")
        from installer import uninstall_claude
        result = uninstall_claude()
        self.assertFalse(result)

    @patch("installer._claude_config_path")
    def test_uninstall_claude_no_synapse_key(self, mock_path):
        mock_path.return_value = self.config_path
        config = {"mcpServers": {"other": {"command": "node"}}}
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        from installer import uninstall_claude
        result = uninstall_claude()
        self.assertFalse(result)


class TestUninstallOpenClaw(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("installer._openclaw_workspace_root")
    def test_uninstall_openclaw(self, mock_root):
        mock_root.return_value = self.tmpdir
        skill_dir = os.path.join(self.tmpdir, "synapse")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("test")

        from installer import uninstall_openclaw
        result = uninstall_openclaw()
        self.assertTrue(result)
        self.assertFalse(os.path.exists(skill_dir))

    @patch("installer._openclaw_workspace_root")
    def test_uninstall_openclaw_not_found(self, mock_root):
        mock_root.return_value = self.tmpdir
        from installer import uninstall_openclaw
        result = uninstall_openclaw()
        self.assertFalse(result)


class TestUninstallNanoClaw(unittest.TestCase):
    @patch("installer._nanoclaw_workspace_root")
    def test_uninstall_nanoclaw_not_found(self, mock_root):
        mock_root.return_value = tempfile.mkdtemp()
        from installer import uninstall_nanoclaw
        result = uninstall_nanoclaw()
        self.assertFalse(result)


class TestUninstallAll(unittest.TestCase):
    @patch("installer.uninstall_nanoclaw", return_value=False)
    @patch("installer.uninstall_openclaw", return_value=True)
    @patch("installer.uninstall_claude", return_value=True)
    def test_uninstall_all(self, mock_c, mock_o, mock_n):
        from installer import uninstall_all
        results = uninstall_all()
        self.assertTrue(results["claude"])
        self.assertTrue(results["openclaw"])
        self.assertFalse(results["nanoclaw"])


if __name__ == "__main__":
    unittest.main()
