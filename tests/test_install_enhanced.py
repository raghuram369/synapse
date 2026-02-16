"""Tests for enhanced installer features."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import (
    _detect_targets,
    _verify_claude,
    _verify_openclaw_skill,
    install_all,
    install_claude_enhanced,
    install_openclaw_enhanced,
    install_nanoclaw_enhanced,
    _openclaw_manifest,
)


class TestDetectTargets(unittest.TestCase):
    def test_detect_returns_dict(self):
        targets = _detect_targets("test_db")
        self.assertIsInstance(targets, dict)
        self.assertIn("claude", targets)
        self.assertIn("openclaw", targets)
        self.assertIn("nanoclaw", targets)


class TestManifestVersion(unittest.TestCase):
    def test_version_is_0_8_0(self):
        manifest = _openclaw_manifest()
        self.assertEqual(manifest["version"], "0.8.0")


class TestVerifyClaude(unittest.TestCase):
    def test_verify_missing_config(self):
        with patch("installer._claude_config_path", return_value="/tmp/nonexistent_config_xyz.json"):
            ok, detail = _verify_claude()
            self.assertFalse(ok)

    def test_verify_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {"synapse": {}}}, f)
            f.flush()
            with patch("installer._claude_config_path", return_value=f.name):
                ok, detail = _verify_claude()
                self.assertTrue(ok)
            os.unlink(f.name)


class TestInstallClaudeEnhanced(unittest.TestCase):
    def test_dry_run(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            with patch("installer._claude_config_path", return_value=f.name):
                result = install_claude_enhanced("test_db", dry_run=True)
                self.assertTrue(result)
            os.unlink(f.name)

    def test_verify_only(self):
        with patch("installer._claude_config_path", return_value="/tmp/nonexistent.json"):
            result = install_claude_enhanced("test_db", verify_only=True)
            self.assertFalse(result)

    def test_actual_install(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            with patch("installer._claude_config_path", return_value=f.name):
                result = install_claude_enhanced("test_db")
                self.assertTrue(result)
                with open(f.name) as fp:
                    cfg = json.load(fp)
                self.assertIn("synapse", cfg.get("mcpServers", {}))
            os.unlink(f.name)


class TestInstallOpenClawEnhanced(unittest.TestCase):
    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("installer._openclaw_workspace_root", return_value=tmpdir):
                result = install_openclaw_enhanced("test_db", dry_run=True)
                self.assertTrue(result)

    def test_actual_install_and_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("installer._openclaw_workspace_root", return_value=tmpdir):
                result = install_openclaw_enhanced("test_db")
                self.assertTrue(result)
                ok, _ = _verify_openclaw_skill(tmpdir)
                self.assertTrue(ok)


class TestInstallAll(unittest.TestCase):
    def test_install_all_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "claude_config.json")
            with open(config_path, "w") as f:
                json.dump({}, f)
            with patch("installer._claude_config_path", return_value=config_path), \
                 patch("installer._openclaw_workspace_root", return_value=tmpdir), \
                 patch("installer._nanoclaw_workspace_root", return_value=tmpdir):
                results = install_all("test_db", dry_run=True)
                self.assertIsInstance(results, dict)


if __name__ == "__main__":
    unittest.main()
