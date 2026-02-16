"""Tests for service management."""

import os
import platform
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import _detect_platform, install_service, uninstall_service, service_status


class TestDetectPlatform(unittest.TestCase):
    @patch("platform.system", return_value="Darwin")
    def test_macos(self, _):
        self.assertEqual(_detect_platform(), "macos")

    @patch("platform.system", return_value="Linux")
    def test_linux(self, _):
        self.assertEqual(_detect_platform(), "linux")

    @patch("platform.system", return_value="Windows")
    def test_other(self, _):
        self.assertEqual(_detect_platform(), "other")


class TestInstallServiceMacOS(unittest.TestCase):
    @patch("service._detect_platform", return_value="macos")
    @patch("service.PLIST_PATH")
    @patch("subprocess.run")
    def test_install_creates_plist(self, mock_run, mock_plist, mock_plat):
        tmpdir = tempfile.mkdtemp()
        plist_path = os.path.join(tmpdir, "com.synapse.memory.plist")
        with patch("service.PLIST_PATH", plist_path):
            from service import _install_launchd
            result = _install_launchd("/tmp/test_db", "daily")
            self.assertEqual(result, plist_path)
            self.assertTrue(os.path.exists(plist_path))
            content = open(plist_path).read()
            self.assertIn("com.synapse.memory", content)
            self.assertIn("/tmp/test_db", content)

    @patch("service._detect_platform", return_value="other")
    def test_install_unsupported(self, _):
        result = install_service("/tmp/db")
        self.assertEqual(result, "")


class TestUninstallService(unittest.TestCase):
    @patch("service._detect_platform", return_value="macos")
    @patch("subprocess.run")
    def test_uninstall_macos_not_installed(self, mock_run, _):
        with patch("service.PLIST_PATH", "/tmp/nonexistent_plist"):
            result = uninstall_service()
            self.assertFalse(result)


class TestServiceStatus(unittest.TestCase):
    @patch("service._detect_platform", return_value="macos")
    def test_status_not_installed(self, _):
        with patch("service.PLIST_PATH", "/tmp/nonexistent_plist"):
            result = service_status()
            self.assertFalse(result["installed"])
            self.assertFalse(result["running"])


if __name__ == "__main__":
    unittest.main()
