"""Tests for capture module (clip functions + router integration)."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse
from capture import clip_text, clip_stdin, IngestResult


class TestClipText(unittest.TestCase):
    def test_clip_text_basic(self):
        s = Synapse(":memory:")
        result = clip_text(s, "I love programming in Python")
        self.assertEqual(result, IngestResult.STORED)
        self.assertGreaterEqual(s.count(), 1)
        s.close()

    def test_clip_text_with_tags(self):
        s = Synapse(":memory:")
        result = clip_text(s, "Important project deadline is Friday", tags=["project:x"])
        self.assertEqual(result, IngestResult.STORED)
        s.close()

    def test_clip_text_with_source(self):
        s = Synapse(":memory:")
        result = clip_text(s, "We decided to use React for the frontend", source="test")
        self.assertEqual(result, IngestResult.STORED)
        s.close()

    def test_clip_text_fluff_ignored(self):
        s = Synapse(":memory:")
        result = clip_text(s, "ok")
        self.assertEqual(result, IngestResult.IGNORED_FLUFF)
        s.close()

    def test_clip_text_secret_rejected(self):
        s = Synapse(":memory:")
        result = clip_text(s, "sk-abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(result, IngestResult.REJECTED_SECRET)
        s.close()


class TestClipStdin(unittest.TestCase):
    def test_clip_stdin(self):
        s = Synapse(":memory:")
        with patch("sys.stdin", StringIO("I prefer using VS Code for development\n")):
            result = clip_stdin(s)
        self.assertIsNotNone(result)
        self.assertEqual(result, IngestResult.STORED)
        s.close()

    def test_clip_stdin_empty(self):
        s = Synapse(":memory:")
        with patch("sys.stdin", StringIO("")):
            result = clip_stdin(s)
        self.assertIsNone(result)
        s.close()


class TestGetClipboard(unittest.TestCase):
    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_get_clipboard_macos(self, mock_sys, mock_run):
        mock_run.return_value = MagicMock(stdout="clipboard text", returncode=0)
        from capture import _get_clipboard
        result = _get_clipboard()
        self.assertEqual(result, "clipboard text")
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
