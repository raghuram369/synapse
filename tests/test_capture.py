"""Tests for capture module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse
from capture import clip_text, clip_stdin


class TestClipText(unittest.TestCase):
    def test_clip_text_basic(self):
        s = Synapse(":memory:")
        memory = clip_text(s, "hello world")
        self.assertEqual(memory.content, "hello world")
        s.close()

    def test_clip_text_with_tags(self):
        s = Synapse(":memory:")
        memory = clip_text(s, "tagged content", tags=["project:x"])
        self.assertIn("project:x", memory.metadata.get("tags", []))
        s.close()

    def test_clip_text_with_source(self):
        s = Synapse(":memory:")
        memory = clip_text(s, "sourced", source="test")
        self.assertEqual(memory.metadata.get("source"), "test")
        s.close()


class TestClipStdin(unittest.TestCase):
    def test_clip_stdin(self):
        s = Synapse(":memory:")
        with patch("sys.stdin", StringIO("piped input\n")):
            memory = clip_stdin(s)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "piped input")
        s.close()

    def test_clip_stdin_empty(self):
        s = Synapse(":memory:")
        with patch("sys.stdin", StringIO("")):
            memory = clip_stdin(s)
        self.assertIsNone(memory)
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
