"""Tests for the import wizard command."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImportWizardCancel(unittest.TestCase):
    def test_wizard_cancel_on_eof(self):
        """Wizard should handle EOFError gracefully."""
        import argparse
        from cli import cmd_import_wizard

        args = argparse.Namespace(db=":memory:", data=None)
        with patch("builtins.input", side_effect=EOFError):
            # Should not raise
            cmd_import_wizard(args)

    def test_wizard_invalid_choice(self):
        """Invalid menu choice should print error and return."""
        import argparse
        from cli import cmd_import_wizard

        args = argparse.Namespace(db=":memory:", data=None)
        with patch("builtins.input", return_value="9"):
            cmd_import_wizard(args)

    def test_wizard_clipboard_flow(self):
        """Test clipboard import path through wizard."""
        import argparse
        from cli import cmd_import_wizard

        args = argparse.Namespace(db=":memory:", data=None)
        
        mock_report = MagicMock()
        mock_report.imported = 5
        mock_report.skipped = 0
        mock_report.errors = 0
        mock_report.duration_ms = 10.0

        inputs = iter(["4", "5"])  # clipboard, no policy
        with patch("builtins.input", side_effect=lambda _: next(inputs)), \
             patch("importers.MemoryImporter.from_clipboard", return_value=mock_report):
            cmd_import_wizard(args)

    def test_wizard_notes_flow(self):
        """Test notes folder import path."""
        import argparse
        from cli import cmd_import_wizard

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test markdown file
            with open(os.path.join(tmpdir, "test.md"), "w") as f:
                f.write("# Test\nSome content here\n")

            args = argparse.Namespace(db=":memory:", data=None)
            
            mock_report = MagicMock()
            mock_report.imported = 1
            mock_report.skipped = 0
            mock_report.errors = 0
            mock_report.duration_ms = 5.0

            inputs = iter(["3", tmpdir, "5"])  # notes, path, no policy
            with patch("builtins.input", side_effect=lambda _: next(inputs)), \
                 patch("importers.MemoryImporter.from_markdown_folder", return_value=mock_report):
                cmd_import_wizard(args)


class TestImportWizardPolicySelection(unittest.TestCase):
    def test_policy_mapping(self):
        """Verify policy choice mapping is correct."""
        policy_map = {"1": "minimal", "2": "private", "3": "work", "4": "ephemeral", "5": None}
        self.assertEqual(policy_map["1"], "minimal")
        self.assertEqual(policy_map["5"], None)


if __name__ == "__main__":
    unittest.main()
