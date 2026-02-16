"""Tests for the enhanced synapse doctor command."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDoctorEnhanced(unittest.TestCase):
    def _make_args(self, db_path, json_flag=False):
        import argparse
        return argparse.Namespace(db=db_path, data=None, json=json_flag)

    def test_doctor_runs_without_error(self):
        from cli import cmd_doctor_enhanced
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_store")
            args = self._make_args(db)
            # Should not raise
            cmd_doctor_enhanced(args)

    def test_doctor_json_output(self):
        from cli import cmd_doctor_enhanced
        import io
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_store")
            args = self._make_args(db, json_flag=True)
            
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                cmd_doctor_enhanced(args)
            
            output = buf.getvalue()
            # Should be valid JSON
            results = json.loads(output)
            self.assertIsInstance(results, list)
            self.assertTrue(len(results) > 0)
            # Each result should have name, status, detail
            for r in results:
                self.assertIn("name", r)
                self.assertIn("status", r)

    def test_doctor_checks_storage(self):
        from cli import cmd_doctor_enhanced
        import io
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_store")
            args = self._make_args(db, json_flag=True)
            
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                cmd_doctor_enhanced(args)
            
            results = json.loads(buf.getvalue())
            names = [r["name"] for r in results]
            # Should have storage checks
            self.assertTrue(any("storage" in n for n in names))


if __name__ == "__main__":
    unittest.main()
