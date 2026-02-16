#!/usr/bin/env python3

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import cli
from sleep import SleepReport


class TestSleepDigest(unittest.TestCase):
    def _sample_report(self, **kwargs):
        payload = {
            "consolidated": 3,
            "promoted": 2,
            "patterns_found": 1,
            "contradictions": 1,
            "pruned": 4,
            "graph_cleaned": 2,
            "duration_ms": 12.5,
            "details": {
                "promoted_facts": ["User typically does planning"],
                "patterns": {"examples": ["User frequently combines python and database tasks."]},
                "contradictions": {"examples": ["Conflicting diet preferences."]},
                "consolidation": {"memos_merged": 3},
                "graph_cleanup": {"items": 2},
                "hot_topics": ["python (0.900)", "database (0.700)"],
            },
        }
        payload.update(kwargs)
        return SleepReport(**payload)

    def test_to_digest_includes_header_and_summary(self):
        text = self._sample_report().to_digest()
        self.assertIn("Sleep Digest —", text)
        self.assertIn("Summary:", text)
        self.assertIn("consolidated=3", text)

    def test_to_digest_includes_promoted_facts_section_when_present(self):
        text = self._sample_report().to_digest()
        self.assertIn("Promoted Facts", text)
        self.assertIn("User typically does planning", text)

    def test_to_digest_includes_patterns_contradictions_cleanup_and_hot_topics(self):
        text = self._sample_report().to_digest()
        self.assertIn("Patterns Discovered", text)
        self.assertIn("Contradictions Found", text)
        self.assertIn("Cleanup Stats", text)
        self.assertIn("Hot Topics", text)
        self.assertIn("- merged: 3", text)
        self.assertIn("- orphans: 2", text)

    def test_to_digest_includes_suggestions(self):
        text = self._sample_report().to_digest()
        self.assertIn("Suggestions", text)
        self.assertIn("Review and resolve contradiction pairs.", text)

    def test_to_digest_uses_none_markers_when_sections_empty(self):
        report = self._sample_report(
            promoted=0,
            patterns_found=0,
            contradictions=0,
            pruned=0,
            details={
                "patterns": {"examples": []},
                "contradictions": {"examples": []},
                "hot_topics": [],
            },
        )
        text = report.to_digest()
        self.assertNotIn("Promoted Facts", text)
        self.assertIn("Patterns Discovered", text)
        self.assertIn("- none", text)
        self.assertIn("Capture more repeated behaviors to improve pattern mining.", text)

    def test_cmd_sleep_digest_prints_human_readable_digest(self):
        report = self._sample_report()

        class _FakeSynapse:
            def __init__(self, _db):
                pass

            def sleep(self, verbose=False):
                self.verbose = verbose
                return report

            def close(self):
                return None

        with patch("synapse.Synapse", _FakeSynapse):
            out = io.StringIO()
            with redirect_stdout(out):
                cli.cmd_sleep(type("Args", (), {"db": ":memory:", "digest": True, "verbose": False})())
        text = out.getvalue()
        self.assertIn("Sleep Digest —", text)
        self.assertIn("Hot Topics", text)


if __name__ == "__main__":
    unittest.main()
