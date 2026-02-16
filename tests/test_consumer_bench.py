"""Tests for the consumer-facing benchmark harness."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.consumer_bench import (
    ALL_SCENARIOS,
    generate_report_md,
    generate_transcript_md,
    run_all,
    run_contradictions_scenario,
    run_recall_scenario,
    run_timetravel_scenario,
    save_artifacts,
)


class TestRecallScenario(unittest.TestCase):
    def test_runs_without_error(self):
        result = run_recall_scenario()
        self.assertEqual(result["name"], "Long Thread Recall")
        self.assertIn("without", result)
        self.assertIn("with", result)

    def test_synapse_returns_results(self):
        result = run_recall_scenario()
        self.assertGreaterEqual(result["with"]["relevant_found"], 0)
        self.assertGreater(result["with"]["tokens_injected"], 0)

    def test_total_relevant(self):
        result = run_recall_scenario()
        self.assertEqual(result["total_relevant"], 7)


class TestTimeTravelScenario(unittest.TestCase):
    def test_runs_without_error(self):
        result = run_timetravel_scenario()
        self.assertEqual(result["name"], "Time Travel")
        self.assertIn("queries", result)

    def test_has_results(self):
        result = run_timetravel_scenario()
        self.assertEqual(len(result["queries"]), 3)
        self.assertGreaterEqual(result["correct"], 0)


class TestContradictionsScenario(unittest.TestCase):
    def test_runs_without_error(self):
        result = run_contradictions_scenario()
        self.assertEqual(result["name"], "Contradiction Resilience")
        self.assertIn("tests", result)

    def test_has_results(self):
        result = run_contradictions_scenario()
        self.assertEqual(len(result["tests"]), 2)


class TestReportGeneration(unittest.TestCase):
    def setUp(self):
        self.results = run_all()

    def test_markdown_report(self):
        md = generate_report_md(self.results)
        self.assertIn("Synapse AI Memory", md)
        self.assertIn("Long Thread Recall", md)
        self.assertIn("Time Travel", md)
        self.assertIn("Contradiction Resilience", md)
        self.assertIn("Summary", md)

    def test_transcript_generation(self):
        md = generate_transcript_md(self.results)
        self.assertIn("Without Synapse", md)
        self.assertIn("With Synapse", md)

    def test_json_output_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_artifacts(self.results, tmpdir, fmt="json")
            json_path = os.path.join(tmpdir, "results.json")
            self.assertTrue(os.path.exists(json_path))
            with open(json_path) as f:
                data = json.load(f)
            self.assertIn("recall", data)
            self.assertIn("timetravel", data)
            self.assertIn("contradictions", data)

    def test_save_artifacts_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_artifacts(self.results, tmpdir, fmt="md")
            self.assertTrue(any("report.md" in p for p in written))
            self.assertTrue(any("transcript.md" in p for p in written))

    def test_save_artifacts_both(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_artifacts(self.results, tmpdir, fmt="both")
            basenames = [os.path.basename(p) for p in written]
            self.assertIn("report.md", basenames)
            self.assertIn("transcript.md", basenames)
            self.assertIn("results.json", basenames)


class TestCLIIntegration(unittest.TestCase):
    def test_cmd_bench_exists(self):
        from cli import cmd_bench
        self.assertTrue(callable(cmd_bench))

    def test_individual_scenario_selection(self):
        for name in ALL_SCENARIOS:
            result = run_all([name])
            self.assertIn(name, result)
            self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
