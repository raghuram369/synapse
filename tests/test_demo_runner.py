import argparse
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import cli
from demo_runner import DemoRunner


class TestDemoRunnerCore(unittest.TestCase):
    def setUp(self):
        self.runner = DemoRunner()

    def test_scenarios_exist_with_expected_structure(self):
        self.assertEqual(set(self.runner.SCENARIOS.keys()), {"diet", "travel", "project"})
        for name, data in self.runner.SCENARIOS.items():
            self.assertIn("description", data)
            self.assertIsInstance(data["memories"], list)
            self.assertIsInstance(data["queries"], list)
            self.assertGreater(len(data["memories"]), 0)
            self.assertGreater(len(data["queries"]), 0)

    def test_run_unknown_scenario_fails(self):
        with self.assertRaises(ValueError):
            self.runner.run("planet")

    def test_run_markdown_includes_sections_and_scores(self):
        output = self.runner.run("travel", output="markdown")
        self.assertIn("# Demo Scenario: Travel", output)
        self.assertIn("## Stored memories", output)
        self.assertIn("## Queries", output)
        self.assertIn("## Sleep digest", output)
        self.assertIn("## Contradictions", output)
        self.assertIn("Score breakdown", output)
        self.assertIn("BM25:", output)

    def test_run_terminal_prints_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            output = self.runner.run("diet", output="terminal")
        rendered = buf.getvalue()
        self.assertIn("Demo Scenario: Diet", rendered)
        self.assertIn("Stored memories", rendered)
        self.assertIn("Queries", rendered)
        self.assertIn("Sleep digest", rendered)
        self.assertIn("Contradictions", rendered)
        self.assertTrue(output)

    def test_run_all_runs_all_scenarios_in_markdown(self):
        output = self.runner.run_all(output="markdown")
        self.assertIn("# Demo Scenario: Diet", output)
        self.assertIn("# Demo Scenario: Travel", output)
        self.assertIn("# Demo Scenario: Project", output)


class TestDemoRunnerCLI(unittest.TestCase):
    def _run_main(self, argv):
        with patch.object(sys, "argv", ["synapse"] + argv):
            out = io.StringIO()
            with redirect_stdout(out):
                try:
                    cli.main()
                except SystemExit:
                    pass
            return out.getvalue()

    def test_cmd_demo_dispatch_from_cli(self):
        with patch.object(cli, "cmd_demo") as cmd_demo:
            self._run_main(["demo", "--scenario", "travel"])
            self.assertTrue(cmd_demo.called)
            namespace = cmd_demo.call_args[0][0]
            self.assertEqual(namespace.scenario, "travel")
            self.assertFalse(namespace.markdown)

    def test_cmd_demo_accepts_markdown_flag(self):
        with patch.object(cli, "cmd_demo") as cmd_demo:
            self._run_main(["demo", "--scenario", "all", "--markdown"])
            namespace = cmd_demo.call_args[0][0]
            self.assertEqual(namespace.scenario, "all")
            self.assertTrue(namespace.markdown)

    def test_cmd_demo_runs_markdown_output(self):
        with patch.object(cli, "DemoRunner") as demo_runner:
            runner = demo_runner.return_value
            runner.run.return_value = "MARKDOWN"
            runner.run_all.return_value = "ALL MARKDOWN"
            out = io.StringIO()
            with redirect_stdout(out):
                cli.cmd_demo(argparse.Namespace(scenario="all", markdown=True))
            self.assertIn("ALL MARKDOWN", out.getvalue())
            runner.run_all.assert_called_once_with(output="markdown")

    def test_cmd_demo_runs_terminal_output(self):
        with patch.object(cli, "DemoRunner") as demo_runner:
            runner = demo_runner.return_value
            runner.run.return_value = "TEXT"
            out = io.StringIO()
            with redirect_stdout(out):
                cli.cmd_demo(argparse.Namespace(scenario="project", markdown=False))
            self.assertEqual(out.getvalue(), "")
            runner.run.assert_called_once_with(scenario="project", output="terminal")


if __name__ == "__main__":
    unittest.main()
