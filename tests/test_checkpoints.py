#!/usr/bin/env python3

import argparse
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

import cli
from synapse import Synapse


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-checkpoint-tests-")
        self.db = os.path.join(self.tempdir.name, "synapse")

    def tearDown(self):
        self.tempdir.cleanup()

    def _new_synapse(self):
        return Synapse(self.db)

    def test_create_checkpoint_and_verify_exists(self):
        s = self._new_synapse()
        try:
            s.remember("User likes tea", deduplicate=False)
            cp = s.checkpoint("initial", desc="bootstrap")

            self.assertEqual(cp.name, "initial")
            self.assertTrue(cp.checksum)
            self.assertEqual(cp.stats.get("memory_count"), 1)
            self.assertTrue(os.path.exists(cp.snapshot_path))

            items = s.checkpoints.list()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].name, "initial")
        finally:
            s.close()

    def test_list_checkpoints_newest_first(self):
        s = self._new_synapse()
        try:
            s.remember("One", deduplicate=False)
            first = s.checkpoint("first")
            time.sleep(0.01)
            s.remember("Two", deduplicate=False)
            second = s.checkpoint("second")

            names = [cp.name for cp in s.checkpoints.list()]
            self.assertEqual(names[0], second.name)
            self.assertEqual(names[1], first.name)
            self.assertEqual(len(names), 2)
        finally:
            s.close()

    def test_diff_between_checkpoints_detects_changes(self):
        s = self._new_synapse()
        try:
            s.remember("Alice likes tea", deduplicate=False)
            s.remember("Sky is blue", deduplicate=False)
            cp_a = s.checkpoint("before")
            s.remember("Alice likes coffee", deduplicate=False)
            s.remember("It's raining", deduplicate=False)
            cp_b = s.checkpoint("after")

            result = s.checkpoints.diff(cp_a.name, cp_b.name)
            self.assertIn("memories_added", result)
            self.assertIn("beliefs_changed", result)
            self.assertTrue(result["memories_added"])
            self.assertTrue(any(item["fact"] == "alice|likes" for item in result["beliefs_changed"]))
            self.assertTrue(result["contradictions_introduced"] == [])
        finally:
            s.close()

    def test_restore_from_checkpoint(self):
        s = self._new_synapse()
        try:
            s.remember("A memory before restore", deduplicate=False)
            cp = s.checkpoint("restore-point")
            s.remember("Temporary memory", deduplicate=False)

            before_restore = s.count()
            self.assertEqual(before_restore, 2)
            report = s.checkpoints.restore(cp.name)
            self.assertGreaterEqual(report["memories_restored"], 1)
            self.assertEqual(report["checkpoint"], cp.name)
            self.assertEqual(s.count(), 1)
        finally:
            s.close()

    def test_restore_missing_name_raises(self):
        s = self._new_synapse()
        try:
            with self.assertRaises(ValueError):
                s.checkpoints.restore("missing")
        finally:
            s.close()

    def test_delete_checkpoint(self):
        s = self._new_synapse()
        try:
            s.remember("Delete me", deduplicate=False)
            s.checkpoint("to-delete")
            self.assertEqual(len(s.checkpoints.list()), 1)
            deleted = s.checkpoints.delete("to-delete")
            self.assertTrue(deleted)
            self.assertEqual(len(s.checkpoints.list()), 0)
            self.assertFalse(s.checkpoints.delete("to-delete"))
        finally:
            s.close()

    def test_auto_checkpoint_on_memory_threshold(self):
        s = self._new_synapse()
        try:
            s.checkpoints.auto_checkpoint(every_n_memories=2)
            s.remember("first", deduplicate=False)
            s.remember("second", deduplicate=False)
            self.assertEqual(len(s.checkpoints.list()), 0)

            s.sleep(verbose=False)
            self.assertEqual(len(s.checkpoints.list()), 1)
        finally:
            s.close()


class TestCheckpointCli(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-checkpoint-cli-")
        self.db = os.path.join(self.tempdir.name, "synapse")

    def tearDown(self):
        self.tempdir.cleanup()

    def _capture(self, fn, **kwargs):
        buf = io.StringIO()
        namespace = argparse.Namespace(
            db=self.db,
            name=None,
            desc=None,
            a=None,
            b=None,
            confirm=False,
            path=None,
            checkpoint_action=None,
        )
        namespace.__dict__.update(kwargs)
        with redirect_stdout(buf):
            fn(namespace)
        return buf.getvalue()

    def test_cli_create_checkpoint(self):
        out = self._capture(cli.cmd_checkpoint_create, name="cli-checkpoint")
        self.assertIn("Created checkpoint", out)
        self.assertIn("cli-checkpoint", out)

    def test_cli_list_checkpoint(self):
        self._capture(cli.cmd_checkpoint_create, name="cli-list")
        out = self._capture(cli.cmd_checkpoint_list)
        self.assertIn("cli-list", out)
        self.assertIn("Checkpoints", out)

    def test_cli_diff_checkpoint(self):
        s = Synapse(self.db)
        try:
            s.remember("Alice likes tea", deduplicate=False)
            s.checkpoint("a")
            s.remember("Alice likes coffee", deduplicate=False)
            s.checkpoint("b")
        finally:
            s.close()

        out = self._capture(cli.cmd_checkpoint_diff, a="a", b="b")
        self.assertIn("Diff a -> b", out)
        self.assertIn("beliefs_changed", out)

    def test_cli_restore_requires_confirmation(self):
        s = Synapse(self.db)
        try:
            s.remember("Base", deduplicate=False)
            s.checkpoint("restore-cp")
            s.remember("After", deduplicate=False)
        finally:
            s.close()

        denied = self._capture(cli.cmd_checkpoint_restore, name="restore-cp", confirm=False)
        self.assertIn("destructive", denied)

        # now confirm and restore
        restored = self._capture(cli.cmd_checkpoint_restore, name="restore-cp", confirm=True)
        self.assertIn("Restored checkpoint", restored)

        s = Synapse(self.db)
        try:
            self.assertEqual(s.count(), 1)
            self.assertIn("Base", [m for m in [m_data.get('content', '') for m_data in s.store.memories.values()]])
            self.assertNotIn("After", [m for m in [m_data.get('content', '') for m_data in s.store.memories.values()]])
        finally:
            s.close()

    def test_cli_delete_checkpoint(self):
        self._capture(cli.cmd_checkpoint_create, name="delme")
        out = self._capture(cli.cmd_checkpoint_delete, name="delme")
        self.assertIn("Deleted checkpoint", out)
        out = self._capture(cli.cmd_checkpoint_list)
        self.assertIn("No checkpoints found", out)


if __name__ == "__main__":
    unittest.main()
