import argparse
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

import cli
import packs
from packs import BrainPack, parse_range_days
from synapse import Synapse


class TestBrainPackCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synapse-pack-")
        self.db = os.path.join(self.tmp.name, "core.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _new_synapse(self):
        syn = Synapse(self.db)
        syn.should_sleep = lambda: False
        return syn

    def _seed(self, syn, entries):
        memories = []
        for text in entries:
            memories.append(syn.remember(text, deduplicate=False))
        return memories

    def test_parse_range_days_variants(self):
        self.assertEqual(parse_range_days("30d"), 30)
        self.assertEqual(parse_range_days("2w"), 14)
        self.assertEqual(parse_range_days("1m"), 30)
        self.assertEqual(parse_range_days(14), 14)
        self.assertEqual(parse_range_days(None), 30)

    def test_build_collects_memories_and_range(self):
        s = self._new_synapse()
        m1 = self._seed(s, ["I like coffee in the morning"])[0]
        m2 = self._seed(s, ["I like tea in the morning"])[0]
        old_time = time.time() - (3 * 24 * 60 * 60)
        s.store.update_memory(m2.id, {"created_at": old_time, "last_accessed": old_time})
        s.temporal_index.remove_memory(m2.id)
        s.temporal_index.add_memory(m2.id, old_time)
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("coffee", range_days=1).build(syn)
        syn.close()
        self.assertEqual(len(pack.memories), 1)
        self.assertEqual(pack.memories[0]["content"], "I like coffee in the morning")

    def test_build_generates_context_pack(self):
        s = self._new_synapse()
        self._seed(s, ["Alice likes jazz", "Alice likes tea", "Bob likes jazz"])
        s.close()
        syn = self._new_synapse()
        pack = BrainPack("Alice", range_days=30).build(syn)
        syn.close()
        self.assertIsNotNone(pack.context_pack)
        self.assertTrue(pack.graph_slice["nodes"])
        self.assertEqual(pack.range_days, 30)

    def test_build_includes_belief_snapshot(self):
        s = self._new_synapse()
        self._seed(s, ["Alice likes coffee", "Alice likes tea"])
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("alice", range_days=30).build(syn)
        syn.close()
        self.assertIn("alice|likes", pack.belief_snapshot)
        self.assertIn("value", pack.belief_snapshot["alice|likes"])

    def test_build_includes_contradictions(self):
        s = self._new_synapse()
        self._seed(s, ["The sky is blue", "The sky is not blue"])
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("sky", range_days=30).build(syn)
        syn.close()
        self.assertTrue(pack.contradictions)
        self.assertEqual(pack.contradictions[0]["kind"], "polarity")
        self.assertIn(pack.contradictions[0]["memory_id_a"], {1, 2})

    def test_to_markdown_format_has_sections(self):
        s = self._new_synapse()
        self._seed(s, ["Alice enjoys jazz"])
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("Alice", range_days=30).build(syn)
        syn.close()

        report = pack.to_markdown()
        self.assertIn("# 🧠 Brain Pack: Alice", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Timeline", report)
        self.assertIn("## Current Beliefs", report)
        self.assertIn("## Contradictions", report)
        self.assertIn("## Graph", report)
        self.assertIn("Checksum:", report)

    def test_save_load_roundtrip_and_checksum(self):
        s = self._new_synapse()
        self._seed(s, ["Project Sync is running"])
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("project", range_days=30).build(syn)
        syn.close()
        path = os.path.join(self.tmp.name, "project-pack.brain")
        pack.save(path)

        loaded = packs.BrainPack.load(path)
        self.assertEqual(loaded.topic, "project")
        self.assertEqual(loaded.range_days, 30)
        self.assertEqual(loaded.memories, pack.memories)
        self.assertEqual(loaded.checksum, pack.checksum)

    def test_load_reports_checksum_mismatch(self):
        s = self._new_synapse()
        self._seed(s, ["Integrity test memory"])
        s.close()

        syn = self._new_synapse()
        pack = BrainPack("integrity", range_days=30).build(syn)
        syn.close()
        path = os.path.join(self.tmp.name, "bad.brain")
        pack.save(path)

        with open(path, "r+", encoding="utf-8") as file:
            content = file.read()
            file.seek(0)
            file.write(content.replace("integrity", "tampered"))
            file.truncate()

        with self.assertRaises(ValueError):
            packs.BrainPack.load(path)

    def test_replay_reports_missing_and_matches(self):
        db_original = os.path.join(self.tmp.name, "replay_src.db")
        s = Synapse(db_original)
        s.remember("I like coffee", deduplicate=False)
        s.remember("I like tea", deduplicate=False)
        s.close()

        s_src = Synapse(db_original)
        pack = BrainPack("like", range_days=30).build(s_src)
        s_src.close()
        path = os.path.join(self.tmp.name, "like-pack.brain")
        pack.save(path)

        db_target = os.path.join(self.tmp.name, "replay_target.db")
        s = Synapse(db_target)
        s.remember("I like coffee", deduplicate=False)
        s.close()

        s_target = Synapse(db_target)
        replay = packs.BrainPack.load(path).replay(s_target)
        s_target.close()
        self.assertEqual(len(replay["memory_matches"]), 1)
        self.assertEqual(len(replay["memory_missing"]), 1)
        self.assertTrue(replay["memory_missing"][0]["content"].endswith("tea"))

    def test_diff_identifies_memory_changes(self):
        db1 = os.path.join(self.tmp.name, "diff_src1.db")
        db2 = os.path.join(self.tmp.name, "diff_src2.db")
        s1 = Synapse(db1)
        s1.remember("Alice likes coffee", deduplicate=False)
        s1.remember("Alice likes tea", deduplicate=False)
        s1.close()
        s2 = Synapse(db2)
        s2.remember("Alice likes coffee", deduplicate=False)
        s2.remember("Alice likes cake", deduplicate=False)
        s2.close()

        s1 = Synapse(db1)
        p1 = BrainPack("alice", range_days=30).build(s1)
        s1.close()
        s2 = Synapse(db2)
        p2 = BrainPack("alice", range_days=30).build(s2)
        s2.close()
        report = p1.diff(p2)
        self.assertEqual(len(report["memory_added"]), 1)
        self.assertEqual(len(report["memory_removed"]), 1)
        self.assertIn("Brain Pack Diff", report["markdown"])


class TestBrainPackCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synapse-pack-cli-")
        self.db = os.path.join(self.tmp.name, "cli.db")
        self.db2 = os.path.join(self.tmp.name, "cli2.db")
        self.pack_dir = os.path.join(self.tmp.name, "packs")

    def tearDown(self):
        self.tmp.cleanup()

    def _capture(self, fn, **kwargs):
        buffer = io.StringIO()
        defaults = {
            "topic": None,
            "range": "30d",
            "output": None,
            "replay": None,
            "diff": None,
            "list": False,
            "db": None,
            "data": None,
            "pack_dir": self.pack_dir,
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        with redirect_stdout(buffer):
            fn(args)
        return buffer.getvalue()

    def _seed(self, db_path, entries):
        s = Synapse(db_path)
        for text in entries:
            s.remember(text, deduplicate=False)
        s.close()

    def test_cli_build_pack_writes_file(self):
        self._seed(self.db, ["I like coffee"])
        pack_path = os.path.join(self.pack_dir, "cli-build.brain")
        output = self._capture(
            cli.cmd_pack,
            topic="coffee",
            range="30d",
            output=pack_path,
            db=self.db,
        )
        self.assertTrue(os.path.exists(pack_path))
        self.assertIn("Saved brain pack", output)

    def test_cli_replay_pack_prints_report(self):
        source = os.path.join(self.tmp.name, "replay-src.db")
        target = os.path.join(self.tmp.name, "replay-target.db")
        self._seed(source, ["Alice likes jazz", "Alice likes tea"])
        self._seed(target, ["Alice likes jazz"])

        source_pack_path = os.path.join(self.pack_dir, "replay-src.brain")
        self._capture(
            cli.cmd_pack,
            topic="likes",
            output=source_pack_path,
            db=source,
        )

        output = self._capture(
            cli.cmd_pack,
            replay=source_pack_path,
            db=target,
        )
        self.assertIn("Brain Pack Replay", output)
        self.assertIn("missing", output)

    def test_cli_diff_pack_command(self):
        first = os.path.join(self.tmp.name, "diff-a.db")
        second = os.path.join(self.tmp.name, "diff-b.db")
        self._seed(first, ["Task A started", "Task B started"])
        self._seed(second, ["Task A started", "Task C started"])

        path_a = os.path.join(self.pack_dir, "diff-a.brain")
        path_b = os.path.join(self.pack_dir, "diff-b.brain")
        self._capture(cli.cmd_pack, topic="task", output=path_a, db=first)
        self._capture(cli.cmd_pack, topic="task", output=path_b, db=second)

        output = self._capture(cli.cmd_pack, diff=[path_a, path_b])
        self.assertIn("Brain Pack Diff", output)

    def test_cli_list_pack_command(self):
        self._seed(self.db, ["I like coffee", "I like tea"])
        path_a = os.path.join(self.pack_dir, "list-a.brain")
        self._capture(cli.cmd_pack, topic="coffee", output=path_a, db=self.db)

        output = self._capture(cli.cmd_pack, list=True, db=self.db, pack_dir=self.pack_dir)
        self.assertIn("Saved brain packs", output)
        self.assertIn(os.path.basename(path_a), output)


if __name__ == "__main__":
    unittest.main()
