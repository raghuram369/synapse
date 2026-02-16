#!/usr/bin/env python3

import os
import tempfile
import time
import unittest

from synapse import Synapse
from command_dsl import MemoryCommandParser


class TestMemoryCommandParser(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-command-dsl-")
        self.db = os.path.join(self.tempdir.name, "db")
        self.s = Synapse(self.db)
        self.parser = MemoryCommandParser(self.s)

    def tearDown(self):
        self.s.close()
        self.tempdir.cleanup()

    def test_is_memory_command(self):
        self.assertTrue(self.parser.is_memory_command('/mem recall what') )
        self.assertTrue(self.parser.is_memory_command('/mem recall "hello"'))
        self.assertFalse(self.parser.is_memory_command('hello /mem recall'))

    def test_help_lists_supported_commands(self):
        out = self.parser.parse_and_execute('/mem help')

        self.assertIn('/mem help', out)
        self.assertIn('/mem remember', out)
        self.assertIn('/mem recall', out)
        self.assertIn('/mem search', out)

    def test_remember_with_quoted_and_unquoted_text(self):
        out = self.parser.parse_and_execute('/mem remember "User likes pizza"')
        self.assertIn('Stored memory', out)

        out2 = self.parser.parse_and_execute('/mem remember User likes tacos')
        self.assertIn('Stored memory', out2)

        results = self.s.recall('pizza', limit=1)
        self.assertEqual(len(results), 1)
        self.assertIn('pizza', results[0].content)

    def test_recall_command_formats_top_results(self):
        self.s.remember('Alice likes tea')
        self.s.remember('Alice likes coffee')

        out = self.parser.parse_and_execute('/mem recall Alice likes')

        self.assertIn('🔎 Recall', out)
        self.assertIn('Alice', out)

    def test_pack_command_formats_context_pack(self):
        for idx in range(6):
            self.s.remember(f'Project planning note {idx}')

        out = self.parser.parse_and_execute('/mem pack "Project planning" 180')

        self.assertIn('Context Pack', out)
        self.assertIn('Query: Project planning', out)

    def test_search_command_includes_explain_breakdown(self):
        self.s.remember('Alice enjoys biking in summer')

        out = self.parser.parse_and_execute('/mem search "Alice enjoys biking"')

        self.assertIn('🔎 Recall', out)
        self.assertIn('Breakdown:', out)

    def test_rewind_command_filters_delta_window(self):
        first = self.s.remember('Team mood is positive', deduplicate=False)
        second = self.s.remember('Team mood is neutral', deduplicate=True)

        out = self.parser.parse_and_execute('/mem rewind 30 Team')

        self.assertIn('Rewind', out)
        self.assertIn(f"#{first.id}", out)
        self.assertIn(f"#{second.id}", out)

    def test_contradict_command_filters_active_contradictions(self):
        self.s.remember('The sky is blue', deduplicate=False)
        self.s.remember('The sky is not blue', deduplicate=False)

        out = self.parser.parse_and_execute('/mem contradict sky')

        self.assertIn('Active contradictions', out)
        self.assertIn('confidence', out)

    def test_history_command(self):
        first = self.s.remember('User prefers dark mode', deduplicate=False)
        self.s.remember('User prefers light mode', deduplicate=True)

        out = self.parser.parse_and_execute('/mem history "user prefers"')

        self.assertIn('Fact history', out)
        self.assertIn(f"#{first.id}", out)

    def test_timeline_command_with_days_filter(self):
        now = time.time()
        old = self.s.remember('Alice works on backend', deduplicate=False)
        self.s.remember('Alice works on frontend', deduplicate=True)
        self.s.store.update_memory(old.id, {
            'created_at': now - (10 * 86400),
            'last_accessed': now - (10 * 86400),
        })
        self.s.temporal_index.remove_memory(old.id)
        self.s.temporal_index.add_memory(old.id, now - (10 * 86400))

        out = self.parser.parse_and_execute('/mem timeline 1 Alice')

        self.assertIn('Timeline', out)
        self.assertNotIn(f"#{old.id}", out)

    def test_stats_command_summary(self):
        self.s.remember('Alice likes Python')

        out = self.parser.parse_and_execute('/mem stats')

        self.assertIn('Store statistics', out)
        self.assertIn('Memories:', out)
        self.assertIn('Concepts:', out)

    def test_forget_command_removes_memories_by_topic(self):
        self.s.remember('Alice owns a red car')

        out = self.parser.parse_and_execute('/mem forget red')

        self.assertIn('Forget complete', out)
        self.assertEqual(len(self.s.recall('red car', limit=3)), 0)

    def test_sleep_command(self):
        out = self.parser.parse_and_execute('/mem sleep')

        self.assertIn('Sleep cycle complete', out)

    def test_export_command_creates_file(self):
        output_file = os.path.join(self.tempdir.name, 'snapshot')
        out = self.parser.parse_and_execute(f'/mem export {output_file}')

        self.assertIn('Exported memory store to', out)
        self.assertTrue(os.path.exists(output_file + '.synapse'))

    def test_unknown_command_suggests_closest_match(self):
        out = self.parser.parse_and_execute('/mem recal "query"')

        self.assertIn('Did you mean /mem recall', out)

    def test_synapse_command_wrapper(self):
        out = self.s.command('/mem help')

        self.assertIn('/mem recall', out)


if __name__ == '__main__':
    unittest.main()
