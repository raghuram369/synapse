import argparse
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

import cli
from synapse import Synapse


class TestCliDebugCommands(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-cli-debug-")
        self.db = os.path.join(self.tempdir.name, "db")

    def tearDown(self):
        self.tempdir.cleanup()

    def _capture(self, fn, **kwargs):
        buffer = io.StringIO()
        defaults = {
            "db": None,
            "days": None,
            "concept": None,
            "id": None,
            "query": None,
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        with redirect_stdout(buffer):
            fn(args)
        return buffer.getvalue()

    def _new_synapse(self):
        syn = Synapse(self.db)
        syn.should_sleep = lambda: False
        return syn

    def _init_fact_chain(self):
        s = self._new_synapse()
        m1 = s.remember("Task status is open", deduplicate=False)
        m2 = s.remember("Task status is closed", deduplicate=False)
        s.link(m2.id, m1.id, "supersedes", 1.0)

        old_meta = json.loads(s.store.memories[m1.id]['metadata'])
        old_meta['superseded_by'] = m2.id
        old_meta['fact_chain_id'] = 'task-status'
        s.store.update_memory(m1.id, {'metadata': json.dumps(old_meta)})

        new_meta = json.loads(s.store.memories[m2.id]['metadata'])
        new_meta['supersedes'] = m1.id
        new_meta['fact_chain_id'] = 'task-status'
        s.store.update_memory(m2.id, {'metadata': json.dumps(new_meta)})

        s.close()
        return m1, m2

    def _seed(self, entries):
        s = self._new_synapse()
        memories = []
        for text in entries:
            memories.append(s.remember(text, deduplicate=False))
        s.close()
        return memories

    def test_why_includes_score_components_and_concepts(self):
        m1, _ = self._seed(["Alice likes tea", "Alice likes coffee"])
        out = self._capture(cli.cmd_why, id=m1.id, db=self.db)
        self.assertIn("Why memory", out)
        self.assertIn("Score components", out)
        self.assertIn("BM25", out)
        self.assertIn("Concept:", out)
        self.assertIn("Linked concepts", out)

    def test_why_shows_belief_chain(self):
        m1, m2 = self._init_fact_chain()
        out = self._capture(cli.cmd_why, id=m2.id, db=self.db)
        self.assertIn("Belief chain", out)
        self.assertIn(f"#{m1.id}", out)
        self.assertIn(f"#{m2.id}", out)

    def test_graph_shows_hop_neighbors_and_memories(self):
        self._seed([
            "Alice works on project Synapse",
            "Bob works on project Synapse",
            "Synapse tracks memory facts",
        ])
        out = self._capture(cli.cmd_graph, concept="alice", db=self.db)
        self.assertIn("Concept neighborhood: alice", out)
        self.assertIn("1-hop neighbors", out)
        self.assertIn("memories:", out)
        self.assertIn("2-hop neighbors", out)

    def test_conflicts_lists_unresolved_pairs(self):
        self._seed(["The sky is blue", "The sky is not blue"])
        out = self._capture(cli.cmd_conflicts, db=self.db)
        self.assertIn("Active contradictions", out)
        self.assertIn("POLARITY", out)
        self.assertIn("Suggestion:", out)

    def test_beliefs_lists_current_worldview_with_versions(self):
        self._seed(["Alice likes tea", "Alice likes coffee"])
        out = self._capture(cli.cmd_beliefs, db=self.db)
        self.assertIn("Current worldview", out)
        self.assertIn("alice|likes", out)
        self.assertIn("versions", out)

    def test_timeline_filters_by_concept(self):
        memory_a, memory_b, memory_c = self._seed([
            "Alice works on project Synapse",
            "Bob works on project Synapse",
            "Carol prefers tea",
        ])
        out = self._capture(cli.cmd_timeline, concept="project", db=self.db)
        self.assertIn(f"#{memory_a.id}", out)
        self.assertIn(f"#{memory_b.id}", out)
        self.assertNotIn(f"#{memory_c.id}", out)

    def test_timeline_filters_by_days(self):
        old = self._seed(["Alice likes tea"])[0]
        s = self._new_synapse()
        s.store.update_memory(old.id, {
            'created_at': time.time() - (10 * 24 * 3600),
        })
        s.close()
        self._seed(["Bob likes coffee"])

        out = self._capture(cli.cmd_timeline, days=5, db=self.db)
        self.assertIn("Bob likes coffee", out)
        self.assertNotIn("Alice likes tea", out)

    def test_timeline_shows_validity_window(self):
        s = self._new_synapse()
        now = time.time()
        mem = s.remember("Decision is approved", deduplicate=False)
        s.store.update_memory(mem.id, {
            'valid_from': now - 3600,
            'valid_to': now + 3600,
        })
        s.close()
        out = self._capture(cli.cmd_timeline, db=self.db)
        self.assertIn("validity=[", out)

    def test_stats_displays_dashboard_metrics(self):
        self._seed(["The sky is blue", "The sky is not blue", "Carol likes tea"])
        out = self._capture(cli.cmd_stats, db=self.db)
        self.assertIn("Synapse Debug Stats", out)
        self.assertIn("Total memories", out)
        self.assertIn("Contradictions", out)
        self.assertIn("Top 10 hottest concepts", out)
        self.assertIn("Memory age distribution", out)
        self.assertIn("Last sleep", out)


if __name__ == "__main__":
    unittest.main()
