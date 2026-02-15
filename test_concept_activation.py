#!/usr/bin/env python3

import time
import unittest

from synapse import Synapse


class TestConceptActivation(unittest.TestCase):

    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def _age_memory(self, mid: int, *, days: float, last_accessed_days: float | None = None):
        now = time.time()
        created = now - days * 86400.0
        accessed = now - (last_accessed_days * 86400.0) if last_accessed_days is not None else created
        self.s.store.memories[mid]['created_at'] = created
        self.s.store.memories[mid]['last_accessed'] = accessed
        # temporal index uses created_at
        self.s.temporal_index.remove_memory(mid)
        self.s.temporal_index.add_memory(mid, created)

    def test_concept_activation_increases_on_recall(self):
        food = self.s.remember("My preferences: vegetarian food.")
        other = self.s.remember("My preferences: strength training.")

        # Make both old
        self._age_memory(food.id, days=200)
        self._age_memory(other.id, days=200)

        # Recall a few times, limited so only food memory is returned/activated.
        for _ in range(5):
            res = self.s.recall("dietary preferences", limit=1, min_strength=1e-9)
            self.assertEqual(res[0].id, food.id)

        concepts = self.s.concept_graph.get_memory_concepts(food.id)
        self.assertIn("food", concepts)
        food_node = self.s.concept_graph.concepts["food"]
        self.assertGreaterEqual(food_node.activation_count, 5)
        self.assertGreater(self.s.concept_graph.concept_activation_strength("food"), 0.0)

    def test_concept_activation_boosts_old_memory(self):
        food = self.s.remember("My preferences: vegetarian food.")
        other = self.s.remember("My preferences: strength training.")
        self._age_memory(food.id, days=200)
        self._age_memory(other.id, days=200)

        # Activate food concept via targeted recalls.
        for _ in range(5):
            self.s.recall("dietary preferences", limit=1, min_strength=1e-9)

        # Query that matches both lexically; food should rank higher due to concept activation.
        res = self.s.recall("preferences", limit=2, explain=True, min_strength=1e-9)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].id, food.id)
        self.assertGreater(res[0].score_breakdown.concept_activation_score,
                           res[1].score_breakdown.concept_activation_score)

    def test_hot_concepts(self):
        food = self.s.remember("I prefer vegetarian food")
        self._age_memory(food.id, days=200)
        for _ in range(3):
            self.s.recall("diet", limit=1, min_strength=1e-9)
        hot = self.s.hot_concepts(k=5)
        self.assertTrue(hot)
        names = [n for n, _ in hot]
        self.assertIn("food", names)

    def test_concept_activation_decay(self):
        food = self.s.remember("vegetarian food")
        self.s.recall("diet", limit=1, min_strength=1e-9)

        now = time.time()
        node = self.s.concept_graph.concepts["food"]
        initial = self.s.concept_graph.concept_activation_strength("food", now=now)

        # Simulate time passing (2 half-lives)
        node.last_activated = now - (self.s.concept_graph.CONCEPT_ACTIVATION_HALF_LIFE_SECS * 2)
        decayed = self.s.concept_graph.concept_activation_strength("food", now=now)
        self.assertLess(decayed, initial)

    def test_prune_dry_run_and_actual(self):
        cold = self.s.remember("Totally unrelated note about something")
        hot = self.s.remember("My preferences: vegetarian food")

        # Make both old (created long ago) but last_accessed recently enough
        # that recall can still find them (effective_strength must exceed min_strength).
        self._age_memory(cold.id, days=200, last_accessed_days=1)
        self._age_memory(hot.id, days=200, last_accessed_days=1)
        self.s.store.memories[cold.id]['strength'] = 0.05
        self.s.store.memories[hot.id]['strength'] = 0.05
        # Reset access counts that remember's dedup may have set
        self.s.store.memories[cold.id]['access_count'] = 0
        self.s.store.memories[hot.id]['access_count'] = 0

        # Activate food concept so the hot memory should be protected.
        for _ in range(5):
            self.s.recall("dietary preferences", limit=1, min_strength=1e-9)

        # Now age last_accessed far back and drop strength so prune criteria trigger
        self._age_memory(cold.id, days=200, last_accessed_days=200)
        self._age_memory(hot.id, days=200, last_accessed_days=200)
        self.s.store.memories[cold.id]['strength'] = 0.01
        self.s.store.memories[hot.id]['strength'] = 0.01
        self.s.store.memories[cold.id]['access_count'] = 0
        self.s.store.memories[hot.id]['access_count'] = 0

        would_prune = self.s.prune(min_strength=0.1, min_access=0, max_age_days=90, dry_run=True)
        self.assertIn(cold.id, would_prune)
        self.assertNotIn(hot.id, would_prune)
        self.assertIn(cold.id, self.s.store.memories)  # dry run should not delete

        pruned = self.s.prune(min_strength=0.1, min_access=0, max_age_days=90, dry_run=False)
        self.assertIn(cold.id, pruned)
        self.assertNotIn(cold.id, self.s.store.memories)


if __name__ == '__main__':
    unittest.main()
