import unittest

from belief import BeliefTracker
from synapse import Synapse


class _DummyMemory:
    def __init__(self, memory_id, content, created_at=0.0, valid_from=None):
        self.id = memory_id
        self.content = content
        self.created_at = created_at
        self.valid_from = valid_from


class TestBeliefTracker(unittest.TestCase):
    def test_tracks_new_fact_version(self):
        tracker = BeliefTracker()
        memory = _DummyMemory(1, "Alice likes tea", created_at=1.0)

        created = tracker.on_remember(memory)

        self.assertEqual(len(created), 1)
        current = tracker.get_current("alice|likes")
        self.assertIsNotNone(current)
        self.assertEqual(current.value, "tea")
        self.assertIsNone(current.valid_to)

    def test_skips_memory_without_id(self):
        tracker = BeliefTracker()
        memory = _DummyMemory(None, "Alice likes tea", created_at=1.0)

        created = tracker.on_remember(memory)

        self.assertEqual(created, [])
        self.assertIsNone(tracker.get_current("alice|likes"))

    def test_closes_previous_and_opens_new(self):
        tracker = BeliefTracker()
        first = _DummyMemory(1, "Alice likes tea", created_at=1.0)
        second = _DummyMemory(2, "Alice likes coffee", created_at=2.0)

        tracker.on_remember(first)
        tracker.on_remember(second)

        history = tracker.get_history("alice|likes")
        self.assertEqual(len(history), 2)
        self.assertIsNotNone(history[0].valid_to)
        self.assertEqual(history[0].valid_to, 2.0)
        self.assertIsNone(history[1].valid_to)
        self.assertEqual(history[1].value, "coffee")

    def test_polarity_flip_marks_user_correction(self):
        tracker = BeliefTracker()
        first = _DummyMemory(1, "Alice likes tea", created_at=1.0)
        second = _DummyMemory(2, "Alice does not like tea", created_at=2.0)

        tracker.on_remember(first)
        created = tracker.on_remember(second)

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].reason, "user correction")
        self.assertEqual(created[0].value, "tea")

    def test_mutual_exclusion_marks_contradiction_resolved(self):
        tracker = BeliefTracker()
        first = _DummyMemory(1, "Alice is vegetarian", created_at=1.0)
        second = _DummyMemory(2, "Alice is vegan", created_at=2.0)

        tracker.on_remember(first)
        created = tracker.on_remember(second)

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].reason, "contradiction resolved")
        self.assertEqual(created[0].value, "vegan")

    def test_identical_followup_does_not_add_version(self):
        tracker = BeliefTracker()
        first = _DummyMemory(1, "Alice is vegetarian", created_at=1.0)
        second = _DummyMemory(2, "Alice is vegetarian", created_at=2.0)

        tracker.on_remember(first)
        tracker.on_remember(second)

        history = tracker.get_history("alice|is")
        self.assertEqual(len(history), 1)

    def test_get_all_current_returns_open_versions_only(self):
        tracker = BeliefTracker()
        tracker.on_remember(_DummyMemory(1, "Alice likes tea", created_at=1.0))
        tracker.on_remember(_DummyMemory(2, "Alice likes coffee", created_at=2.0))
        current = tracker.get_all_current()

        self.assertIn("alice|likes", current)
        self.assertEqual(current["alice|likes"].value, "coffee")

    def test_get_matching_history(self):
        tracker = BeliefTracker()
        tracker.on_remember(_DummyMemory(1, "Alice likes tea", created_at=1.0))
        tracker.on_remember(_DummyMemory(2, "Alice works at acme", created_at=2.0))

        matches = tracker.get_matching_history("likes")
        self.assertTrue(any(item.fact_key == "alice|likes" for item in matches))
        self.assertFalse(any(item.fact_key == "alice|works_at" for item in matches))

    def test_rebuild_from_unsorted_memories(self):
        tracker = BeliefTracker()
        later = _DummyMemory(2, "Alice likes coffee", created_at=20.0)
        earlier = _DummyMemory(1, "Alice likes tea", created_at=10.0)

        tracker.rebuild([later, earlier])
        history = tracker.get_history("alice|likes")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].value, "tea")
        self.assertEqual(history[1].value, "coffee")


class TestSynapseBeliefs(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def test_synepse_beliefs(self):
        first = self.s.remember("Alice likes tea", deduplicate=False)
        second = self.s.remember("Alice likes coffee", deduplicate=False)

        beliefs = self.s.beliefs()
        self.assertIn("alice|likes", beliefs)
        self.assertEqual(beliefs["alice|likes"].memory_id, second.id)
        self.assertEqual(beliefs["alice|likes"].value, "coffee")

    def test_synapse_belief_history(self):
        self.s.remember("Alice is vegan", deduplicate=False)
        self.s.remember("Alice is vegetarian", deduplicate=False)

        history = self.s.belief_history("alice")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1].fact_key, "alice|is")
        self.assertNotEqual(history[0].value, history[1].value)


if __name__ == "__main__":
    unittest.main()
