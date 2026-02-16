import unittest
import json

from contradictions import ContradictionDetector
from synapse import Synapse


class TestContradictionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ContradictionDetector()

    def test_detect_polarity_negative(self):
        result = self.detector.detect_polarity(
            "User likes coffee",
            "User does not like coffee",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "polarity")

    def test_detect_polarity_no_conflict(self):
        result = self.detector.detect_polarity(
            "User likes coffee",
            "User likes tea",
        )
        self.assertIsNone(result)

    def test_detect_mutual_exclusion(self):
        result = self.detector.detect_mutual_exclusion(
            "User is vegetarian",
            "User is vegan",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "mutual_exclusion")

    def test_detect_mutual_exclusion_requires_overlap(self):
        result = self.detector.detect_mutual_exclusion(
            "User is vegan",
            "User is vegan",
        )
        self.assertIsNone(result)

    def test_detect_numeric_conflict(self):
        result = self.detector.detect_numeric_conflict(
            "User is 25 years old",
            "User is 30 years old",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "numeric_range")
        self.assertIn("age", result.description)

    def test_detect_numeric_no_conflict(self):
        result = self.detector.detect_numeric_conflict(
            "User is 25 years old",
            "User is 25 years old",
        )
        self.assertIsNone(result)

    def test_detect_temporal_conflict(self):
        result = self.detector.detect_temporal_conflict(
            "User is 25 years old",
            "User is 30 years old",
            0,
            60 * 60,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "temporal_conflict")
        self.assertGreater(result.confidence, 0.0)

    def test_scan_memories_detects_contradictions(self):
        memories = [
            {"id": 1, "content": "User is vegan", "created_at": 0.0},
            {"id": 2, "content": "User is vegetarian", "created_at": 1.0},
            {"id": 3, "content": "User likes jazz", "created_at": 2.0},
        ]

        detected = self.detector.scan_memories(memories)
        kinds = {c.kind for c in detected}

        self.assertIn("mutual_exclusion", kinds)
        self.assertNotIn("polarity", kinds)

    def test_scan_memories_only_overlapping_concepts(self):
        memories = [
            {"id": 1, "content": "User likes coffee", "created_at": 0.0},
            {"id": 2, "content": "Project status is active", "created_at": 1.0},
        ]

        detected = self.detector.scan_memories(memories)
        self.assertEqual(detected, [])

    def test_add_exclusive_set(self):
        self.detector.add_exclusive_set({"north", "south", "east", "west"})

        result = self.detector.detect_mutual_exclusion(
            "Heading north",
            "Heading south",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "mutual_exclusion")

    def test_check_new_memory_detects_with_existing(self):
        existing = [
            {"id": 1, "content": "User is vegetarian", "created_at": 1.0},
        ]

        detected = self.detector.check_new_memory(
            "User is vegan",
            existing,
            new_memory_id=2,
            new_time=2.0,
        )

        self.assertTrue(len(detected) >= 1)
        pair_ids = {detected[0].memory_id_a, detected[0].memory_id_b}
        self.assertEqual(pair_ids, {1, 2})
        self.assertTrue(any(c.kind in {"mutual_exclusion", "temporal_conflict"} for c in detected))


class TestSynapseContradictionsIntegration(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def test_check_new_memory_registers_on_remember(self):
        first = self.s.remember("User is vegetarian", deduplicate=False)
        second = self.s.remember("User is vegan", deduplicate=False)

        contradictions = self.s.contradictions()

        self.assertTrue(len(contradictions) >= 1)
        self.assertEqual(contradictions[0].kind, "mutual_exclusion")
        self.assertEqual({contradictions[0].memory_id_a, contradictions[0].memory_id_b}, {first.id, second.id})

    def test_scan_memories_on_detector_returns_results(self):
        self.s.remember("I am vegetarian", deduplicate=False)
        self.s.remember("I am vegan", deduplicate=False)

        contradictions = self.s.contradiction_detector.scan_memories(
            [self.s._memory_data_to_object(memory_data) for memory_data in self.s.store.memories.values()]
        )

        kinds = {c.kind for c in contradictions}
        self.assertIn("mutual_exclusion", kinds)

    def test_contradictions_method_returns_unresolved(self):
        a = self.s.remember("User is married", deduplicate=False)
        self.s.remember("User is single", deduplicate=False)

        all_contradictions = self.s.contradictions()
        self.assertTrue(len(all_contradictions) >= 1)

        while self.s.contradictions():
            current = self.s.contradictions()
            self.s.resolve_contradiction(0, winner_memory_id=current[0].memory_id_a)

        unresolved = self.s.contradictions()
        self.assertEqual(unresolved, [])

    def test_resolve_contradiction_supersedes_loser(self):
        a = self.s.remember("User is married", deduplicate=False)
        b = self.s.remember("User is single", deduplicate=False)

        contradictions = self.s.contradictions()
        mutual_idx = next(
            idx for idx, contradiction in enumerate(contradictions)
            if contradiction.kind == "mutual_exclusion"
        )
        target = contradictions[mutual_idx]

        self.s.resolve_contradiction(mutual_idx, winner_memory_id=target.memory_id_a)

        loser_id = target.memory_id_b
        loser_memory = self.s.store.memories[loser_id]
        loser_metadata = json.loads(loser_memory["metadata"])
        self.assertEqual(loser_metadata.get("superseded_by"), target.memory_id_a)
        self.assertIn("superseded_at", loser_metadata)

        while self.s.contradictions():
            current = self.s.contradictions()
            self.s.resolve_contradiction(0, winner_memory_id=current[0].memory_id_a)

        contradictions = self.s.contradictions()
        self.assertEqual(contradictions, [])

    def test_conflict_aware_recall_downweights_contradicted_memories(self):
        neutral = self.s.remember("I like tea", deduplicate=False)
        self.s.store.update_memory(neutral.id, {"strength": 1.0})

        pos = self.s.remember("User likes coffee", deduplicate=False)
        neg = self.s.remember("User does not like coffee", deduplicate=False)

        self.s.store.update_memory(pos.id, {"strength": 6.0})
        self.s.store.update_memory(neg.id, {"strength": 6.0})

        baseline = self.s.recall(limit=3)
        conflict_free = self.s.recall(limit=3, conflict_aware=True)

        self.assertIn(neutral.id, [m.id for m in baseline])
        self.assertIn(neutral.id, [m.id for m in conflict_free])

        self.assertIn(pos.id, {baseline[0].id, baseline[1].id})
        self.assertIn(neg.id, {baseline[0].id, baseline[1].id})
        self.assertNotIn(conflict_free[0].id, {pos.id, neg.id})
        self.assertEqual(conflict_free[0].id, neutral.id)


if __name__ == "__main__":
    unittest.main()
