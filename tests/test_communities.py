#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from communities import Community, CommunityDetector
from context_pack import ContextPack
from sleep import SleepReport
from synapse import Synapse


class _DummyMemory:
    def __init__(self, content: str):
        self.content = content


class _FakeConceptGraph:
    def __init__(self, memory_concepts, concepts=None):
        self.memory_concepts = {idx: set(values) for idx, values in memory_concepts.items()}
        if concepts is None:
            concepts = set()
            for values in self.memory_concepts.values():
                concepts.update(values)
        self.concepts = {name: object() for name in set(concepts)}


class TestCommunityDetector(unittest.TestCase):
    def test_detect_two_disjoint_communities(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph(
            {
                1: {"apple", "banana", "cherry"},
                2: {"apple", "banana", "cherry"},
                3: {"delta", "echo", "foxtrot"},
                4: {"delta", "echo", "foxtrot"},
            }
        )

        communities = detector.detect_communities(graph, min_size=3)
        self.assertEqual(len(communities), 2)
        member_sets = {frozenset(c.concepts) for c in communities}
        self.assertIn(frozenset({"apple", "banana", "cherry"}), member_sets)
        self.assertIn(frozenset({"delta", "echo", "foxtrot"}), member_sets)

    def test_detect_single_triangle_community(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph(
            {
                1: {"python", "database", "performance"},
                2: {"python", "database", "performance"},
            }
        )

        communities = detector.detect_communities(graph, min_size=3)
        self.assertEqual(len(communities), 1)
        self.assertEqual(communities[0].concepts, {"database", "performance", "python"})
        self.assertGreaterEqual(len(communities[0].hub_concepts), 1)
        self.assertGreater(communities[0].density, 0.0)

    def test_detect_filters_communities_smaller_than_min_size(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph({1: {"weather", "forecast"}})

        communities = detector.detect_communities(graph)
        self.assertEqual(communities, [])

        communities_two = detector.detect_communities(graph, min_size=2)
        self.assertEqual(len(communities_two), 1)
        self.assertEqual(communities_two[0].concepts, {"weather", "forecast"})

    def test_incremental_adds_new_node_to_existing_community(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph(
            {
                1: {"apple", "banana", "cherry"},
            }
        )
        communities = detector.detect_communities(graph, min_size=3)
        self.assertEqual(len(communities), 1)

        updated = detector.incremental_update(
            communities=communities,
            new_nodes=["dragon"],
            new_edges=[("cherry", "dragon", 1.0)],
        )
        self.assertEqual(len(updated), 1)
        self.assertEqual(len(updated[0].concepts), 4)
        self.assertIn("dragon", updated[0].concepts)

    def test_incremental_grows_existing_cluster_without_merging(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph(
            {
                1: {"apple", "banana", "cherry"},
                2: {"apple", "banana", "cherry"},
                3: {"delta", "echo", "foxtrot"},
                4: {"delta", "echo", "foxtrot"},
            }
        )
        communities = detector.detect_communities(graph, min_size=3)
        self.assertEqual(len(communities), 2)

        updated = detector.incremental_update(
            communities=communities,
            new_nodes=["golf"],
            new_edges=[
                ("delta", "golf", 1.0),
                ("echo", "golf", 1.0),
                ("foxtrot", "golf", 1.0),
            ],
        )

        self.assertEqual(len(updated), 2)
        self.assertTrue(any("golf" in c.concepts for c in updated))

    def test_incremental_no_change_is_stable(self):
        detector = CommunityDetector()
        graph = _FakeConceptGraph({1: {"alpha", "beta", "gamma"}})
        communities = detector.detect_communities(graph, min_size=3)

        updated = detector.incremental_update(
            communities=communities,
            new_nodes=[],
            new_edges=[],
        )

        self.assertEqual({frozenset(c.concepts) for c in communities}, {frozenset(c.concepts) for c in updated})
        self.assertEqual(len(updated), 1)


class TestCommunitySummary(unittest.TestCase):
    def test_summary_generates_key_facts(self):
        community = Community(
            id=1,
            concepts={"vegetarian", "food"},
            hub_concepts=["food", "vegetarian"],
            density=0.8,
            label="Food & Diet",
        )
        memories = [
            _DummyMemory("User is vegetarian and prefers Italian cuisine."),
            _DummyMemory("User is allergic to peanuts."),
        ]

        summary = community.summary(memories)
        self.assertTrue(summary.startswith("Food & Diet:"))
        self.assertIn("vegetarian", summary.lower())
        self.assertIn("peanuts", summary.lower())

    def test_summary_falls_back_when_no_matching_memories(self):
        community = Community(
            id=1,
            concepts={"food", "diet"},
            hub_concepts=["food"],
            density=0.5,
            label="Food & Diet",
        )
        memories = [
            _DummyMemory("The project schedule was moved."),
            _DummyMemory("We discussed infrastructure this quarter."),
        ]

        summary = community.summary(memories)
        self.assertEqual(summary, "Food & Diet: no explicit details available")

    def test_summary_deduplicates_repeated_facts(self):
        community = Community(
            id=1,
            concepts={"fitness"},
            hub_concepts=["fitness"],
            density=1.0,
            label="Fitness",
        )
        memory = _DummyMemory("Fitness tracking shows user has daily morning workouts.")
        summary = community.summary([memory, memory])
        self.assertEqual(summary, "Fitness: Fitness tracking shows user has daily morning workouts.")


class TestSynapseCommunityIntegration(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def _seed_memories(self) -> None:
        self.s.remember("I built a python database migration script.")
        self.s.remember("The python database team reviewed query optimization.")
        self.s.remember("A performance test verified database reads from python jobs.")

    def test_synapse_communities(self):
        self._seed_memories()
        communities = self.s.communities()
        self.assertTrue(communities)
        self.assertTrue(any("python" in c.concepts for c in communities))
        self.assertTrue(any("database" in c.concepts for c in communities))

    def test_synapse_community_summary(self):
        self._seed_memories()
        summary = self.s.community_summary("python")
        self.assertIsInstance(summary, str)
        self.assertNotEqual(summary, "No community summary available")
        self.assertIn(":", summary)

    def test_synapse_forget_invalidates_community_cache(self):
        first = self.s.remember("I configured python database credentials.")
        second = self.s.remember("I benchmarked database writes from python.")
        _ = self.s.communities()
        self.assertFalse(self.s._communities_dirty)

        self.s.forget(first.id)
        self.assertTrue(self.s._communities_dirty)

        refreshed = self.s.communities()
        self.assertIsInstance(refreshed, list)
        summary = self.s.community_summary("python")
        self.assertIsInstance(summary, str)

    def test_context_pack_includes_community_summaries(self):
        self._seed_memories()
        with patch.object(
            self.s,
            "community_summary",
            side_effect=lambda concept: f"Community: {concept}",
        ):
            pack = self.s.compile_context("python", policy="balanced", budget=5000)

        self.assertIsInstance(pack, ContextPack)
        self.assertTrue(any(item.startswith("Community:") for item in pack.summaries))

    def test_sleep_auto_refreshes_communities(self):
        self._seed_memories()
        _ = self.s.communities()
        self.assertFalse(self.s._communities_dirty)

        self.s.remember("Python scripts improve database backups.")
        self.assertTrue(self.s._communities_dirty)

        with patch.object(
            self.s.sleep_runner,
            "sleep",
            return_value=SleepReport(
                consolidated=0,
                promoted=0,
                patterns_found=0,
                contradictions=0,
                pruned=0,
                graph_cleaned=0,
                duration_ms=0.0,
                details={"status": "patched"},
            ),
        ):
            self.s.sleep(verbose=False)

        self.assertFalse(self.s._communities_dirty)
        communities = self.s.communities()
        self.assertTrue(communities)


if __name__ == "__main__":
    unittest.main()
