#!/usr/bin/env python3

import time
import unittest

from context_pack import ContextPack
from synapse import Synapse


class TestContextPack(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def _remember(self, content):
        return self.s.remember(content, deduplicate=False)

    def test_compile_context_returns_context_pack(self):
        self._remember("User likes dark mode and jazz music.")
        self._remember("User used a vegetarian diet during travel.")
        pack = self.s.compile_context("user", budget=1200, policy="balanced")

        self.assertIsInstance(pack, ContextPack)
        self.assertEqual(pack.query, "user")
        self.assertEqual(pack.budget_total, 1200)
        self.assertTrue(pack.budget_used <= 1200)
        self.assertTrue(pack.to_system_prompt())
        self.assertIn("summaries", pack.to_dict())

    def test_to_system_prompt_renders_cleanly(self):
        self._remember("Python database query was optimized.")
        self._remember("Meeting notes mention python database integration.")
        pack = self.s.compile_context("python database")

        prompt = pack.to_system_prompt()
        self.assertIn("System Context Pack", prompt)
        self.assertIn("query:", prompt)
        self.assertIn("Summaries:", prompt)
        self.assertIn("Top Memories:", prompt)
        self.assertIn("Graph Concepts:", prompt)
        self.assertIn("Evidence:", prompt)

    def test_to_dict_is_full_serializable_shape(self):
        self._remember("Meeting with API review team.")
        self._remember("Python API tests passed.")
        pack = self.s.compile_context("api")

        payload = pack.to_dict()
        self.assertEqual(payload["query"], "api")
        self.assertIsInstance(payload["memories"], list)
        self.assertIsInstance(payload["graph_slice"], dict)
        self.assertIsInstance(payload["summaries"], list)
        self.assertIsInstance(payload["evidence"], list)
        self.assertIsInstance(payload["metadata"], dict)
        self.assertIn("timing_ms", payload["metadata"])

    def test_budget_is_respected(self):
        for idx in range(12):
            self._remember(f"Budget stress test note {idx} mentioning python")
        pack = self.s.compile_context("python", budget=180, policy="balanced")
        compact = pack.to_compact()

        self.assertLessEqual(len(compact), 180)
        self.assertEqual(pack.budget_total, 180)
        self.assertLessEqual(pack.budget_used, 180)

    def test_zero_budget_returns_empty_compact(self):
        self._remember("A tiny preference memory.")
        pack = self.s.compile_context("preference", budget=0, policy="balanced")
        self.assertEqual(pack.budget_total, 0)
        self.assertEqual(pack.budget_used, 0)
        self.assertEqual(pack.to_compact(), "")
        self.assertEqual(pack.memories, [])
        self.assertEqual(pack.graph_slice["nodes"], [])

    def test_policies_produce_different_results(self):
        for idx in range(14):
            self._remember(f"Cross-policy memory {idx} about planning and roadmaps.")
        broad = self.s.compile_context("planning", budget=20000, policy="broad")
        balanced = self.s.compile_context("planning", budget=20000, policy="balanced")
        precise = self.s.compile_context("planning", budget=20000, policy="precise")

        self.assertGreaterEqual(len(broad.memories), len(balanced.memories))
        self.assertGreater(len(balanced.memories), len(precise.memories))
        self.assertLessEqual(len(precise.memories), 4)
        self.assertEqual(len(broad.memories), min(14, 20))

    def test_temporal_policy_prioritizes_chronology(self):
        base = time.time()
        old = self._remember("Temporal memory from 2019")
        mid = self._remember("Temporal memory from 2021")
        recent = self._remember("Temporal memory from 2024")

        for memory, created_at in ((old, base - 1000), (mid, base - 500), (recent, base - 10)):
            self.s.store.memories[memory.id]["created_at"] = created_at
            self.s.store.memories[memory.id]["last_accessed"] = created_at
            self.s.temporal_index.remove_memory(memory.id)
            self.s.temporal_index.add_memory(memory.id, created_at)

        pack = self.s.compile_context("Temporal memory", budget=20000, policy="temporal")
        ids = [memory["id"] for memory in pack.memories if memory["id"] in {old.id, mid.id, recent.id}]
        self.assertEqual(len(ids), 3)
        ordered = [item["created_at"] for item in pack.memories if item["id"] in ids]
        self.assertEqual(ordered, sorted(ordered))

    def test_graph_slice_contains_relevant_concepts(self):
        m1 = self._remember("Python scripts query the database every morning.")
        m2 = self._remember("A meeting discussed a database migration using python.")
        self.s.link(m1.id, m2.id, "supports", 0.9)

        pack = self.s.compile_context("python database", budget=4000)
        graph = pack.graph_slice

        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        self.assertTrue(graph["nodes"])
        self.assertTrue(graph["edges"])
        concept_names = {item["name"] for item in graph.get("concepts", []) if isinstance(item, dict)}
        self.assertIn("python", concept_names)
        self.assertIn("database", concept_names)
        edge = graph["edges"][0]
        self.assertEqual(edge["source_id"], m1.id)
        self.assertEqual(edge["target_id"], m2.id)
        self.assertEqual(edge["relation"], "supports")

    def test_evidence_chains_link_memories(self):
        m1 = self._remember("Feature tests passed for python.")
        m2 = self._remember("Python feature rollout was supported with confidence.")
        self.s.link(m1.id, m2.id, "supports")

        pack = self.s.compile_context("python feature", budget=4000)
        evidence_pairs = [(item["source_id"], item["relation"], item["target_id"]) for item in pack.evidence]
        self.assertIn((m1.id, "supports", m2.id), evidence_pairs)

    def test_to_compact_includes_key_sections(self):
        for idx in range(4):
            self._remember(f"Compact test memory {idx} about software release cadence and testing.")
        pack = self.s.compile_context("software", budget=420)
        compact = pack.to_compact()
        self.assertIn("Query:", compact)
        self.assertIn("Budget:", compact)
        self.assertTrue(len(compact) <= 420)

    def test_unknown_policy_falls_back_to_balanced(self):
        for idx in range(6):
            self._remember(f"Fallback policy memory {idx} about ai")
        pack = self.s.compile_context("ai", policy="not-a-policy")
        self.assertEqual(pack.metadata["policy"], "balanced")
        self.assertTrue(pack.metadata["stats"]["selected_memories"] <= 6)
        self.assertTrue(pack.memories)


if __name__ == "__main__":
    unittest.main()
