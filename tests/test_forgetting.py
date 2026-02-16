#!/usr/bin/env python3
"""Tests for forgetting policy and conflict-aware recall visibility."""

import json
import time
import unittest

from synapse import Synapse


class TestForgettingAndConflictAwareRecall(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    @staticmethod
    def _to_timestamp_days_ago(days: float) -> float:
        return time.time() - (days * 86400.0)

    def test_recall_show_disputes_includes_conflicts(self):
        vegan = self.s.remember("I am vegan", deduplicate=False)
        vegetarian = self.s.remember("I am vegetarian", deduplicate=False)

        recalled = self.s.recall(context="", limit=2, show_disputes=True)
        found = {memory.id: memory.disputes for memory in recalled}

        self.assertIn(vegan.id, found)
        self.assertIn(vegetarian.id, found)
        self.assertEqual(len(found[vegan.id]), 1)
        dispute = found[vegan.id][0]
        self.assertEqual(dispute["memory_id"], vegetarian.id)
        self.assertIn("text", dispute)
        self.assertEqual(dispute["kind"], "mutual_exclusion")
        self.assertIsInstance(dispute["confidence"], float)

    def test_recall_show_disputes_default_is_silent(self):
        self.s.remember("I am vegan", deduplicate=False)
        self.s.remember("I am vegetarian", deduplicate=False)

        recalled = self.s.recall(context="", limit=2)
        for memory in recalled:
            self.assertEqual(memory.disputes, [])

    def test_recall_exclude_conflicted(self):
        conflict_a = self.s.remember("I am vegan", deduplicate=False)
        conflict_b = self.s.remember("I am vegetarian", deduplicate=False)
        neutral = self.s.remember("I enjoy running", deduplicate=False)

        recalled = self.s.recall(limit=5, exclude_conflicted=True)
        recalled_ids = {memory.id for memory in recalled}

        self.assertNotIn(conflict_a.id, recalled_ids)
        self.assertNotIn(conflict_b.id, recalled_ids)
        self.assertIn(neutral.id, recalled_ids)

    def test_forget_topic_removes_related_memories(self):
        coffee = self.s.remember("I enjoy coffee", deduplicate=False)
        science = self.s.remember("I enjoy science", deduplicate=False)
        self.s.link(coffee.id, science.id, "supports")

        report = self.s.forget_topic("coffee")

        self.assertIn(coffee.id, report["deleted_ids"])
        self.assertNotIn(coffee.id, self.s.concept_graph.memory_concepts)
        self.assertNotIn(coffee.id, self.s.store.memories)
        self.assertIn(science.id, self.s.store.memories)

        for source_id, edges in self.s.edge_graph.edges.items():
            for edge in edges:
                self.assertNotEqual(edge.target_id, coffee.id)

    def test_forget_cascades_to_beliefs_and_contradictions(self):
        positive = self.s.remember("I like tea", deduplicate=False)
        negative = self.s.remember("I do not like tea", deduplicate=False)
        neutral = self.s.remember("I like coffee", deduplicate=False)
        self.s.link(neutral.id, negative.id, "supports")

        self.assertTrue(self.s.contradictions())
        self.assertTrue(self.s.beliefs())

        self.s.forget(negative.id)

        self.assertIsNone(self.s.store.memories.get(negative.id))
        self.assertEqual(self.s.contradictions(), [])
        current_versions = self.s.beliefs()
        memory_ids = {version.memory_id for version in current_versions.values()}
        self.assertNotIn(negative.id, memory_ids)
        self.assertTrue(all(not e.target_id == negative.id for edges in self.s.edge_graph.edges.values() for e in edges))
        self.assertEqual(self.s.triple_index.get_triples_for_memory(negative.id), [])

    def test_redact_preserves_provenance(self):
        anchor = self.s.remember("Project note", deduplicate=False)
        secret = self.s.remember(
            "Secret token is ABC123",
            deduplicate=False,
            metadata={"tags": ["redact"], "owner": "alice"},
        )
        self.s.link(anchor.id, secret.id, "supports")
        before = self.s.store.memories[secret.id]

        response = self.s.redact(secret.id)

        after = self.s.store.memories[secret.id]
        self.assertTrue(response["redacted"])
        self.assertEqual(after["content"], "[REDACTED]")
        self.assertEqual(before["created_at"], after["created_at"])
        self.assertEqual(before["last_accessed"], after["last_accessed"])
        self.assertTrue(
            any(edge.target_id == secret.id for edge in self.s.edge_graph.get_outgoing_edges(anchor.id))
        )
        self.assertNotEqual(before["content"], after["content"])
        after_metadata = json.loads(after["metadata"])
        self.assertEqual(after_metadata["owner"], "alice")
        self.assertEqual(after_metadata["tags"], ["redact"])

    def test_gdpr_delete_by_user_id(self):
        alice = self.s.remember(
            "Alice memory one",
            deduplicate=False,
            metadata={"tags": ["user:alice", "topic"]},
        )
        bob = self.s.remember(
            "Bob memory one",
            deduplicate=False,
            metadata={"tags": ["user:bob", "topic"]},
        )

        report = self.s.gdpr_delete(user_id="alice")

        self.assertIn(alice.id, report["deleted_ids"])
        self.assertNotIn(alice.id, self.s.store.memories)
        self.assertIn(bob.id, self.s.store.memories)

    def test_gdpr_delete_by_concept(self):
        with_quantum = self.s.remember("Quantum experiments are running", deduplicate=False)
        plain = self.s.remember("Weather is fine", deduplicate=False)

        report = self.s.gdpr_delete(concept="quantum")
        self.assertIn(with_quantum.id, report["deleted_ids"])
        self.assertNotIn(with_quantum.id, self.s.store.memories)
        self.assertIn(plain.id, self.s.store.memories)

    def test_apply_ttl_policy_deletes_old_memory(self):
        old_memory = self.s.remember("Old note", deduplicate=False)
        fresh_memory = self.s.remember("Fresh note", deduplicate=False)

        self.s.store.update_memory(
            old_memory.id,
            {
                "created_at": self._to_timestamp_days_ago(90),
                "metadata": json.dumps({"ttl_days": 30}),
            },
        )
        self.s.store.update_memory(
            fresh_memory.id,
            {
                "created_at": self._to_timestamp_days_ago(1),
            },
        )

        report = self.s.forgetting_policy.apply_ttl(default_ttl_days=60)

        self.assertIn(old_memory.id, report["deleted_ids"])
        self.assertNotIn(old_memory.id, self.s.store.memories)
        self.assertIn(fresh_memory.id, self.s.store.memories)

    def test_set_retention_rules_archive_action(self):
        target = self.s.remember("Archive candidate", deduplicate=False)
        survivor = self.s.remember("Archive keep", deduplicate=False)

        self.s.store.update_memory(
            target.id,
            {
                "created_at": self._to_timestamp_days_ago(45),
                "metadata": json.dumps({"tags": ["ephemeral"]}),
            },
        )
        self.s.store.update_memory(
            survivor.id,
            {
                "created_at": self._to_timestamp_days_ago(1),
                "metadata": json.dumps({"tags": ["ephemeral"]}),
            },
        )

        report = self.s.set_retention_rules([
            {"tag": "ephemeral", "older_than_days": 30, "action": "archive"},
        ])

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["archived_count"], 1)
        self.assertIn(target.id, report[0]["archived_ids"])
        archived = self.s.store.memories[target.id]
        archived_meta = json.loads(archived["metadata"])
        self.assertTrue(archived_meta.get("archived", False))
        self.assertIn(survivor.id, self.s.store.memories)

    def test_set_retention_rules_delete_action(self):
        target = self.s.remember("Delete candidate", deduplicate=False)
        self.s.store.update_memory(
            target.id,
            {
                "created_at": self._to_timestamp_days_ago(120),
                "metadata": json.dumps({"tags": ["expired"]}),
                "access_count": 0,
            },
        )
        report = self.s.set_retention_rules([
            {"tag": "expired", "older_than_days": 30, "action": "delete", "min_access": 0},
        ])

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["deleted_count"], 1)
        self.assertIn(target.id, report[0]["deleted_ids"])
        self.assertNotIn(target.id, self.s.store.memories)


if __name__ == "__main__":
    unittest.main()
