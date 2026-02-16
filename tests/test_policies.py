"""Tests for memory policy presets, PII detection, and retention behavior."""

import json
import time
import unittest

from synapse import Synapse


class TestPolicyPresets(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")
        self.s.should_sleep = lambda: False

    def tearDown(self):
        self.s.close()

    @staticmethod
    def _to_days_ago(days: float) -> float:
        return time.time() - (days * 86400.0)

    def _metadata_tags(self, memory_id: int):
        return json.loads(self.s.store.memories[memory_id]["metadata"]).get("tags", [])

    def test_policy_presets_include_required_entries(self):
        presets = self.s.list_presets()
        self.assertEqual(set(presets.keys()), {"minimal", "private", "work", "ephemeral"})
        for name, payload in presets.items():
            self.assertIn("description", payload)
            self.assertIsInstance(payload["ttl_days"], int)

    def test_policy_show_none_when_unset(self):
        self.assertIsNone(self.s.policy())

    def test_apply_known_policy_and_readback(self):
        minimal = self.s.policy("minimal")
        self.assertEqual(minimal["name"], "minimal")
        self.assertEqual(minimal["ttl_days"], 30)
        self.assertEqual(self.s.get_active_policy()["name"], "minimal")

        private = self.s.policy("private")
        self.assertEqual(private["name"], "private")
        self.assertEqual(self.s.get_active_policy()["name"], "private")

    def test_apply_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            self.s.policy("nonexistent")

    def test_detect_pii_includes_email_and_phone(self):
        findings = self.s.policy_manager.detect_pii(
            "Reach me at user@example.com or +1 (415) 555-2671."
        )
        types = {item["type"] for item in findings}
        self.assertIn("email", types)
        self.assertIn("phone", types)
        self.assertTrue(any(item["value"] == "user@example.com" for item in findings))

    def test_detect_pii_includes_ssn_and_credit_card(self):
        findings = self.s.policy_manager.detect_pii(
            "SSN 123-45-6789 and card 4111 1111 1111 1111."
        )
        self.assertEqual(
            {item["type"] for item in findings},
            {"ssn", "credit_card"},
        )

    def test_detect_pii_includes_address(self):
        findings = self.s.policy_manager.detect_pii(
            "Moved to 221B Baker Street yesterday."
        )
        self.assertEqual(findings[0]["type"], "address")

    def test_auto_redact_replaces_all_detected_pii(self):
        redacted = self.s.policy_manager.auto_redact(
            "user@example.com +1 415-555-2671 123-45-6789 4111 1111 1111 1111"
        )
        self.assertIn("[REDACTED-EMAIL]", redacted)
        self.assertIn("[REDACTED-PHONE]", redacted)
        self.assertIn("[REDACTED-SSN]", redacted)
        self.assertIn("[REDACTED-CREDIT_CARD]", redacted)
        self.assertNotIn("4111", redacted)

    def test_auto_redact_respects_pattern_filter(self):
        redacted = self.s.policy_manager.auto_redact(
            "user@example.com and +1 415-555-2671",
            pii_patterns=["email"],
        )
        self.assertIn("[REDACTED-EMAIL]", redacted)
        self.assertIn("415-555-2671", redacted)
        self.assertNotIn("[REDACTED-PHONE]", redacted)

    def test_private_policy_redacts_content_on_remember(self):
        self.s.policy("private")
        memory = self.s.remember(
            "Email alice@example.com and card 4111 1111 1111 1111.",
            deduplicate=False,
        )
        self.assertIn("[REDACTED-", memory.content)
        self.assertNotIn("alice@example.com", memory.content)
        self.assertNotIn("4111 1111 1111 1111", memory.content)

    def test_minimal_policy_keeps_tagged_memories_and_prunes_untagged(self):
        self.s.policy("minimal")
        pref = self.s.remember("I prefer warm climates", memory_type="preference", deduplicate=False)
        commit = self.s.remember("I will commit to finishing this task", deduplicate=False)
        generic = self.s.remember("General project planning ideas", deduplicate=False)

        self.assertIn("preference", self._metadata_tags(pref.id))
        self.assertIn("commitment", self._metadata_tags(commit.id))
        self.assertEqual(self._metadata_tags(generic.id), [])

        for memory_id in (pref.id, commit.id, generic.id):
            metadata = self._metadata_for(memory_id=self.s.store.memories[memory_id])
            metadata["ttl_days"] = 90
            self.s.store.update_memory(memory_id, {
                "metadata": json.dumps(metadata),
                "created_at": self._to_days_ago(45),
                "access_count": 0,
            })

        self.s._apply_policy_retention()
        self.assertIn(pref.id, self.s.store.memories)
        self.assertIn(commit.id, self.s.store.memories)
        self.assertNotIn(generic.id, self.s.store.memories)

    def _metadata_for(self, memory_id: int):
        return json.loads(self.s.store.memories[memory_id]["metadata"])

    def test_work_policy_auto_tags_project(self):
        self.s.policy("work")
        memory = self.s.remember("Sprint update for #ProjectHorizon and project:horizon", deduplicate=False)
        tags = self._metadata_tags(memory.id)
        self.assertIn("project:atlas", tags)
        self.assertIn("project:horizon", tags)

    def test_ephemeral_policy_keeps_pinned_and_prunes_unpinned(self):
        self.s.policy("ephemeral")
        pinned = self.s.remember("Pinned note", metadata={"tags": ["pinned"]}, deduplicate=False)
        regular = self.s.remember("Ephemeral note", deduplicate=False)

        for memory_id in (pinned.id, regular.id):
            metadata = self._metadata_for(memory_id=memory_id)
            self.s.store.update_memory(memory_id, {
                "metadata": json.dumps(metadata),
                "created_at": self._to_days_ago(14),
                "access_count": 0,
            })

        self.s._apply_policy_retention()
        self.assertIn(pinned.id, self.s.store.memories)
        self.assertNotIn(regular.id, self.s.store.memories)


if __name__ == "__main__":
    unittest.main()
