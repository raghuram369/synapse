#!/usr/bin/env python3

import json
import time
import unittest

from temporal import parse_temporal
from sleep import SleepReport
from synapse import Synapse


class TestSleepRunner(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    @staticmethod
    def _metadata(memory_data):
        raw = memory_data.get("metadata", "{}")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                return {}
        return raw if isinstance(raw, dict) else {}

    def _semantic_memories(self):
        return [
            (memory_id, data, self._metadata(data))
            for memory_id, data in self.s.store.memories.items()
            if data.get("memory_type") == "semantic" and not data.get("consolidated", False)
        ]

    def _set_memory_time(self, memory_id: int, created_at: float) -> None:
        self.s.store.update_memory(memory_id, {"created_at": created_at, "last_accessed": created_at})
        self.s.temporal_index.remove_memory(memory_id)
        self.s.temporal_index.add_memory(memory_id, created_at)

    def _run_sleep(self):
        return self.s.sleep(verbose=False)

    def test_sleep_report_shape(self):
        report = self._run_sleep()

        self.assertIsInstance(report, SleepReport)
        self.assertIsInstance(report.consolidated, int)
        self.assertIsInstance(report.promoted, int)
        self.assertIsInstance(report.patterns_found, int)
        self.assertIsInstance(report.contradictions, int)
        self.assertIsInstance(report.pruned, int)
        self.assertIsInstance(report.graph_cleaned, int)
        self.assertIsInstance(report.duration_ms, float)
        self.assertTrue(report.duration_ms >= 0.0)
        self.assertIsInstance(report.details, dict)
        self.assertIn("verbose", report.details)

    def test_schedule_hook_memory_threshold(self):
        for idx in range(100):
            self.s.remember(f"Memory chunk {idx}", deduplicate=False)

        hook = self.s.sleep_runner.schedule_hook()
        self.assertTrue(hook["memory_due"])
        self.assertTrue(hook["should_sleep"])
        self.assertGreaterEqual(hook["active_memory_count"], 100)

    def test_schedule_hook_time_threshold(self):
        self.s._last_sleep_at = time.time() - (25 * 3600)
        hook = self.s.sleep_runner.schedule_hook()

        self.assertTrue(hook["time_due"])
        self.assertTrue(hook["should_sleep"])
        self.assertGreaterEqual(hook["seconds_since_last_sleep"], 24 * 3600)

    def test_auto_sleep_triggered_from_remember(self):
        self.s.sleep_runner.memory_threshold = 1
        last_sleep = self.s._last_sleep_at

        self.s.remember("Auto-sleep trigger sample", deduplicate=False)

        self.assertNotEqual(self.s._last_sleep_at, last_sleep)
        self.assertIsNotNone(self.s._last_sleep_at)

    def test_consolidation_step_creates_summary(self):
        for _ in range(3):
            self.s.remember(
                "User stores data in the database.",
                deduplicate=False,
            )

        report = self._run_sleep()
        self.assertGreaterEqual(report.consolidated, 3)
        self.assertTrue(any(
            self.s.store.memories[mid].get("memory_type") == "consolidated"
            for mid in self.s.store.memories
        ))

    def test_promotion_from_episodic_to_semantic(self):
        episodic = [
            self.s.remember("User attended a status meeting.", deduplicate=False)
            for _ in range(3)
        ]
        for mem in episodic:
            self.s.store.update_memory(mem.id, {"access_count": 3})

        report = self._run_sleep()
        semantic = [
            payload
            for _, _, payload in self._semantic_memories()
            if payload.get("semantic_topics")
        ]
        self.assertGreaterEqual(report.promoted, 1)
        self.assertTrue(any(
            "meeting" in {topic.lower() for topic in payload.get("semantic_topics", [])}
            for payload in semantic
        ))

    def test_pattern_mining_co_occurrence(self):
        for _ in range(3):
            self.s.remember(
                "User uses python for database jobs.",
                deduplicate=False,
            )

        report = self._run_sleep()
        self.assertGreaterEqual(report.patterns_found, 1)
        patterns = [
            metadata
            for _, memory_data, metadata in self._semantic_memories()
            if metadata.get("pattern_type") == "co_occurrence"
        ]
        self.assertTrue(patterns)
        self.assertTrue(any(
            "python" in memory_data.get("content", "").lower()
            and "database" in memory_data.get("content", "").lower()
            for _, memory_data, _ in self._semantic_memories()
            if self._metadata(memory_data).get("pattern_type") == "co_occurrence"
        ))

    def test_pattern_mining_streak(self):
        for idx in range(3):
            self.s.remember(
                "User attends planning meeting.",
                deduplicate=False,
                episode=f"meeting-episode-{idx}",
            )

        report = self._run_sleep()
        streak_patterns = [
            data
            for _, _, data in self._semantic_memories()
            if data.get("pattern_type") == "streak"
        ]
        self.assertGreaterEqual(report.patterns_found, 1)
        self.assertTrue(streak_patterns)

    def test_pattern_mining_seasonal(self):
        monday_timestamps = [
            parse_temporal("2025-02-03"),
            parse_temporal("2025-02-10"),
            parse_temporal("2025-02-17"),
        ]
        for ts in monday_timestamps:
            memory = self.s.remember(
                "User joins the weekly meeting.",
                deduplicate=False,
            )
            self._set_memory_time(memory.id, ts)

        report = self._run_sleep()
        seasonal_patterns = [
            self._metadata(memory_data).get("pattern_type")
            for _, memory_data, _ in self._semantic_memories()
            if self._metadata(memory_data).get("pattern_type") == "seasonal"
        ]
        self.assertGreaterEqual(report.patterns_found, 1)
        self.assertTrue(seasonal_patterns)

    def test_contradiction_scan_detects_conflicts(self):
        self.s.remember("User is vegetarian", deduplicate=False)
        self.s.remember("User is vegan", deduplicate=False)

        report = self._run_sleep()
        self.assertGreaterEqual(report.contradictions, 1)

    def test_pruning_removes_stale_low_value_memories(self):
        stale = self.s.remember("User once tracked a legacy endpoint.", deduplicate=False)
        stale_time = time.time() - (95 * 86400)
        self.s.store.update_memory(
            stale.id,
            {
                "strength": 0.02,
                "access_count": 0,
                "created_at": stale_time,
                "last_accessed": stale_time,
            },
        )
        self._set_memory_time(stale.id, stale_time)

        report = self._run_sleep()
        self.assertGreaterEqual(report.pruned, 1)
        self.assertNotIn(stale.id, self.s.store.memories)

    def test_graph_cleanup_removes_orphan_concepts(self):
        stale = self.s.remember("User wrote python script for parsing.", deduplicate=False)
        self.s.store.delete_memory(stale.id)

        report = self._run_sleep()
        self.assertGreaterEqual(report.graph_cleaned, 1)
        self.assertFalse(
            any(stale.id in node.memory_ids for node in self.s.concept_graph.concepts.values())
        )

    def test_graph_cleanup_merges_near_duplicate_concepts(self):
        first = self.s.remember("Python parser runs database checks.", deduplicate=False)
        second = self.s.remember("Python parser writes output.", deduplicate=False)
        self.s.concept_graph.link_memory_concept(first.id, "py-thon", "software")
        self.s.concept_graph.link_memory_concept(second.id, "python", "software")

        report = self._run_sleep()
        self.assertGreaterEqual(report.graph_cleaned, 1)
        self.assertNotIn("py-thon", self.s.concept_graph.concepts)

        merged_node = self.s.concept_graph.concepts.get("python")
        if merged_node is not None:
            self.assertIn(first.id, merged_node.memory_ids)
            self.assertIn(second.id, merged_node.memory_ids)


if __name__ == "__main__":
    unittest.main()
