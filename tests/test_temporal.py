import unittest
import time
from datetime import datetime, timezone

from synapse import Synapse
from temporal import (
    latest_facts,
    memories_as_of,
    memories_during,
    parse_temporal,
    temporal_chain,
)


class TestTemporalFields(unittest.TestCase):

    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def test_parse_temporal_iso_date(self):
        ts = parse_temporal("2024-03-15")
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        self.assertEqual((dt.year, dt.month, dt.day), (2024, 3, 15))

    def test_parse_temporal_month_numeric(self):
        ts = parse_temporal("2024-03")
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        self.assertEqual((dt.year, dt.month, dt.day), (2024, 3, 1))

    def test_parse_temporal_month_name(self):
        ts = parse_temporal("March 2024")
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        self.assertEqual((dt.year, dt.month, dt.day), (2024, 3, 1))

    def test_parse_temporal_relative(self):
        now = time.time()
        yesterday = parse_temporal("yesterday")
        self.assertLess(yesterday, now)
        self.assertGreater(yesterday, now - 86400 * 2)

    def test_parse_temporal_timestamp_string(self):
        self.assertAlmostEqual(parse_temporal("1700000000"), 1700000000.0)

    def test_as_of_before_and_after_validity(self):
        base = parse_temporal("2024-01-01")
        old = self.s.remember(
            "User worked in Seattle",
            deduplicate=False,
            valid_from="2023-01-01",
            valid_to="2023-12-31",
        )
        recent = self.s.remember(
            "User works in Austin",
            deduplicate=False,
            valid_from="2024-06-01",
            valid_to="2024-12-31",
        )

        before = self.s.recall("User", temporal="as_of:2024-03-01")
        after = self.s.recall("User", temporal="as_of:2024-08-01")

        before_ids = [m.id for m in before]
        after_ids = [m.id for m in after]

        self.assertIn(old.id, before_ids)
        self.assertNotIn(recent.id, before_ids)
        self.assertIn(recent.id, after_ids)
        self.assertNotIn(old.id, after_ids)

    def test_during_interval_query(self):
        self.s.remember(
            "Project timeline first phase",
            deduplicate=False,
            valid_from="2023-01-01",
            valid_to="2024-01-01",
        )
        self.s.remember(
            "Project timeline second phase",
            deduplicate=False,
            valid_from="2024-06-01",
            valid_to="2025-01-01",
        )

        results = self.s.recall("Project timeline", temporal="during:2023-03:2023-12")
        self.assertEqual(len(results), 1)
        self.assertIn("first phase", results[0].content)

    def test_latest_facts_dedup(self):
        m1 = self.s.remember(
            "I like Python",
            deduplicate=False,
            valid_from="2023-01-01",
        )
        m2 = self.s.remember(
            "I like Python for scripts",
            deduplicate=False,
            valid_from="2024-01-01",
        )
        m3 = self.s.remember(
            "I like Java",
            deduplicate=False,
            valid_from="2024-01-01",
        )

        merged = latest_facts([m1, m2, m3])
        merged_ids = [m.id for m in merged]

        self.assertEqual(len(merged), 2)
        self.assertIn(m2.id, merged_ids)
        self.assertIn(m3.id, merged_ids)
        self.assertNotIn(m1.id, merged_ids)

    def test_temporal_chain_order(self):
        m1 = self.s.remember("Python 1.x architecture", deduplicate=False, valid_from="2024-01-01")
        m2 = self.s.remember("Python 2.x architecture", deduplicate=False, valid_from="2024-06-01")
        m3 = self.s.remember("Python 1.y patch", deduplicate=False, valid_from="2024-03-01")

        chain = temporal_chain([m1, m2, m3], {"python"})
        chain_ids = [m.id for m in chain]

        self.assertEqual(chain_ids, [m1.id, m3.id, m2.id])

    def test_remember_parses_temporal_strings(self):
        m = self.s.remember(
            "Recorded with strings",
            deduplicate=False,
            observed_at="2024-07-01",
            valid_from="2024-07-01",
            valid_to="2024-07-31",
        )

        self.assertEqual(m.observed_at, parse_temporal("2024-07-01"))
        self.assertEqual(m.valid_from, parse_temporal("2024-07-01"))
        self.assertEqual(m.valid_to, parse_temporal("2024-07-31"))

    def test_recall_month_filter_existing(self):
        march = self.s.remember("March activity", deduplicate=False)
        april = self.s.remember("April activity", deduplicate=False)

        self.s.store.update_memory(march.id, {"created_at": parse_temporal("2024-03-15")})
        self.s.store.update_memory(april.id, {"created_at": parse_temporal("2024-04-15")})

        results = self.s.recall("activity", temporal="2024-03", limit=10)
        result_ids = [m.id for m in results]
        self.assertIn(march.id, result_ids)
        self.assertNotIn(april.id, result_ids)

    def test_recall_temporal_latest_mode(self):
        m1 = self.s.remember("User likes Python 3", deduplicate=False, valid_from="2023-01-01")
        m2 = self.s.remember("User likes Python 4", deduplicate=False, valid_from="2024-01-01")
        m3 = self.s.remember("User likes tea", deduplicate=False, valid_from="2024-01-01")

        latest = self.s.recall("User likes", temporal="latest", limit=10)
        latest_ids = [m.id for m in latest]

        self.assertIn(m2.id, latest_ids)
        self.assertIn(m3.id, latest_ids)
        self.assertNotIn(m1.id, latest_ids)

    def test_recall_during_mode(self):
        self.s.remember("Recall during memory", deduplicate=False, valid_from="2024-01-01", valid_to="2024-02-01")
        self.s.remember("Recall outside memory", deduplicate=False, valid_from="2024-07-01", valid_to="2024-08-01")

        results = self.s.recall("Recall", temporal="during:2024-01:2024-02", limit=10)
        self.assertEqual(len(results), 1)
        self.assertIn("during memory", results[0].content)


if __name__ == "__main__":
    unittest.main()
