import unittest

from synapse.compat import Mem0Client


class TestMem0Compatibility(unittest.TestCase):
    def setUp(self):
        self.client = Mem0Client(path=":memory:")

    def tearDown(self):
        self.client.close()

    @staticmethod
    def _assert_mem_shape(row):
        assert isinstance(row, dict)
        assert set(row.keys()) == {"id", "memory", "score", "metadata", "created_at"}
        assert isinstance(row["id"], str)
        assert isinstance(row["memory"], str)
        assert isinstance(row["score"], (int, float))
        assert isinstance(row["metadata"], dict)
        assert isinstance(row["created_at"], float)

    def test_add_includes_user_tag_and_keeps_metadata(self):
        add_out = self.client.add(
            "Call mom", user_id="alice", metadata={"tags": "manual", "importance": "high"}
        )
        self.assertIn("id", add_out)

        row = self.client.get(add_out["id"])
        self.assertIsNotNone(row)
        assert row is not None
        self._assert_mem_shape(row)
        self.assertEqual(row["memory"], "Call mom")
        self.assertEqual(row["metadata"]["importance"], "high")
        self.assertEqual(row["metadata"]["tags"], ["manual", "user:alice"])

    def test_search_filters_by_user_id_and_returns_expected_shape(self):
        alice = self.client.add("Need to buy blue notebook", user_id="alice")
        bob = self.client.add("Need to buy blue notebook", user_id="bob")
        self.client.add("Different note for alice", user_id="alice")

        results_alice = self.client.search("blue notebook", user_id="alice", limit=10)
        self.assertEqual(len(results_alice), 1)
        self.assertEqual(results_alice[0]["id"], str(alice["id"]))
        self._assert_mem_shape(results_alice[0])

        results_bob = self.client.search("blue notebook", user_id="bob", limit=10)
        self.assertEqual(len(results_bob), 1)
        self.assertEqual(results_bob[0]["id"], str(bob["id"]))

        results_all = self.client.search("blue notebook", limit=10)
        self.assertEqual(len(results_all), 2)

    def test_get_all_filters_by_user_id(self):
        a1 = self.client.add("User alice one", user_id="alice")
        a2 = self.client.add("User alice two", user_id="alice")
        b1 = self.client.add("User bob", user_id="bob")

        alice_rows = self.client.get_all(user_id="alice")
        self.assertEqual([row["id"] for row in alice_rows], [str(a1["id"]), str(a2["id"])])
        for row in alice_rows:
            self._assert_mem_shape(row)

        bob_rows = self.client.get_all(user_id="bob")
        self.assertEqual([row["id"] for row in bob_rows], [str(b1["id"])])

        all_rows = self.client.get_all()
        self.assertEqual(len(all_rows), 3)

    def test_get_returns_none_for_missing(self):
        self.assertIsNone(self.client.get("99999"))

    def test_get_returns_expected_shape(self):
        created = self.client.add("Persistent memory", user_id="alice")
        row = self.client.get(created["id"])

        self.assertIsNotNone(row)
        assert row is not None
        self._assert_mem_shape(row)
        self.assertEqual(row["memory"], "Persistent memory")

    def test_update_changes_memory_and_adds_history(self):
        created = self.client.add("Draft version", user_id="alice")
        updated = self.client.update(created["id"], "Final version")

        self.assertIsNotNone(updated)
        assert updated is not None
        self._assert_mem_shape(updated)
        self.assertEqual(updated["memory"], "Final version")

        current = self.client.get(created["id"])
        assert current is not None
        self.assertEqual(current["memory"], "Final version")

        history = self.client.history(created["id"])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["id"], str(created["id"]))
        self.assertFalse(history[0]["current"])
        self.assertEqual(history[0]["version"], 1)
        self.assertEqual(history[0]["memory"], "Draft version")
        self.assertTrue(history[1]["current"])
        self.assertEqual(history[1]["version"], 2)
        self.assertEqual(history[1]["memory"], "Final version")

    def test_update_returns_none_for_unknown(self):
        self.assertIsNone(self.client.update("missing", "Nope"))

    def test_delete_returns_bool_and_removes(self):
        created = self.client.add("Gone soon", user_id="alice")
        self.assertTrue(self.client.delete(created["id"]))
        self.assertIsNone(self.client.get(created["id"]))
        self.assertFalse(self.client.delete(created["id"]))

    def test_delete_all_respects_user_filter_and_returns_count(self):
        a1 = self.client.add("Alice one", user_id="alice")
        a2 = self.client.add("Alice two", user_id="alice")
        b1 = self.client.add("Bob only", user_id="bob")
        a3 = self.client.add("No user")

        deleted_for_alice = self.client.delete_all(user_id="alice")
        self.assertEqual(deleted_for_alice, 2)

        remaining = self.client.get_all()
        remaining_ids = {row["id"] for row in remaining}
        self.assertEqual(remaining_ids, {str(b1["id"]), str(a3["id"])})

        deleted_all = self.client.delete_all()
        self.assertEqual(deleted_all, 2)
        self.assertEqual(self.client.get_all(), [])
