"""Tests for review queue."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse
from review_queue import ReviewQueue


class TestReviewQueue(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.synapse = Synapse(":memory:")
        self.rq = ReviewQueue(self.synapse, pending_dir=self.tmpdir)

    def tearDown(self):
        self.synapse.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_submit_and_count(self):
        self.assertEqual(self.rq.count(), 0)
        item_id = self.rq.submit("test content")
        self.assertEqual(self.rq.count(), 1)
        self.assertTrue(item_id)

    def test_list_pending(self):
        self.rq.submit("item 1")
        self.rq.submit("item 2")
        items = self.rq.list_pending()
        self.assertEqual(len(items), 2)
        contents = {i["content"] for i in items}
        self.assertIn("item 1", contents)
        self.assertIn("item 2", contents)

    def test_approve(self):
        item_id = self.rq.submit("remember me")
        memory = self.rq.approve(item_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "remember me")
        self.assertEqual(self.rq.count(), 0)
        # Should be in synapse now
        results = self.synapse.recall("remember me", limit=1)
        self.assertTrue(len(results) > 0)

    def test_approve_nonexistent(self):
        result = self.rq.approve("nonexistent")
        self.assertIsNone(result)

    def test_reject(self):
        item_id = self.rq.submit("reject me")
        self.assertTrue(self.rq.reject(item_id))
        self.assertEqual(self.rq.count(), 0)

    def test_reject_nonexistent(self):
        self.assertFalse(self.rq.reject("nonexistent"))

    def test_approve_all(self):
        self.rq.submit("a")
        self.rq.submit("b")
        self.rq.submit("c")
        results = self.rq.approve_all()
        self.assertEqual(len(results), 3)
        self.assertEqual(self.rq.count(), 0)

    def test_metadata_preserved(self):
        item_id = self.rq.submit("meta test", metadata={"source": "test"})
        items = self.rq.list_pending()
        self.assertEqual(items[0]["metadata"]["source"], "test")


if __name__ == "__main__":
    unittest.main()
