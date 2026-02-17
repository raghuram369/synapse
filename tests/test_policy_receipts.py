import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture import ingest
from policy_receipts import load_receipts
from synapse import Synapse
import cli


class TestPolicyReceipts(unittest.TestCase):
    def test_capture_write_receipts_emit_allow_and_deny(self):
        with tempfile.TemporaryDirectory(prefix="synapse-receipts-") as tmpdir:
            db_path = os.path.join(tmpdir, "engine")
            s = Synapse(db_path)
            try:
                allow = ingest(
                    "I like testing memory receipts with deterministic content",
                    synapse=s,
                    policy="auto",
                )
                deny = ingest("quick note", synapse=s, policy="off")
            finally:
                s.close()

            self.assertEqual(allow.value, "stored")
            self.assertEqual(deny.value, "ignored_policy")

            records, _source = load_receipts(last=5, db_path=db_path)
            self.assertGreaterEqual(len(records), 2)
            decisions = {r.get("decision") for r in records if isinstance(r, dict)}
            self.assertIn("allow", decisions)
            self.assertIn("deny", decisions)

            write_entries = [r for r in records if isinstance(r, dict) and r.get("purpose") == "memory_write"]
            self.assertTrue(write_entries)
            self.assertEqual(write_entries[-1].get("purpose"), "memory_write")

    def test_recall_read_receipts_visible_in_cli(self):
        with tempfile.TemporaryDirectory(prefix="synapse-receipts-") as tmpdir:
            db_path = os.path.join(tmpdir, "engine")
            s = Synapse(db_path)
            try:
                s.remember("Public roadmap", scope="public", deduplicate=False)
                s.remember("Shared note", scope="shared", deduplicate=False)
                s.remember("Private note", scope="private", deduplicate=False)

                results = s.recall(context="", limit=10, scope="public")
                self.assertGreaterEqual(len(results), 1)
            finally:
                s.close()

            records, _source = load_receipts(last=5, db_path=db_path)
            reads = [r for r in records if isinstance(r, dict) and r.get("purpose") == "memory_read"]
            self.assertTrue(reads)
            last_read = reads[-1]
            self.assertEqual(last_read.get("scope_requested"), "public")
            self.assertEqual(last_read.get("scope_applied"), "public")
            self.assertIn("scope_too_restrictive", last_read.get("block_reasons", []))
            counts = last_read.get("memory_counts", {})
            self.assertGreaterEqual(counts.get("considered", 0), 2)
            self.assertGreaterEqual(counts.get("blocked", 0), 1)

            output = io.StringIO()
            with redirect_stdout(output):
                cli.cmd_permit(
                    type("Args", (), {
                        "permit_action": "receipts",
                        "last": 3,
                        "db": db_path,
                        "json": True,
                    })(),
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema"], "synapse.permit.receipt.v1")
            self.assertTrue(payload["available"])
            self.assertGreaterEqual(payload["count"], 1)
            self.assertEqual(payload["receipts"][-1]["purpose"], "memory_read")
