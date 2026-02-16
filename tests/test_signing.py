"""Tests for signing module."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signing import sign_artifact, verify_artifact, sign_pack, verify_pack, _compute_hash


class TestSigning(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".brain")
        self.tmpfile.write(b"test brain content 12345")
        self.tmpfile.close()
        self.path = self.tmpfile.name

    def tearDown(self):
        for p in [self.path, f"{self.path}.sig"]:
            try:
                os.remove(p)
            except OSError:
                pass

    def test_compute_hash(self):
        h = _compute_hash(self.path)
        self.assertTrue(h.startswith("sha256:"))
        self.assertEqual(len(h), 7 + 64)  # "sha256:" + 64 hex chars

    def test_sign_creates_sig_file(self):
        sig_path = sign_artifact(self.path)
        self.assertTrue(os.path.exists(sig_path))
        self.assertEqual(sig_path, f"{self.path}.sig")

        with open(sig_path) as f:
            sig = json.load(f)
        self.assertIn("hash", sig)
        self.assertIn("timestamp", sig)
        self.assertEqual(sig["signer"], "synapse-ai-memory")
        self.assertEqual(sig["version"], "0.8.1")

    def test_verify_valid(self):
        sign_artifact(self.path)
        result = verify_artifact(self.path)
        self.assertTrue(result["valid"])

    def test_verify_tampered(self):
        sign_artifact(self.path)
        # Tamper with the file
        with open(self.path, "ab") as f:
            f.write(b"tampered")
        result = verify_artifact(self.path)
        self.assertFalse(result["valid"])

    def test_verify_no_sig(self):
        result = verify_artifact(self.path)
        self.assertFalse(result["valid"])
        self.assertIn("error", result)

    def test_sign_pack_alias(self):
        sig_path = sign_pack(self.path)
        self.assertTrue(os.path.exists(sig_path))

    def test_verify_pack_alias(self):
        sign_pack(self.path)
        result = verify_pack(self.path)
        self.assertTrue(result["valid"])

    def test_idempotent_hash(self):
        h1 = _compute_hash(self.path)
        h2 = _compute_hash(self.path)
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
