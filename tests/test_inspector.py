import json
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch
import webbrowser

from synapse import Synapse
from inspector import SynapseInspector


class TestSynapseInspectorAPI(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-inspector-")
        self.db = f"{self.tempdir.name}/db"
        self.syn = Synapse(self.db)
        self.mem_a = self.syn.remember("Alice likes tea", deduplicate=False)
        self.mem_b = self.syn.remember("Alice likes coffee", deduplicate=False)
        self.mem_c = self.syn.remember("The sky is blue", deduplicate=False)
        self.mem_d = self.syn.remember("The sky is not blue", deduplicate=False)
        self.syn.link(self.mem_a.id, self.mem_b.id, "related", weight=0.9)
        self.browser_patch = None
        self._start_server()

    def tearDown(self):
        if self.browser_patch is not None:
            self.browser_patch.stop()
        if self.inspector is not None:
            self.inspector.stop()
        if self.server_thread is not None:
            self.server_thread.join(timeout=1)
        self.syn.close()
        self.tempdir.cleanup()

    def _start_server(self):
        self.inspector = None
        self.server_thread = None
        self.browser_patch = patch.object(webbrowser, "open", return_value=True)
        self.browser_patch.start()
        self.inspector = SynapseInspector(self.syn, port=0)
        self.server_thread = threading.Thread(target=self.inspector.start, daemon=True)
        self.server_thread.start()

        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self.inspector._server is not None:
                break
            time.sleep(0.05)

        self.assertIsNotNone(self.inspector._server)
        self.base = f"http://127.0.0.1:{self.inspector.port}"

    def _get_json(self, path: str):
        with urllib.request.urlopen(self.base + path) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def test_stats_endpoint_returns_json(self):
        payload = self._get_json("/api/stats")
        self.assertIn("memory_count", payload)
        self.assertIn("concept_count", payload)
        self.assertIn("contradictions", payload)
        self.assertIn("belief_count", payload)

    def test_memories_endpoint_returns_json_list(self):
        payload = self._get_json("/api/memories")
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 3)
        item = payload[0]
        self.assertIn("id", item)
        self.assertIn("content", item)
        self.assertIn("created_at", item)

    def test_concepts_endpoint_returns_graph_data(self):
        payload = self._get_json("/api/concepts")
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)
        self.assertIsInstance(payload["nodes"], list)
        self.assertIsInstance(payload["edges"], list)

    def test_contradictions_endpoint_returns_json(self):
        payload = self._get_json("/api/contradictions")
        self.assertIn("contradictions", payload)
        contradictions = payload["contradictions"]
        self.assertIsInstance(contradictions, list)
        if contradictions:
            item = contradictions[0]
            self.assertIn("left_id", item)
            self.assertIn("right_id", item)
            self.assertIn("kind", item)

    def test_beliefs_endpoint_search(self):
        payload = self._get_json("/api/beliefs?q=alice")
        self.assertIn("beliefs", payload)
        beliefs = payload["beliefs"]
        self.assertIsInstance(beliefs, list)
        if beliefs:
            item = beliefs[0]
            self.assertIn("fact_key", item)
            self.assertIn("current", item)

    def test_recall_endpoint_returns_score_breakdowns(self):
        payload = self._get_json("/api/recall?q=Alice&limit=5")
        self.assertIn("results", payload)
        results = payload["results"]
        self.assertIsInstance(results, list)
        if results:
            result = results[0]
            self.assertIn("id", result)
            self.assertIn("score", result)
            self.assertIn("score_breakdown", result)

    def test_compile_endpoint_returns_pack(self):
        payload = self._get_json("/api/compile?q=Alice&budget=1600")
        self.assertIn("context_text", payload)
        self.assertIn("pack", payload)
        self.assertIsInstance(payload["pack"], dict)


if __name__ == "__main__":
    unittest.main()
