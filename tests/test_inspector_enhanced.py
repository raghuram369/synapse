"""Tests for the enhanced inspector dashboard and API endpoints."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse
from inspector import SynapseInspector


def _make_synapse_with_data() -> Synapse:
    s = Synapse(path=":memory:")
    s.remember("Alice likes Python", memory_type="preference")
    s.remember("Alice lives in Austin", memory_type="semantic")
    s.remember("Alice lives in Denver", memory_type="semantic")
    s.remember("Had coffee with Bob yesterday", memory_type="event")
    return s


class _FakeRequest:
    """Minimal fake for BaseHTTPRequestHandler wfile/rfile."""
    def __init__(self, method: str, path: str, body: bytes = b""):
        self.method = method
        self.path = path
        self.body = body


def _make_handler(inspector: SynapseInspector):
    """Create a handler class from inspector."""
    return inspector._build_handler()


def _get(inspector: SynapseInspector, path: str) -> dict:
    """Simulate a GET request to the inspector and return parsed JSON."""
    from http.server import BaseHTTPRequestHandler
    from io import BytesIO

    handler_cls = _make_handler(inspector)

    class FakeHandler(handler_cls):
        def __init__(self, path):
            self.path = path
            self.headers = {}
            self._response_code = None
            self._response_headers = {}
            self._body = BytesIO()
            self.wfile = self._body
            self.requestline = f"GET {path} HTTP/1.1"
            self.client_address = ("127.0.0.1", 0)
            self.request_version = "HTTP/1.1"
            self.command = "GET"

        def send_response(self, code):
            self._response_code = code

        def send_header(self, key, value):
            self._response_headers[key] = value

        def end_headers(self):
            pass

        def log_message(self, *a):
            pass

    h = FakeHandler(path)
    h.do_GET()
    h._body.seek(0)
    raw = h._body.read()
    if h._response_headers.get("Content-Type", "").startswith("application/json"):
        return json.loads(raw)
    return {"_html": raw.decode("utf-8"), "_status": h._response_code}


def _post(inspector: SynapseInspector, path: str, body: dict) -> dict:
    """Simulate a POST request."""
    from io import BytesIO

    handler_cls = _make_handler(inspector)
    body_bytes = json.dumps(body).encode("utf-8")

    class FakeHandler(handler_cls):
        def __init__(self, path, body_bytes):
            self.path = path
            self.headers = {"Content-Length": str(len(body_bytes))}
            self.rfile = BytesIO(body_bytes)
            self._response_code = None
            self._response_headers = {}
            self._body = BytesIO()
            self.wfile = self._body
            self.requestline = f"POST {path} HTTP/1.1"
            self.client_address = ("127.0.0.1", 0)
            self.request_version = "HTTP/1.1"
            self.command = "POST"

        def send_response(self, code):
            self._response_code = code

        def send_header(self, key, value):
            self._response_headers[key] = value

        def end_headers(self):
            pass

        def log_message(self, *a):
            pass

    h = FakeHandler(path, body_bytes)
    h.do_POST()
    h._body.seek(0)
    return json.loads(h._body.read())


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_health_returns_all_fields(self):
        result = _get(self.inspector, "/api/health")
        self.assertIn("memory_count", result)
        self.assertIn("concept_count", result)
        self.assertIn("triple_count", result)
        self.assertIn("contradiction_count", result)
        self.assertIn("belief_count", result)
        self.assertIn("storage_display", result)
        self.assertIn("last_sleep_display", result)
        self.assertGreater(result["memory_count"], 0)

    def test_health_storage_display(self):
        result = _get(self.inspector, "/api/health")
        self.assertIn("storage_bytes", result)
        self.assertIsInstance(result["storage_bytes"], int)
        self.assertGreater(result["storage_bytes"], 0)


class TestTimelineEndpoint(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_timeline_returns_memories(self):
        result = _get(self.inspector, "/api/timeline")
        self.assertIn("memories", result)
        self.assertGreater(len(result["memories"]), 0)

    def test_timeline_with_date_range(self):
        result = _get(self.inspector, "/api/timeline?from=2020-01-01&to=2099-12-31")
        self.assertIn("memories", result)
        self.assertGreater(len(result["memories"]), 0)

    def test_timeline_empty_range(self):
        result = _get(self.inspector, "/api/timeline?from=1999-01-01&to=1999-01-02")
        self.assertEqual(len(result["memories"]), 0)

    def test_timeline_memory_fields(self):
        result = _get(self.inspector, "/api/timeline")
        m = result["memories"][0]
        self.assertIn("id", m)
        self.assertIn("content", m)
        self.assertIn("memory_type", m)
        self.assertIn("is_promotion", m)
        self.assertIn("created_at", m)


class TestBeliefsEndpoint(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_beliefs_returns_list(self):
        result = _get(self.inspector, "/api/beliefs")
        self.assertIn("beliefs", result)
        self.assertIsInstance(result["beliefs"], list)

    def test_beliefs_filtered(self):
        result = _get(self.inspector, "/api/beliefs?q=nonexistent_xyz")
        self.assertEqual(len(result["beliefs"]), 0)


class TestContradictionsEndpoint(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_contradictions_returns_list(self):
        result = _get(self.inspector, "/api/contradictions")
        self.assertIn("contradictions", result)
        self.assertIsInstance(result["contradictions"], list)


class TestResolveContradiction(unittest.TestCase):
    def setUp(self):
        self.synapse = Synapse(path=":memory:")
        self.synapse.remember("Alice lives in Austin", memory_type="semantic")
        self.synapse.remember("Alice lives in Denver", memory_type="semantic")
        self.inspector = SynapseInspector(self.synapse)

    def test_resolve_missing_params(self):
        result = _post(self.inspector, "/api/resolve-contradiction", {})
        self.assertIn("error", result)

    def test_resolve_invalid_id(self):
        result = _post(self.inspector, "/api/resolve-contradiction", {"contradiction_id": 999, "winner_memory_id": 1})
        # Should error (invalid id)
        self.assertIn("error", result)


class TestPreviewContext(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_preview_context_returns_text(self):
        result = _post(self.inspector, "/api/preview-context", {"query": "Alice", "budget": 1000, "policy": "balanced"})
        self.assertIn("context_text", result)
        self.assertIn("pack", result)
        self.assertEqual(result["policy"], "balanced")

    def test_preview_context_different_policies(self):
        for policy in ["balanced", "precise", "broad", "temporal"]:
            result = _post(self.inspector, "/api/preview-context", {"query": "Alice", "budget": 1000, "policy": policy})
            self.assertIn("context_text", result)


class TestDigestsEndpoint(unittest.TestCase):
    def test_digests_no_dir(self):
        synapse = _make_synapse_with_data()
        inspector = SynapseInspector(synapse)
        # With no ~/.synapse/digests, should return empty
        with patch("inspector.os.path.isdir", return_value=False):
            result = _get(inspector, "/api/digests")
        self.assertEqual(result["digests"], [])

    def test_digests_with_files(self):
        synapse = _make_synapse_with_data()
        inspector = SynapseInspector(synapse)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a fake digest
            digest = {"date": "2026-02-14", "consolidated": 5, "promoted": 2, "pruned": 1, "hot_topics": ["alice", "python"]}
            with open(os.path.join(tmpdir, "2026-02-14.json"), "w") as f:
                json.dump(digest, f)

            with patch("inspector.os.path.expanduser", return_value=tmpdir):
                with patch("inspector.os.path.isdir", return_value=True):
                    result = inspector._collect_digests()

            self.assertEqual(len(result["digests"]), 1)
            self.assertEqual(result["digests"][0]["date"], "2026-02-14")
            self.assertEqual(result["digests"][0]["consolidated"], 5)
            self.assertEqual(result["digests"][0]["hot_topics"], ["alice", "python"])


class TestDashboardHTML(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_dashboard_returns_html(self):
        result = _get(self.inspector, "/")
        self.assertIn("_html", result)
        html = result["_html"]
        self.assertIn("Synapse Inspector", html)
        self.assertIn("tab-overview", html)
        self.assertIn("tab-timeline", html)
        self.assertIn("tab-beliefs", html)
        self.assertIn("tab-contradictions", html)
        self.assertIn("tab-context", html)
        self.assertIn("tab-sleep", html)

    def test_dashboard_has_search_bar(self):
        result = _get(self.inspector, "/")
        self.assertIn("global-search", result["_html"])

    def test_dashboard_has_policy_selector(self):
        result = _get(self.inspector, "/")
        self.assertIn("compile-policy", result["_html"])


class TestUnknownEndpoints(unittest.TestCase):
    def setUp(self):
        self.synapse = _make_synapse_with_data()
        self.inspector = SynapseInspector(self.synapse)

    def test_unknown_get(self):
        result = _get(self.inspector, "/api/nonexistent")
        self.assertIn("error", result)

    def test_unknown_post(self):
        result = _post(self.inspector, "/api/nonexistent", {})
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
