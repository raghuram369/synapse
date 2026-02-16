import asyncio
import json
import os
import tempfile
import unittest

try:
    from mcp import types
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without mcp
    types = None

from synapse import Synapse
if types is not None:
    import mcp_appliance
    import mcp_server
else:  # pragma: no cover - import sentinel for skipped tests
    mcp_appliance = None
    mcp_server = None


@unittest.skipUnless(types is not None, "mcp package is not installed")
class TestMcpApplianceServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synapse-mcp-appliance-")
        self.db = os.path.join(self.tmp.name, "db")
        self.syn = Synapse(self.db)
        self.server, _lock = mcp_appliance._build_server(syn=self.syn)

    def tearDown(self):
        self.syn.close()
        self.tmp.cleanup()

    def _run(self, coroutine):
        return asyncio.run(coroutine)

    async def _call(self, name, args=None):
        response = await self.server.call_tool(name, args or {})
        payload = json.loads(response[0].text)
        return payload

    def _call_sync(self, name, args=None):
        payload = self._run(self._call(name, args))
        return payload

    def _tool_names(self):
        tools = self._run(self.server.list_tools())
        return sorted(t.name for t in tools)

    def test_appliance_mode_exposes_eight_tools(self):
        tool_names = set(self._tool_names())
        self.assertEqual(len(tool_names), 8)
        self.assertEqual(tool_names, mcp_appliance.APPLIANCE_TOOL_NAMES)

    def test_full_mode_preserves_rich_surface(self):
        server, _ = mcp_server._build_server(syn=self.syn)
        names = {tool.name for tool in self._run(server.list_tools())}
        self.assertGreaterEqual(len(names), 21)
        self.assertIn("remember", names)
        self.assertIn("compile_context", names)

    def test_remember_normalizes_entities_and_kind_map(self):
        payload = self._call_sync(
            "remember",
            {
                "content": "Alice in NYC attended the meetup.",
                "kind": "fact",
                "tags": ["NYC", " Meetup "],
            },
        )
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["memory_type"], "fact")
        tags = {tag.lower() for tag in result["auto_normalized_tags"]}
        self.assertTrue({"nyc", "meetup", "new york", "new york city"} & tags)

    def test_compile_context_returns_structured_text_and_sections(self):
        self._call_sync("remember", {"content": "Project Orion launched in 2024.", "kind": "event"})
        payload = self._call_sync(
            "compile_context",
            {
                "query": "Project Orion",
                "budget_tokens": 1800,
                "mode": "balanced",
            },
        )
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertIn("context_text", result)
        self.assertIn("relevant_memories", result)
        self.assertIn("graph_slice_summary", result)
        self.assertIn("evidence", result)
        self.assertIsInstance(result["context_text"], str)
        self.assertGreaterEqual(len(result["context_text"]), 1)

    def test_timeline_filters_recent_window(self):
        old = self.syn.remember("Engine power is stable over time.")
        self.syn.store.update_memory(old.id, {"created_at": old.created_at - 8 * 24 * 60 * 60})
        self._call_sync("remember", {"content": "Engine power is stable and efficient."})

        payload = self._call_sync("timeline", {"range": "7d"})
        self.assertTrue(payload["ok"])
        timeline = payload["result"]["timeline"]
        ids = {entry["id"] for entry in timeline if "id" in entry}
        self.assertNotIn(old.id, ids)
        self.assertTrue(len(ids) >= 1)

    def test_timeline_filters_by_topic(self):
        self._call_sync("remember", {"content": "Alice designs engines for the airframe."})
        self._call_sync("remember", {"content": "Alice redesigned aircraft engines."})
        self._call_sync("remember", {"content": "Bob builds bridges in town."})
        self._call_sync("remember", {"content": "Bob builds bridges across rivers."})

        payload = self._call_sync("timeline", {"range": "all", "topic": "alice"})
        self.assertTrue(payload["ok"])
        entries = payload["result"]["timeline"]
        self.assertGreaterEqual(len(entries), 1)
        self.assertTrue(all("alice" in (entry["content"].lower()) for entry in entries))

    def test_what_changed_captures_belief_updates(self):
        self._call_sync("remember", {"content": "Alice's age is 30"})
        self._call_sync("remember", {"content": "Alice's age is 31"})
        payload = self._call_sync("what_changed", {"range": "30d", "topic": "alice"})
        self.assertTrue(payload["ok"])
        self.assertIsInstance(payload["result"]["changed_beliefs"], list)
        self.assertIsInstance(payload["result"]["new_facts"], list)
        self.assertGreaterEqual(len(payload["result"]["new_facts"]), 1)

    def test_contradictions_filter_by_topic(self):
        self._call_sync("remember", {"content": "The sky is blue"})
        self._call_sync("remember", {"content": "The sky is not blue"})
        payload = self._call_sync("contradictions", {"topic": "sky"})
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(payload["result"]["count"], 1)

    def test_fact_history_applies_relation_filter(self):
        self._call_sync("remember", {"content": "Alice's age is 30"})
        self._call_sync("remember", {"content": "Alice's age is 31"})
        payload = self._call_sync("fact_history", {"subject": "Alice", "relation": "age"})
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(len(payload["result"]["chain"]), 1)
        self.assertEqual(payload["result"]["relation"], "age")

    def test_sleep_respects_disable_flags(self):
        payload = self._call_sync("sleep", {"consolidate": False, "prune": False})
        self.assertTrue(payload["ok"])
        report = payload["result"]["sleep_report"]
        self.assertEqual(report["consolidated"], 0)
        self.assertEqual(report["pruned"], 0)
        self.assertEqual(report["details"]["status"], "skipped")

    def test_stats_includes_counts_and_drift(self):
        self._call_sync("remember", {"content": "Alice likes tea"})
        payload = self._call_sync("stats")
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertIn("memory_count", result)
        self.assertIn("concept_count", result)
        self.assertIn("hot_concepts", result)
        self.assertIn("drift_indicators", result)
        self.assertIn("store_size_bytes", result)


if __name__ == "__main__":
    unittest.main()
