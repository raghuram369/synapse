import unittest

from entity_graph import extract_concepts, expand_query
from synapse import Synapse


class TestEntityGraph(unittest.TestCase):

    def test_extract_concepts_finds_hardware_tools_people(self):
        text = (
            "Alice configured a Docker environment on her Mac Mini M4 Pro during a local project. "
            "The setup uses Python and Ollama."
        )
        concepts = extract_concepts(text)
        concept_map = dict(concepts)
        categories = set(concept_map.values())
        self.assertIn("hardware", categories)
        self.assertIn("tools", categories)
        self.assertIn("people", categories)
        self.assertIn("docker", concept_map)
        self.assertTrue(any(name == "alice" for name, _ in concepts))
        self.assertTrue(any(name == "ollama" for name, _ in concepts))

    def test_expand_query_maps_machine_to_hardware(self):
        expanded = expand_query(["machine"])
        self.assertIn("hardware", expanded)

    def test_recall_machine_query_finds_mac_mini(self):
        s = Synapse(":memory:")
        try:
            target = s.remember("Mac Mini M4 Pro is my daily machine.", deduplicate=False)
            other = s.remember("I wrote notes about project planning today.", deduplicate=False)
            results = s.recall("what machine am I on", limit=5)
            ids = [m.id for m in results]
            self.assertIn(target.id, ids)
        finally:
            s.close()

    def test_recall_local_ai_finds_ollama(self):
        s = Synapse(":memory:")
        try:
            target = s.remember("Running local AI models with Ollama for internal QA workflows.", deduplicate=False)
            other = s.remember("The weather was clear this afternoon.", deduplicate=False)
            results = s.recall("local AI", limit=5)
            ids = [m.id for m in results]
            self.assertIn(target.id, ids)
        finally:
            s.close()

    def test_concepts_api_returns_memory_counts(self):
        s = Synapse(":memory:")
        try:
            first = s.remember("Docker deployment for the synapse service.")
            s.remember("Project planning notes for the OpenClaw team.")
            concepts = s.concepts()
            concept_names = [row["name"] for row in concepts]
            self.assertIn("docker", concept_names)
            self.assertIn("project", concept_names)
            docker_entry = next(r for r in concepts if r["name"] == "docker")
            self.assertGreaterEqual(docker_entry["memory_count"], 1)
            self.assertIn(first.id, [first.id])
        finally:
            s.close()


    # ── False positive suppression tests ──

    def test_merkle_trees_does_not_produce_meeting(self):
        """'Federation sync uses Merkle trees' should NOT tag 'meeting'."""
        concepts = extract_concepts("Federation sync uses Merkle trees for consistency")
        concept_names = [name for name, _ in concepts]
        self.assertNotIn("meeting", concept_names,
                         "'sync' in technical context should not produce 'meeting'")

    def test_technical_text_no_spurious_hardware(self):
        """'Raghuram creating Synapse' should NOT tag 'hardware' (via 'memory' alias)."""
        concepts = extract_concepts(
            "Raghuram is creating Synapse, a memory database for AI agents"
        )
        concept_names = [name for name, _ in concepts]
        # "memory database" is a technical compound — should suppress hardware match
        self.assertNotIn("hardware", concept_names,
                         "'memory database' should not trigger hardware concept")

    def test_genuine_meeting_still_detected(self):
        """An actual meeting reference should still be detected."""
        concepts = extract_concepts("We have a standup meeting at 9am tomorrow")
        concept_names = [name for name, _ in concepts]
        self.assertIn("meeting", concept_names)

    def test_genuine_hardware_still_detected(self):
        """Actual hardware references should still be detected."""
        concepts = extract_concepts("I bought a new GPU for my desktop workstation")
        concept_names = [name for name, _ in concepts]
        self.assertIn("hardware", concept_names)


if __name__ == "__main__":
    unittest.main()
