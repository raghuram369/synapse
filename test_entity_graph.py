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


if __name__ == "__main__":
    unittest.main()
