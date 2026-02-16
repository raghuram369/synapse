#!/usr/bin/env python3

import unittest

from indexes import ConceptGraph
from graph_retrieval import GraphRetriever
from synapse import Synapse


class TripleIndexStub:
    """Simple triple index stub for unit tests."""

    def __init__(self, edges):
        self.edges = edges

    def get_neighbors(self, concept):
        return self.edges.get(concept, [])


class TestGraphRetriever(unittest.TestCase):
    def test_multi_hop_spread_decay_chain(self):
        cg = ConceptGraph()
        cg.link_memory_concept(1, "alpha", "demo")
        cg.link_memory_concept(1, "beta", "demo")
        cg.link_memory_concept(2, "beta", "demo")
        cg.link_memory_concept(2, "gamma", "demo")
        cg.link_memory_concept(3, "gamma", "demo")
        cg.link_memory_concept(3, "delta", "demo")

        retriever = GraphRetriever(cg)
        scores = retriever.multi_hop_spread(["alpha"], max_hops=3, decay=0.5)

        self.assertAlmostEqual(scores.get("alpha", 0.0), 1.0, places=6)
        self.assertAlmostEqual(scores.get("beta", 0.0), 0.5, places=6)
        self.assertAlmostEqual(scores.get("gamma", 0.0), 0.25, places=6)
        self.assertAlmostEqual(scores.get("delta", 0.0), 0.125, places=6)

    def test_multi_hop_spread_respects_max_hops(self):
        cg = ConceptGraph()
        cg.link_memory_concept(1, "a", "demo")
        cg.link_memory_concept(1, "b", "demo")
        cg.link_memory_concept(2, "b", "demo")
        cg.link_memory_concept(2, "c", "demo")

        retriever = GraphRetriever(cg)
        scores = retriever.multi_hop_spread(["a"], max_hops=1, decay=0.5)

        self.assertIn("a", scores)
        self.assertIn("b", scores)
        self.assertNotIn("c", scores)
        self.assertAlmostEqual(scores.get("b", 0.0), 0.5, places=6)

    def test_multi_hop_spread_edge_weights(self):
        cg = ConceptGraph()
        triple_index = TripleIndexStub({
            "seed": [
                ("related", "factual", 1.0),
                ("adjacent", "derived", 1.0),
            ]
        })
        retriever = GraphRetriever(cg, triple_index=triple_index)
        scores = retriever.multi_hop_spread(
            ["seed"],
            max_hops=1,
            decay=0.5,
            edge_type_weights={"co_occurrence": 1.0, "factual": 1.5, "derived": 0.5},
        )

        self.assertAlmostEqual(scores.get("seed", 0.0), 1.0, places=6)
        self.assertAlmostEqual(scores.get("related", 0.0), 0.75, places=6)
        self.assertAlmostEqual(scores.get("adjacent", 0.0), 0.25, places=6)

    def test_extract_query_concepts_includes_expansion(self):
        retriever = GraphRetriever(ConceptGraph())
        concepts = retriever.extract_query_concepts("llm inference")
        self.assertIn("ai_ml", concepts)

    def test_dual_path_retrieve_includes_bm25_and_graph_paths(self):
        cg = ConceptGraph()
        cg.link_memory_concept(1, "llm", "ai_ml")
        cg.link_memory_concept(1, "python", "software")
        cg.link_memory_concept(2, "python", "software")
        cg.link_memory_concept(2, "runtime", "software")

        retriever = GraphRetriever(cg)
        memories = [
            {"id": 1, "content": "LLM architecture runs on Python examples."},
            {"id": 2, "content": "Python runtime packaging."},
            {"id": 3, "content": "Unrelated meeting notes."},
        ]

        ranked = retriever.dual_path_retrieve("llm", memories, limit=10)
        ids = [memory_id for memory_id, _ in ranked]

        self.assertIn(1, ids)
        self.assertIn(2, ids)
        self.assertEqual(ids[0], 1)

    def test_dual_path_retrieve_limit_respected(self):
        cg = ConceptGraph()
        cg.link_memory_concept(1, "query", "demo")
        cg.link_memory_concept(2, "query", "demo")
        cg.link_memory_concept(3, "query", "demo")

        retriever = GraphRetriever(cg)
        memories = [
            {"id": i, "content": f"query phrase {i}"}
            for i in (1, 2, 3)
        ]
        ranked = retriever.dual_path_retrieve("query", memories, limit=2)
        self.assertEqual(len(ranked), 2)

    def test_dual_path_retrieve_empty_query(self):
        retriever = GraphRetriever(ConceptGraph())
        memories = [{"id": 1, "content": "hello world"}]
        self.assertEqual(retriever.dual_path_retrieve("", memories), [])

    def test_dual_path_retrieval_finds_bm25_misses(self):
        store = Synapse(":memory:")
        try:
            anchor = store.remember("LLM concepts appear in this note.", deduplicate=False)
            hidden = store.remember("Python runtime details for deployments.", deduplicate=False)

            classic = store.recall("llm", limit=10)
            graph = store.recall("llm", limit=10, retrieval_mode="graph")

            classic_ids = [m.id for m in classic]
            graph_ids = [m.id for m in graph]
            self.assertIn(anchor.id, classic_ids)
            self.assertNotIn(hidden.id, classic_ids)
            self.assertIn(hidden.id, graph_ids)
            self.assertIn(anchor.id, graph_ids)
        finally:
            store.close()

    def test_graph_mode_uses_temporal_and_explain_fields(self):
        store = Synapse(":memory:")
        try:
            target = store.remember("LLM evaluation harness uses Python.", deduplicate=False)
            results = store.recall("llm", limit=5, retrieval_mode="graph", explain=True)
            self.assertTrue(results)
            self.assertIsNotNone(results[0].score_breakdown)
            self.assertIn("graph_retriever", results[0].score_breakdown.match_sources)
            self.assertGreater(results[0].score_breakdown.temporal_score, 0.0)
            self.assertEqual(results[0].id, target.id)
        finally:
            store.close()

    def test_graph_mode_unsupported_mode_raises(self):
        store = Synapse(":memory:")
        try:
            store.remember("Simple note", deduplicate=False)
            with self.assertRaises(Exception):
                store.recall("Simple note", retrieval_mode="fuzzy")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
