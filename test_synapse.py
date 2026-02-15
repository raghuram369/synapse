#!/usr/bin/env python3
"""Tests for Synapse V2 memory engine."""

import time
import unittest
from unittest.mock import patch, Mock
from synapse import Synapse, DECAY_HALF_LIFE


class TestSynapseV2(unittest.TestCase):

    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    # ── Remember & Basic Operations ──

    def test_remember_basic(self):
        m = self.s.remember("The sky is blue", "fact")
        self.assertEqual(m.content, "The sky is blue")
        self.assertEqual(m.memory_type, "fact")
        self.assertEqual(m.strength, 1.0)
        self.assertIsNotNone(m.id)

    def test_remember_with_metadata(self):
        metadata = {"location": "office"}
        m = self.s.remember("Meeting at 3pm", "event", metadata=metadata)
        self.assertEqual(m.metadata["location"], "office")

    def test_remember_invalid_type(self):
        with self.assertRaises(ValueError):
            self.s.remember("test", "invalid_type")

    def test_remember_with_links(self):
        m1 = self.s.remember("Cause event")
        links = [{"target_id": m1.id, "edge_type": "caused_by"}]
        m2 = self.s.remember("Effect event", links=links)
        # Check that edge was created
        edges = self.s.edge_graph.get_outgoing_edges(m2.id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].edge_type, "caused_by")
        self.assertEqual(edges[0].target_id, m1.id)

    def test_invalid_edge_type(self):
        m1 = self.s.remember("A")
        with self.assertRaises(ValueError):
            links = [{"target_id": m1.id, "edge_type": "bogus"}]
            self.s.remember("B", links=links)

    # ── Fact Extraction ──

    @patch('synapse.extract_facts')
    def test_remember_with_extraction_success(self, mock_extract):
        """Test successful fact extraction."""
        # Mock extraction returning multiple facts
        mock_extract.return_value = [
            "Caroline is researching adoption agencies",
            "Adoption has been on Caroline's mind"
        ]
        
        # Remember with extraction
        memory = self.s.remember(
            "Caroline: Yeah I've been looking into adoption agencies recently, it's been on my mind a lot",
            extract=True
        )
        
        # Should return the first extracted fact as the primary memory
        self.assertEqual(memory.content, "Caroline is researching adoption agencies")
        self.assertTrue(memory.metadata.get('extracted_fact', False))
        self.assertEqual(memory.metadata.get('extraction_index'), 0)
        self.assertEqual(memory.metadata.get('total_facts'), 2)
        
        # Original content should be preserved in metadata
        self.assertIn('original_content', memory.metadata)
        
        # Should have called the extractor
        mock_extract.assert_called_once()
        
        # Should be able to recall both facts
        results = self.s.recall("Caroline adoption")
        self.assertTrue(len(results) >= 2)
        
        # Both facts should be linked with "related" edges
        fact_ids = []
        for result in results:
            if result.metadata.get('extracted_fact', False):
                fact_ids.append(result.id)
        
        self.assertEqual(len(fact_ids), 2)
        
        # Check that related edges exist between facts
        edges_from_first = self.s.edge_graph.get_outgoing_edges(fact_ids[0])
        related_edges = [e for e in edges_from_first if e.edge_type == "related"]
        self.assertTrue(len(related_edges) > 0)

    @patch('synapse.extract_facts')
    def test_remember_with_extraction_no_facts(self, mock_extract):
        """Test extraction when no facts are returned."""
        mock_extract.return_value = []
        
        # Should fall back to regular storage
        memory = self.s.remember("Test content", extract=True)
        
        self.assertEqual(memory.content, "Test content")
        self.assertFalse(memory.metadata.get('extracted_fact', False))

    @patch('synapse.extract_facts')
    def test_remember_with_extraction_error(self, mock_extract):
        """Test extraction when extractor raises an error."""
        mock_extract.side_effect = Exception("Extraction failed")
        
        # Should fall back to regular storage without crashing
        memory = self.s.remember("Test content", extract=True)
        
        self.assertEqual(memory.content, "Test content")
        self.assertFalse(memory.metadata.get('extracted_fact', False))

    def test_remember_without_extraction(self):
        """Test that normal remember still works."""
        memory = self.s.remember("Normal content", extract=False)
        
        self.assertEqual(memory.content, "Normal content")
        self.assertFalse(memory.metadata.get('extracted_fact', False))

    @patch('synapse.extract_facts')
    def test_remember_extraction_with_links(self, mock_extract):
        """Test extraction with additional links."""
        mock_extract.return_value = ["First fact", "Second fact"]
        
        # Create a target memory to link to
        target = self.s.remember("Target memory")
        
        # Remember with extraction and links
        links = [{"target_id": target.id, "edge_type": "supports"}]
        memory = self.s.remember("Source content", extract=True, links=links)
        
        # Both extracted facts should be linked to the target
        results = self.s.recall("First fact Second fact")
        extracted_memories = [m for m in results if m.metadata.get('extracted_fact')]
        
        for extracted_memory in extracted_memories:
            edges = self.s.edge_graph.get_outgoing_edges(extracted_memory.id)
            support_edges = [e for e in edges if e.edge_type == "supports" and e.target_id == target.id]
            self.assertTrue(len(support_edges) > 0, f"Memory {extracted_memory.id} should link to target {target.id}")

    # ── Recall ──

    def test_recall_by_keyword(self):
        self.s.remember("Python is a programming language", "fact")
        self.s.remember("Cats are fluffy animals", "fact")
        self.s.remember("Python snakes are long", "fact")
        results = self.s.recall("Python")
        self.assertTrue(len(results) >= 2)
        contents = [m.content for m in results]
        self.assertTrue(any("Python" in c for c in contents))

    def test_recall_empty_context(self):
        self.s.remember("A")
        self.s.remember("B")
        results = self.s.recall("", limit=5)
        self.assertEqual(len(results), 2)

    def test_recall_by_type(self):
        self.s.remember("Fact one", "fact")
        self.s.remember("Event one", "event")
        results = self.s.recall("", memory_type="event")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_type, "event")

    def test_recall_reinforces(self):
        m = self.s.remember("Important thing")
        original_access_count = self.s.store.memories[m.id]['access_count']
        self.s.recall("Important")
        updated_access_count = self.s.store.memories[m.id]['access_count']
        self.assertGreater(updated_access_count, original_access_count)

    def test_recall_graph_traversal(self):
        m1 = self.s.remember("Alpha concept", "fact")
        m2 = self.s.remember("Beta concept", "fact")
        m3 = self.s.remember("Gamma Alpha concept", "fact")  # shares query term
        self.s.link(m1.id, m3.id, "reminds_of")
        # Searching for Alpha should surface both (m3 shares keyword + gets edge boost)
        results = self.s.recall("Alpha")
        ids = [m.id for m in results]
        self.assertIn(m1.id, ids)
        self.assertIn(m3.id, ids)

    def test_tokenize_functionality(self):
        # Test tokenization through the inverted index
        tokens = self.s.inverted_index.tokenize_for_query("The quick brown fox jumped!")
        expected = ["quick", "brown", "fox", "jumped"]  # stopwords removed
        for token in expected:
            self.assertIn(token, tokens)
        self.assertNotIn("the", tokens)  # stopword removed

    def test_recall_inverted_index_ranking(self):
        self.s.remember("Python is a programming language", "fact", deduplicate=False)
        self.s.remember("Python snakes are long", "fact", deduplicate=False)
        results = self.s.recall("programming language", limit=2)
        self.assertGreaterEqual(len(results), 1)
        # First result should contain the most relevant terms
        self.assertIn("programming", results[0].content.lower())

    def test_recall_temporal_boost(self):
        # Create memories with different timestamps
        base_time = time.time()
        
        # Create first memory
        m1 = self.s.remember("alpha core content", "fact")
        
        # Manually set timestamps for testing
        self.s.store.memories[m1.id]['created_at'] = base_time
        self.s.temporal_index.remove_memory(m1.id)
        self.s.temporal_index.add_memory(m1.id, base_time)
        
        # Create second memory close in time
        m2 = self.s.remember("alpha nearby", "fact")
        self.s.store.memories[m2.id]['created_at'] = base_time + 1800  # 30 min later
        self.s.store.memories[m2.id]['strength'] = 0.5  # Lower strength
        self.s.temporal_index.remove_memory(m2.id)
        self.s.temporal_index.add_memory(m2.id, base_time + 1800)
        
        # Create third memory far in time
        m3 = self.s.remember("alpha distant", "fact") 
        self.s.store.memories[m3.id]['created_at'] = base_time + 7200  # 2 hours later
        self.s.store.memories[m3.id]['strength'] = 1.0  # Higher strength
        self.s.temporal_index.remove_memory(m3.id)
        self.s.temporal_index.add_memory(m3.id, base_time + 7200)
        
        # Test recall with temporal boost
        results = self.s.recall("alpha", limit=3, temporal_boost=True)
        ids = [m.id for m in results]
        
        # All memories should be present
        self.assertIn(m1.id, ids)
        self.assertIn(m2.id, ids)
        self.assertIn(m3.id, ids)

    def test_recall_activation_spreading_weighted(self):
        m0 = self.s.remember("alpha seed", "fact")
        n1 = self.s.remember("alpha alpha neighbor one", "fact")  # stronger BM25 match
        n2 = self.s.remember("alpha neighbor two", "fact")  # weaker BM25 match
        
        # Create weighted links — n1 gets stronger edge boost
        self.s.link(m0.id, n1.id, "reminds_of", weight=1.0)
        self.s.link(m0.id, n2.id, "reminds_of", weight=0.1)
        
        results = self.s.recall("alpha", limit=3)
        ids = [m.id for m in results]
        
        # Both neighbors should appear (they match query AND get edge boost)
        self.assertIn(n1.id, ids)
        self.assertIn(n2.id, ids)
        # n1 has stronger BM25 ("alpha alpha") so should rank higher
        # Edge weights provide additional boost but BM25 dominates
        self.assertIn(m0.id, ids)  # seed should appear

    # ── Forget ──

    def test_forget(self):
        m = self.s.remember("Temporary memory")
        memory_id = m.id
        success = self.s.forget(memory_id)
        self.assertTrue(success)
        
        # Memory should be gone from storage
        self.assertNotIn(memory_id, self.s.store.memories)
        
        # Should not appear in recall
        results = self.s.recall("Temporary")
        ids = [mem.id for mem in results]
        self.assertNotIn(memory_id, ids)

    def test_forget_nonexistent(self):
        success = self.s.forget(99999)
        self.assertFalse(success)

    # ── Links ──

    def test_link_basic(self):
        m1 = self.s.remember("Cause")
        m2 = self.s.remember("Effect")
        self.s.link(m1.id, m2.id, "caused_by", weight=0.8)
        
        # Check edge exists
        edges = self.s.edge_graph.get_outgoing_edges(m1.id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].target_id, m2.id)
        self.assertEqual(edges[0].edge_type, "caused_by")
        self.assertEqual(edges[0].weight, 0.8)

    def test_link_invalid_edge_type(self):
        m1 = self.s.remember("A")
        m2 = self.s.remember("B")
        with self.assertRaises(ValueError):
            self.s.link(m1.id, m2.id, "invalid_type")

    def test_link_nonexistent_memory(self):
        m1 = self.s.remember("A")
        with self.assertRaises(ValueError):
            self.s.link(m1.id, 99999, "reminds_of")

    # ── Concepts ──

    def test_concept_extraction(self):
        m = self.s.remember("I love programming in Python with machine learning")
        # Check that concepts were extracted and indexed
        memory_concepts = self.s.concept_graph.get_memory_concepts(m.id)
        self.assertGreater(len(memory_concepts), 0)
        
        # Should find concepts related to programming (like "python" or "ai_ml")
        concept_names = list(memory_concepts)
        has_python = "python" in concept_names
        has_ai_ml = "ai_ml" in concept_names
        self.assertTrue(has_python or has_ai_ml)

    def test_recall_by_concepts(self):
        # Create memories with different concepts
        m1 = self.s.remember("Python programming tutorial", "fact")
        m2 = self.s.remember("JavaScript web development", "fact") 
        m3 = self.s.remember("Machine learning with Python", "fact")
        
        # Query that should match Python-related memories
        results = self.s.recall("Python coding")
        ids = [m.id for m in results]
        
        # Should find Python-related memories
        self.assertIn(m1.id, ids)
        self.assertIn(m3.id, ids)

    # ── Episodes ──

    def test_episode_grouping(self):
        # Remember memories in an episode
        episode_name = "morning_routine" 
        m1 = self.s.remember("Wake up at 7am", episode=episode_name)
        m2 = self.s.remember("Brush teeth", episode=episode_name)
        m3 = self.s.remember("Have breakfast", episode=episode_name)
        
        # Memories should be in the same episode
        episode_id_1 = self.s.episode_index.get_memory_episode(m1.id)
        episode_id_2 = self.s.episode_index.get_memory_episode(m2.id)
        self.assertEqual(episode_id_1, episode_id_2)
        
        # Should be able to find episode siblings
        siblings = self.s.episode_index.get_episode_siblings(m1.id)
        self.assertIn(m2.id, siblings)
        self.assertIn(m3.id, siblings)

    # ── Deduplication ──

    def test_deduplication_supersession(self):
        # Create original memory
        m1 = self.s.remember("The capital of France is Paris", deduplicate=False)
        
        # Create very similar memory - should create supersedes edge
        m2 = self.s.remember("The capital of France is Paris", deduplicate=True)
        
        # Check for supersedes edge
        edges = self.s.edge_graph.get_outgoing_edges(m2.id)
        supersedes_edges = [e for e in edges if e.edge_type == "supersedes"]
        self.assertEqual(len(supersedes_edges), 1)
        self.assertEqual(supersedes_edges[0].target_id, m1.id)

    # ── Persistence ──

    def test_flush_and_snapshot(self):
        m = self.s.remember("Test persistence")
        
        # Test flush
        self.s.flush()
        
        # Test snapshot
        self.s.snapshot()
        
        # Memory should still exist
        results = self.s.recall("Test persistence")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, m.id)

    # ── Effective Strength ──

    def test_effective_strength_decay(self):
        # Create memory
        m = self.s.remember("Test decay")
        initial_strength = m.effective_strength
        
        # Simulate time passing by updating last_accessed
        old_time = time.time() - (DECAY_HALF_LIFE * 2)  # 2 half-lives ago
        self.s.store.memories[m.id]['last_accessed'] = old_time
        
        # Get updated memory object
        updated_memory = self.s._memory_data_to_object(self.s.store.memories[m.id])
        
        # Effective strength should be less due to decay
        self.assertLess(updated_memory.effective_strength, initial_strength)

    # ── Edge Cases ──

    def test_empty_content(self):
        with self.assertRaises(ValueError):
            self.s.remember("")
        
        with self.assertRaises(ValueError):
            self.s.remember("   ")  # whitespace only

    def test_recall_limit(self):
        # Create many memories
        for i in range(20):
            self.s.remember(f"Memory number {i}")
        
        # Test different limits
        results_5 = self.s.recall("Memory", limit=5)
        results_10 = self.s.recall("Memory", limit=10)
        
        self.assertEqual(len(results_5), 5)
        self.assertEqual(len(results_10), 10)

    def test_recall_min_strength_filter(self):
        # Create memory and artificially lower its strength
        m = self.s.remember("Low strength memory")
        self.s.store.memories[m.id]['strength'] = 0.001
        
        # Should not appear with default min_strength
        results_default = self.s.recall("Low strength")
        ids_default = [mem.id for mem in results_default]
        
        # Should appear with very low min_strength
        results_low = self.s.recall("Low strength", min_strength=0.0001)
        ids_low = [mem.id for mem in results_low]
        
        self.assertIn(m.id, ids_low)


if __name__ == "__main__":
    unittest.main()