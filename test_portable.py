"""
Comprehensive tests for the Synapse Portable Memory Format.

Tests: export/import round-trip, merge conflicts, partial exports,
format versioning, large file handling, JSON fallback, CRC integrity,
deduplication, provenance tracking, diff.
"""

import json
import os
import sys
import tempfile
import time
import struct
import unittest

# All modules are co-located now
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exceptions import SynapseFormatError

from portable import (
    SynapseWriter, SynapseReader, export_synapse, import_synapse,
    merge_synapse, inspect_synapse, diff_synapse, export_json_fallback,
    _pack_header, _unpack_header, _tokenize, _jaccard,
    MAGIC, FORMAT_VERSION_MAJOR, HEADER_SIZE,
    FLAG_PARTIAL, FLAG_HAS_PROVENANCE,
    SECTION_MEMORIES, SECTION_METADATA,
)


def _make_synapse():
    """Create a fresh in-memory Synapse instance."""
    from synapse import Synapse
    return Synapse(":memory:")


def _populate(s, n=5):
    """Add n memories to a Synapse instance."""
    memories = []
    base_time = 1700000000.0
    for i in range(n):
        m = s.remember(f"Memory number {i}: The quick brown fox jumps over the lazy dog {i}",
                       memory_type="fact")
        memories.append(m)
    return memories


class TestBinaryFormat(unittest.TestCase):
    """Test low-level binary format operations."""
    
    def test_header_roundtrip(self):
        header = _pack_header(flags=FLAG_PARTIAL, timestamp=1700000000.0,
                              num_sections=5, crc=0xDEADBEEF)
        self.assertEqual(len(header), HEADER_SIZE)
        parsed = _unpack_header(header)
        self.assertFalse(parsed['json_fallback'])
        self.assertEqual(parsed['version_major'], FORMAT_VERSION_MAJOR)
        self.assertEqual(parsed['flags'] & FLAG_PARTIAL, FLAG_PARTIAL)
        self.assertEqual(parsed['num_sections'], 5)
        self.assertEqual(parsed['crc'], 0xDEADBEEF)
    
    def test_magic_bytes(self):
        header = _pack_header(0, 0.0, 0, 0)
        self.assertTrue(header.startswith(MAGIC))
    
    def test_invalid_magic_raises(self):
        bad = b'XXXX' + b'\x00' * 28
        with self.assertRaises(SynapseFormatError):
            _unpack_header(bad)
    
    def test_json_fallback_detection(self):
        data = b'{"metadata": {}}' + b'\x00' * 20
        parsed = _unpack_header(data)
        self.assertTrue(parsed['json_fallback'])
    
    def test_future_version_raises(self):
        header = bytearray(_pack_header(0, 0.0, 0, 0))
        header[4] = 99  # major version 99
        with self.assertRaises(SynapseFormatError):
            _unpack_header(bytes(header))


class TestWriterReader(unittest.TestCase):
    """Test SynapseWriter and SynapseReader."""
    
    def test_write_read_empty(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            writer = SynapseWriter(path, source_agent="test")
            writer.write()
            
            reader = SynapseReader(path)
            self.assertFalse(reader.is_json_fallback)
            self.assertTrue(reader.verify_crc())
            
            meta = reader.metadata()
            self.assertEqual(meta['source_agent'], 'test')
            self.assertEqual(meta['memory_count'], 0)
            
            self.assertEqual(list(reader.iter_memories()), [])
        finally:
            os.unlink(path)
    
    def test_write_read_memories(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            writer = SynapseWriter(path, source_agent="alpha")
            for i in range(10):
                writer.add_memory({
                    'id': i, 'content': f'Memory {i}',
                    'memory_type': 'fact', 'strength': 1.0,
                    'created_at': 1700000000.0 + i,
                })
            writer.add_edge({'source_id': 0, 'target_id': 1, 'edge_type': 'related', 'weight': 0.5})
            writer.add_concept({'name': 'test', 'category': 'general', 'memory_ids': [0, 1]})
            writer.add_episode({'id': 1, 'name': 'ep1', 'started_at': 1700000000.0, 'ended_at': 1700001000.0, 'memory_ids': [0, 1]})
            writer.write()
            
            reader = SynapseReader(path)
            self.assertTrue(reader.verify_crc())
            
            memories = list(reader.iter_memories())
            self.assertEqual(len(memories), 10)
            self.assertEqual(memories[0]['content'], 'Memory 0')
            
            edges = list(reader.iter_edges())
            self.assertEqual(len(edges), 1)
            
            concepts = list(reader.iter_concepts())
            self.assertEqual(len(concepts), 1)
            
            episodes = list(reader.iter_episodes())
            self.assertEqual(len(episodes), 1)
            
            meta = reader.metadata()
            self.assertEqual(meta['memory_count'], 10)
        finally:
            os.unlink(path)
    
    def test_crc_corruption_detected(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            writer = SynapseWriter(path)
            writer.add_memory({'id': 1, 'content': 'test', 'created_at': 0.0})
            writer.write()
            
            # Corrupt a byte in the data section
            with open(path, 'r+b') as f:
                f.seek(HEADER_SIZE + 100)
                f.write(b'\xFF')
            
            reader = SynapseReader(path)
            self.assertFalse(reader.verify_crc())
        finally:
            os.unlink(path)


class TestExportImport(unittest.TestCase):
    """Test full export/import roundtrip through Synapse."""
    
    def test_full_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            mems = _populate(s1, 5)
            
            export_synapse(s1, path, source_agent="test-agent")
            
            s2 = _make_synapse()
            stats = import_synapse(s2, path)
            
            self.assertEqual(stats['memories'], 5)
            self.assertEqual(len(s2.store.memories), 5)
            
            # Verify content preserved
            contents_1 = sorted(m['content'] for m in s1.store.memories.values())
            contents_2 = sorted(m['content'] for m in s2.store.memories.values())
            self.assertEqual(contents_1, contents_2)
        finally:
            os.unlink(path)
    
    def test_partial_export_by_time(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s = _make_synapse()
            # Manually set created_at times
            m1 = s.remember("Old memory from 2023")
            s.store.memories[m1.id]['created_at'] = 1672531200.0  # 2023-01-01
            
            m2 = s.remember("New memory from 2024")
            s.store.memories[m2.id]['created_at'] = 1704067200.0  # 2024-01-01
            
            m3 = s.remember("Newest memory from 2025")
            s.store.memories[m3.id]['created_at'] = 1735689600.0  # 2025-01-01
            
            export_synapse(s, path, since="2024-01-01")
            
            reader = SynapseReader(path)
            memories = list(reader.iter_memories())
            self.assertEqual(len(memories), 2)  # m2 and m3
            
            meta = reader.metadata()
            self.assertEqual(meta['filter']['type'], 'partial')
        finally:
            os.unlink(path)
    
    def test_partial_export_by_concepts(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s = _make_synapse()
            s.remember("I love Python programming")
            s.remember("The weather is nice today")
            s.remember("Python is a great language for AI")
            
            export_synapse(s, path, concepts=["python"])
            
            reader = SynapseReader(path)
            memories = list(reader.iter_memories())
            # Should only include memories linked to python concept
            for m in memories:
                self.assertTrue('python' in m['content'].lower() or 'programming' in m['content'].lower())
        finally:
            os.unlink(path)
    
    def test_import_deduplication(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            s1.remember("The quick brown fox jumps over the lazy dog")
            export_synapse(s1, path)
            
            # Import into instance that already has the same memory
            s2 = _make_synapse()
            s2.remember("The quick brown fox jumps over the lazy dog")
            
            stats = import_synapse(s2, path, deduplicate=True, similarity_threshold=0.8)
            
            self.assertEqual(stats['skipped_duplicates'], 1)
            self.assertEqual(len(s2.store.memories), 1)  # should not have duplicated
        finally:
            os.unlink(path)
    
    def test_import_no_dedup(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            s1.remember("The quick brown fox jumps over the lazy dog")
            export_synapse(s1, path)
            
            s2 = _make_synapse()
            s2.remember("The quick brown fox jumps over the lazy dog")
            
            stats = import_synapse(s2, path, deduplicate=False)
            self.assertEqual(len(s2.store.memories), 2)  # both copies
        finally:
            os.unlink(path)


class TestMerge(unittest.TestCase):
    """Test merge semantics."""
    
    def test_merge_new_memories(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            # Create source file
            s1 = _make_synapse()
            s1.remember("Alpha agent knows about cats")
            s1.remember("Alpha agent knows about dogs")
            export_synapse(s1, path, source_agent="alpha")
            
            # Merge into target with different memories
            s2 = _make_synapse()
            s2.remember("Beta agent knows about birds")
            
            stats = merge_synapse(s2, path)
            
            self.assertEqual(stats['memories_added'], 2)
            self.assertEqual(len(s2.store.memories), 3)
        finally:
            os.unlink(path)
    
    def test_merge_newer_wins(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            # Source has newer version of similar memory
            s1 = _make_synapse()
            m = s1.remember("The capital of France is Paris and it is beautiful")
            s1.store.memories[m.id]['created_at'] = time.time() + 1000  # future
            export_synapse(s1, path, source_agent="newer")
            
            # Target has older version
            s2 = _make_synapse()
            m2 = s2.remember("The capital of France is Paris and it is lovely")
            s2.store.memories[m2.id]['created_at'] = time.time() - 1000  # past
            
            stats = merge_synapse(s2, path, conflict_resolution="newer_wins",
                                  similarity_threshold=0.5)
            
            self.assertEqual(stats['memories_updated'], 1)
        finally:
            os.unlink(path)
    
    def test_merge_keep_both(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            s1.remember("The capital of France is Paris and it is beautiful")
            export_synapse(s1, path, source_agent="source-a")
            
            s2 = _make_synapse()
            s2.remember("The capital of France is Paris and it is lovely")
            
            stats = merge_synapse(s2, path, conflict_resolution="keep_both",
                                  similarity_threshold=0.5)
            
            self.assertEqual(stats['memories_added'], 1)
            self.assertEqual(len(s2.store.memories), 2)
            
            # Check provenance tag
            for mid, mdata in s2.store.memories.items():
                meta = mdata.get('metadata', '{}')
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if 'merged_from' in meta:
                    self.assertIn('merge_duplicate_of', meta)
        finally:
            os.unlink(path)
    
    def test_merge_concept_graph_union(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            s1.remember("Python is great for machine learning")
            export_synapse(s1, path)
            
            s2 = _make_synapse()
            s2.remember("Rust is great for systems programming")
            
            merge_synapse(s2, path, similarity_threshold=0.95)
            
            # Both concept sets should exist
            concept_names = set(s2.concept_graph.concepts.keys())
            # Should have concepts from both memories
            self.assertTrue(len(concept_names) > 0)
        finally:
            os.unlink(path)


class TestInspect(unittest.TestCase):
    """Test file inspection."""
    
    def test_inspect(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s = _make_synapse()
            _populate(s, 10)
            export_synapse(s, path, source_agent="inspector")
            
            info = inspect_synapse(path)
            
            self.assertEqual(info['memory_count'], 10)
            self.assertTrue(info['crc_valid'])
            self.assertEqual(info['source_agent'], 'inspector')
            self.assertEqual(info['format'], 'binary')
            self.assertIn('file_size', info)
            self.assertIn('version', info)
        finally:
            os.unlink(path)


class TestDiff(unittest.TestCase):
    """Test diff between two files."""
    
    def test_diff_identical(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as fa, \
             tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as fb:
            path_a, path_b = fa.name, fb.name
        try:
            s = _make_synapse()
            _populate(s, 5)
            export_synapse(s, path_a, source_agent="a")
            export_synapse(s, path_b, source_agent="b")
            
            result = diff_synapse(path_a, path_b)
            self.assertEqual(result['shared'], 5)
            self.assertEqual(result['only_in_a'], 0)
            self.assertEqual(result['only_in_b'], 0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)
    
    def test_diff_disjoint(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as fa, \
             tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as fb:
            path_a, path_b = fa.name, fb.name
        try:
            s1 = _make_synapse()
            s1.remember("Alpha unique content about quantum physics")
            export_synapse(s1, path_a)
            
            s2 = _make_synapse()
            s2.remember("Beta unique content about cooking recipes")
            export_synapse(s2, path_b)
            
            result = diff_synapse(path_a, path_b)
            self.assertEqual(result['shared'], 0)
            self.assertEqual(result['only_in_a'], 1)
            self.assertEqual(result['only_in_b'], 1)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestJSONFallback(unittest.TestCase):
    """Test JSON fallback format."""
    
    def test_json_export_import(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            _populate(s1, 3)
            export_json_fallback(s1, path, source_agent="json-test")
            
            # Verify it's valid JSON
            with open(path) as f:
                data = json.load(f)
            self.assertIn('memories', data)
            self.assertEqual(len(data['memories']), 3)
            
            # Read via SynapseReader (JSON fallback)
            reader = SynapseReader(path)
            self.assertTrue(reader.is_json_fallback)
            memories = list(reader.iter_memories())
            self.assertEqual(len(memories), 3)
        finally:
            os.unlink(path)


class TestLargeFile(unittest.TestCase):
    """Test with larger datasets."""
    
    def test_1000_memories(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            for i in range(1000):
                s1.store.insert_memory({
                    'content': f"Large dataset memory entry number {i} with unique content {i * 37}",
                    'memory_type': 'fact', 'strength': 1.0, 'access_count': 0,
                    'created_at': 1700000000.0 + i, 'last_accessed': 1700000000.0 + i,
                    'metadata': '{}', 'consolidated': False, 'summary_of': '[]',
                })
            
            export_synapse(s1, path)
            
            info = inspect_synapse(path)
            self.assertEqual(info['memory_count'], 1000)
            
            s2 = _make_synapse()
            stats = import_synapse(s2, path, deduplicate=False)
            self.assertEqual(stats['memories'], 1000)
            self.assertEqual(len(s2.store.memories), 1000)
        finally:
            os.unlink(path)


class TestLSHPerformance(unittest.TestCase):
    """Test that LSH dedup is fast enough for large datasets."""
    
    def _fast_populate(self, s, n, prefix="mem"):
        """Insert n memories directly (bypassing slow NLP in remember())."""
        for i in range(n):
            mid = s.store.insert_memory({
                'content': f'{prefix} number {i} about topic {i * 7} with details {i * 13}',
                'memory_type': 'fact', 'strength': 1.0, 'access_count': 0,
                'created_at': 1700000000.0 + i, 'last_accessed': 1700000000.0 + i,
                'metadata': '{}', 'consolidated': False, 'summary_of': '[]',
            })
            s.inverted_index.add_document(mid, f'{prefix} number {i} about topic {i * 7} with details {i * 13}')
            s.temporal_index.add_memory(mid, 1700000000.0 + i)
    
    def test_merge_1000_memories_under_10_seconds(self):
        """Merge 1000+ memories and assert completes in <10 seconds."""
        import time as _time
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            # Create source with 1000 unique memories
            s1 = _make_synapse()
            self._fast_populate(s1, 1000, prefix="source")
            export_synapse(s1, path, source_agent="perf-test")
            
            # Target has 200 memories, some overlapping content
            s2 = _make_synapse()
            self._fast_populate(s2, 200, prefix="source")  # same prefix = overlapping
            
            start = _time.time()
            stats = merge_synapse(s2, path, similarity_threshold=0.85)
            elapsed = _time.time() - start
            
            self.assertLess(elapsed, 10.0, f"Merge took {elapsed:.1f}s, should be <10s")
            self.assertGreater(stats['memories_added'] + stats['memories_updated'] + stats['memories_skipped'], 0)
        finally:
            os.unlink(path)


class TestSimilarityHelpers(unittest.TestCase):
    """Test tokenization and similarity functions."""
    
    def test_tokenize(self):
        tokens = _tokenize("Hello World! This is a test-123.")
        self.assertEqual(tokens, {'hello', 'world', 'this', 'is', 'a', 'test', '123'})
    
    def test_jaccard_identical(self):
        a = {'hello', 'world'}
        self.assertAlmostEqual(_jaccard(a, a), 1.0)
    
    def test_jaccard_disjoint(self):
        a = {'hello', 'world'}
        b = {'foo', 'bar'}
        self.assertAlmostEqual(_jaccard(a, b), 0.0)
    
    def test_jaccard_partial(self):
        a = {'hello', 'world', 'foo'}
        b = {'hello', 'world', 'bar'}
        self.assertAlmostEqual(_jaccard(a, b), 2/4)
    
    def test_jaccard_empty(self):
        self.assertAlmostEqual(_jaccard(set(), set()), 1.0)
        self.assertAlmostEqual(_jaccard({'a'}, set()), 0.0)


class TestProvenance(unittest.TestCase):
    """Test provenance tracking."""
    
    def test_provenance_in_export(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s = _make_synapse()
            s.remember("Test provenance")
            export_synapse(s, path, source_agent="agent-007")
            
            reader = SynapseReader(path)
            memories = list(reader.iter_memories())
            self.assertEqual(len(memories), 1)
            self.assertIn('provenance', memories[0])
            self.assertEqual(memories[0]['provenance']['source'], 'agent-007')
            
            # Check flag
            self.assertTrue(reader.header['flags'] & FLAG_HAS_PROVENANCE)
        finally:
            os.unlink(path)


class TestEdgesAndEpisodes(unittest.TestCase):
    """Test that edges and episodes survive roundtrip."""
    
    def test_edges_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            m1 = s1.remember("First memory about dogs")
            m2 = s1.remember("Second memory about cats")
            s1.link(m1.id, m2.id, "related", 0.9)
            
            export_synapse(s1, path)
            
            s2 = _make_synapse()
            stats = import_synapse(s2, path, deduplicate=False)
            
            self.assertGreaterEqual(stats['edges'], 1)
            self.assertTrue(len(s2.store.edges) >= 1)
        finally:
            os.unlink(path)
    
    def test_episodes_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s1 = _make_synapse()
            s1.remember("Debugging the parser", episode="debug-session")
            s1.remember("Found the bug in line 42", episode="debug-session")
            
            export_synapse(s1, path)
            
            s2 = _make_synapse()
            stats = import_synapse(s2, path, deduplicate=False)
            
            self.assertGreaterEqual(stats['episodes'], 1)
        finally:
            os.unlink(path)


class TestLoadAll(unittest.TestCase):
    """Test load_all convenience method."""
    
    def test_load_all(self):
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            path = f.name
        try:
            s = _make_synapse()
            _populate(s, 3)
            export_synapse(s, path)
            
            reader = SynapseReader(path)
            data = reader.load_all()
            
            self.assertIn('memories', data)
            self.assertIn('edges', data)
            self.assertIn('concepts', data)
            self.assertIn('episodes', data)
            self.assertIn('metadata', data)
            self.assertEqual(len(data['memories']), 3)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
