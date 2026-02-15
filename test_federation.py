"""Tests for Synapse Phase 3 — Federated Agent Memory Network."""

import sys
import os
import time
import unittest
import tempfile
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federation.vector_clock import VectorClock
from federation.content_hash import memory_hash, edge_hash
from federation.merkle import MerkleTree
from federation.memory_object import FederatedMemory, FederatedEdge
from federation.store import FederatedStore
from federation.sync import SyncEngine, resolve_conflict
from federation.node import SynapseNode
from federation.server import SynapseServer
from federation.client import SyncClient


# ══════════════════════════════════════════════════════════════
# Vector Clock Tests
# ══════════════════════════════════════════════════════════════

class TestVectorClock(unittest.TestCase):
    def test_increment(self):
        vc = VectorClock()
        vc.increment("A")
        self.assertEqual(vc.to_dict(), {"A": 1})
        vc.increment("A")
        self.assertEqual(vc.to_dict(), {"A": 2})

    def test_merge(self):
        vc1 = VectorClock({"A": 3, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 4, "C": 2})
        vc1.merge(vc2)
        self.assertEqual(vc1.to_dict(), {"A": 3, "B": 4, "C": 2})

    def test_happened_before(self):
        vc1 = VectorClock({"A": 1})
        vc2 = VectorClock({"A": 2})
        self.assertTrue(vc1 < vc2)
        self.assertFalse(vc2 < vc1)

    def test_concurrent(self):
        vc1 = VectorClock({"A": 2, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 2})
        self.assertTrue(vc1.is_concurrent(vc2))

    def test_equality(self):
        vc1 = VectorClock({"A": 1, "B": 2})
        vc2 = VectorClock({"A": 1, "B": 2})
        self.assertEqual(vc1, vc2)

    def test_serialization(self):
        vc = VectorClock({"node1": 5, "node2": 3})
        j = vc.to_json()
        vc2 = VectorClock.from_json(j)
        self.assertEqual(vc, vc2)


# ══════════════════════════════════════════════════════════════
# Content Hash Tests
# ══════════════════════════════════════════════════════════════

class TestContentHash(unittest.TestCase):
    def test_deterministic(self):
        h1 = memory_hash("hello", "fact")
        h2 = memory_hash("hello", "fact")
        self.assertEqual(h1, h2)

    def test_different_content(self):
        h1 = memory_hash("hello", "fact")
        h2 = memory_hash("world", "fact")
        self.assertNotEqual(h1, h2)

    def test_different_type(self):
        h1 = memory_hash("hello", "fact")
        h2 = memory_hash("hello", "event")
        self.assertNotEqual(h1, h2)

    def test_metadata_order_irrelevant(self):
        h1 = memory_hash("hello", "fact", {"a": 1, "b": 2})
        h2 = memory_hash("hello", "fact", {"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_edge_hash_deterministic(self):
        h1 = edge_hash("abc", "def", "related", 1.0)
        h2 = edge_hash("abc", "def", "related", 1.0)
        self.assertEqual(h1, h2)


# ══════════════════════════════════════════════════════════════
# Merkle Tree Tests
# ══════════════════════════════════════════════════════════════

class TestMerkleTree(unittest.TestCase):
    def test_empty_tree(self):
        t = MerkleTree()
        self.assertEqual(len(t), 0)
        self.assertIsInstance(t.root, str)

    def test_add_items(self):
        t = MerkleTree()
        t.add("abc123")
        t.add("def456")
        self.assertEqual(len(t), 2)
        self.assertIn("abc123", t)

    def test_deterministic_root(self):
        t1 = MerkleTree()
        t2 = MerkleTree()
        items = ["aaa", "bbb", "ccc"]
        for item in items:
            t1.add(item)
        # Add in different order
        for item in reversed(items):
            t2.add(item)
        self.assertEqual(t1.root, t2.root)

    def test_different_content_different_root(self):
        t1 = MerkleTree()
        t2 = MerkleTree()
        t1.add("aaa")
        t2.add("bbb")
        self.assertNotEqual(t1.root, t2.root)

    def test_diff_buckets(self):
        t1 = MerkleTree()
        t2 = MerkleTree()

        # Same items
        for i in range(10):
            h = memory_hash(f"memory {i}", "fact")
            t1.add(h)
            t2.add(h)

        # t1 has an extra item
        extra = memory_hash("extra memory", "fact")
        t1.add(extra)

        diff = t1.diff_buckets(t2.bucket_hashes)
        self.assertTrue(len(diff) > 0)

        # The extra item should be in one of the differing buckets
        bucket = int(extra[:2], 16)
        self.assertIn(bucket, diff)

    def test_remove(self):
        t = MerkleTree()
        t.add("abc")
        t.add("def")
        root_with_both = t.root
        t.remove("def")
        self.assertNotEqual(t.root, root_with_both)
        self.assertEqual(len(t), 1)

    def test_add_many(self):
        t1 = MerkleTree()
        t2 = MerkleTree()
        items = {f"item{i}" for i in range(100)}
        for item in items:
            t1.add(item)
        t2.add_many(items)
        self.assertEqual(t1.root, t2.root)


# ══════════════════════════════════════════════════════════════
# Federated Memory Object Tests
# ══════════════════════════════════════════════════════════════

class TestFederatedMemory(unittest.TestCase):
    def test_create(self):
        mem = FederatedMemory("Python is great", "fact")
        self.assertTrue(len(mem.hash) == 64)  # SHA-256 hex
        self.assertEqual(mem.content, "Python is great")

    def test_serialization(self):
        mem = FederatedMemory(
            "test memory",
            memory_type="fact",
            metadata={"source": "test"},
            namespaces=["public"],
        )
        d = mem.to_dict()
        mem2 = FederatedMemory.from_dict(d)
        self.assertEqual(mem.hash, mem2.hash)
        self.assertEqual(mem.content, mem2.content)
        self.assertEqual(mem.namespaces, mem2.namespaces)

    def test_hash_mismatch_raises(self):
        d = FederatedMemory("test", "fact").to_dict()
        d["hash"] = "badhash"
        with self.assertRaises(ValueError):
            FederatedMemory.from_dict(d)

    def test_same_content_same_hash(self):
        m1 = FederatedMemory("hello world", "fact")
        m2 = FederatedMemory("hello world", "fact")
        self.assertEqual(m1.hash, m2.hash)


# ══════════════════════════════════════════════════════════════
# In-Process Sync Tests (no network)
# ══════════════════════════════════════════════════════════════

class TestInProcessSync(unittest.TestCase):
    def test_two_nodes_sync(self):
        """Two nodes with different memories sync and end up with all memories."""
        store1 = FederatedStore("node-A")
        store2 = FederatedStore("node-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        # Node A has 3 memories
        for i in range(3):
            mem = FederatedMemory(f"Memory A-{i}", "fact", origin_node="node-A",
                                  vclock=VectorClock().increment("node-A"))
            store1.add_memory(mem)

        # Node B has 2 different memories
        for i in range(2):
            mem = FederatedMemory(f"Memory B-{i}", "fact", origin_node="node-B",
                                  vclock=VectorClock().increment("node-B"))
            store2.add_memory(mem)

        self.assertEqual(len(store1.memories), 3)
        self.assertEqual(len(store2.memories), 2)

        # Bidirectional sync
        stats = engine1.sync_with(engine2)
        self.assertEqual(stats["pulled"], 2)  # A gets B's 2
        self.assertEqual(stats["pushed"], 3)  # B gets A's 3

        # Both should have 5
        self.assertEqual(len(store1.memories), 5)
        self.assertEqual(len(store2.memories), 5)

        # Same root hash
        self.assertEqual(store1.memory_tree.root, store2.memory_tree.root)

    def test_already_in_sync(self):
        """Syncing identical stores should be a no-op."""
        store1 = FederatedStore("node-A")
        store2 = FederatedStore("node-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        mem = FederatedMemory("shared memory", "fact")
        store1.add_memory(mem)
        store2.add_memory(FederatedMemory.from_dict(mem.to_dict()))

        stats = engine1.sync_with(engine2)
        self.assertEqual(stats["pulled"], 0)
        self.assertEqual(stats["pushed"], 0)

    def test_namespace_filtering(self):
        """Sync should respect namespace filtering."""
        store1 = FederatedStore("node-A")
        store2 = FederatedStore("node-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        # Node A: public and private memories
        pub = FederatedMemory("public info", "fact", namespaces=["public"],
                              vclock=VectorClock().increment("node-A"))
        priv = FederatedMemory("secret info", "fact", namespaces=["private"],
                               vclock=VectorClock().increment("node-A"))
        store1.add_memory(pub)
        store1.add_memory(priv)
        store1.share("public")

        # Sync only public namespace (push from A to B)
        stats = engine2.sync_with(engine1, namespaces={"public"}, bidirectional=False)

        # Node B should only have the public memory
        self.assertEqual(len(store2.memories), 1)
        self.assertIn(pub.hash, store2.memories)
        self.assertNotIn(priv.hash, store2.memories)

    def test_idempotent_sync(self):
        """Syncing twice should not duplicate memories."""
        store1 = FederatedStore("node-A")
        store2 = FederatedStore("node-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        mem = FederatedMemory("test", "fact", vclock=VectorClock().increment("A"))
        store1.add_memory(mem)

        engine1.sync_with(engine2)
        self.assertEqual(len(store2.memories), 1)

        engine1.sync_with(engine2)
        self.assertEqual(len(store2.memories), 1)

    def test_edge_sync(self):
        """Edges should sync along with memories."""
        store1 = FederatedStore("node-A")
        store2 = FederatedStore("node-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        m1 = FederatedMemory("fact one", "fact", vclock=VectorClock().increment("A"))
        m2 = FederatedMemory("fact two", "fact", vclock=VectorClock().increment("A"))
        store1.add_memory(m1)
        store1.add_memory(m2)

        edge = FederatedEdge(m1.hash, m2.hash, "related", 0.9,
                             vclock=VectorClock().increment("A"))
        store1.add_edge(edge)

        engine1.sync_with(engine2)

        self.assertEqual(len(store2.edges), 1)
        self.assertIn(edge.hash, store2.edges)


# ══════════════════════════════════════════════════════════════
# Conflict Resolution Tests
# ══════════════════════════════════════════════════════════════

class TestConflictResolution(unittest.TestCase):
    def test_later_wins(self):
        """When one clock dominates, that version wins."""
        vc1 = VectorClock({"A": 1})
        vc2 = VectorClock({"A": 2})

        m1 = FederatedMemory("old version", "fact", vclock=vc1, created_at=100)
        m2 = FederatedMemory("new version", "fact", vclock=vc2, created_at=200)

        winner = resolve_conflict(m1, m2)
        self.assertEqual(winner.content, "new version")

    def test_concurrent_lww(self):
        """Concurrent writes use last-writer-wins by timestamp."""
        vc1 = VectorClock({"A": 1})
        vc2 = VectorClock({"B": 1})

        m1 = FederatedMemory("from A", "fact", vclock=vc1, created_at=100)
        m2 = FederatedMemory("from B", "fact", vclock=vc2, created_at=200)

        self.assertTrue(vc1.is_concurrent(vc2))
        winner = resolve_conflict(m1, m2)
        self.assertEqual(winner.content, "from B")  # later timestamp

    def test_merged_vclock(self):
        """Winner should have merged vector clock."""
        vc1 = VectorClock({"A": 3, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 4})

        m1 = FederatedMemory("v1", "fact", vclock=vc1, created_at=100)
        m2 = FederatedMemory("v2", "fact", vclock=vc2, created_at=200)

        winner = resolve_conflict(m1, m2)
        self.assertEqual(winner.vclock.to_dict(), {"A": 3, "B": 4})

    def test_supersession_propagation(self):
        """Supersedes lists should merge."""
        m1 = FederatedMemory("v1", "fact", supersedes=["hash_old1"])
        m2 = FederatedMemory("v2", "fact", supersedes=["hash_old2"])

        winner = resolve_conflict(m1, m2)
        self.assertIn("hash_old1", winner.supersedes)
        self.assertIn("hash_old2", winner.supersedes)


# ══════════════════════════════════════════════════════════════
# SynapseNode High-Level API Tests
# ══════════════════════════════════════════════════════════════

class TestSynapseNode(unittest.TestCase):
    def test_remember_and_list(self):
        node = SynapseNode(node_id="test")
        node.remember("Python uses GIL", namespaces=["public"])
        node.remember("Secret formula", namespaces=["private"])

        all_mems = node.memories()
        self.assertEqual(len(all_mems), 2)

        pub = node.memories(namespace="public")
        self.assertEqual(len(pub), 1)
        self.assertEqual(pub[0].content, "Python uses GIL")

    def test_share_and_subscribe(self):
        node = SynapseNode(node_id="test")
        node.share("public")
        node.subscribe("http://peer:9470", "research")

        self.assertIn("public", node.store.shared_namespaces)
        self.assertIn("research", node.store.get_subscriptions("http://peer:9470"))

    def test_link(self):
        node = SynapseNode(node_id="test")
        m1 = node.remember("fact A")
        m2 = node.remember("fact B")
        edge = node.link(m1.hash, m2.hash, "related", 0.8)
        self.assertEqual(len(node.store.edges), 1)

    def test_forget(self):
        node = SynapseNode(node_id="test")
        m = node.remember("temporary")
        self.assertTrue(node.forget(m.hash))
        self.assertEqual(len(node.memories()), 0)

    def test_status(self):
        node = SynapseNode(node_id="test-node")
        node.remember("hello")
        status = node.status()
        self.assertEqual(status["node_id"], "test-node")
        self.assertEqual(status["memory_count"], 1)


# ══════════════════════════════════════════════════════════════
# HTTP Sync Tests (actual network, localhost)
# ══════════════════════════════════════════════════════════════

class TestHTTPSync(unittest.TestCase):
    def test_push_pull_over_http(self):
        """Test push/pull between two nodes over HTTP."""
        node1 = SynapseNode(node_id="server-node")
        node2 = SynapseNode(node_id="client-node")

        # Server has memories
        node1.remember("Memory from server 1", namespaces=["public"])
        node1.remember("Memory from server 2", namespaces=["public"])

        # Client has a memory
        node2.remember("Memory from client", namespaces=["public"])

        # Start server
        port = 19470  # high port to avoid conflicts
        node1.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)  # let server start

        try:
            # Client pulls from server
            result = node2.pull(f"http://127.0.0.1:{port}")
            self.assertEqual(result["pulled"], 2)
            self.assertEqual(len(node2.store.memories), 3)

            # Client pushes to server
            result = node2.push(f"http://127.0.0.1:{port}")
            self.assertEqual(result["pushed"], 1)
            self.assertEqual(len(node1.store.memories), 3)

            # Both in sync now
            self.assertEqual(
                node1.store.memory_tree.root,
                node2.store.memory_tree.root,
            )
        finally:
            node1.stop()

    def test_sync_bidirectional_http(self):
        """Test bidirectional sync over HTTP."""
        node1 = SynapseNode(node_id="alpha")
        node2 = SynapseNode(node_id="beta")

        node1.remember("alpha fact 1")
        node1.remember("alpha fact 2")
        node2.remember("beta fact 1")

        port = 19471
        node1.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            result = node2.sync(f"http://127.0.0.1:{port}")
            self.assertEqual(result["pulled"], 2)
            self.assertEqual(result["pushed"], 1)
            self.assertEqual(len(node1.store.memories), 3)
            self.assertEqual(len(node2.store.memories), 3)
        finally:
            node1.stop()

    def test_status_endpoint(self):
        """Test the /v1/status endpoint."""
        node = SynapseNode(node_id="status-test")
        node.remember("hello")
        port = 19472
        node.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            client = SyncClient(FederatedStore("tmp"))
            status = client.status(f"http://127.0.0.1:{port}")
            self.assertEqual(status["node_id"], "status-test")
            self.assertEqual(status["memory_count"], 1)
        finally:
            node.stop()


# ══════════════════════════════════════════════════════════════
# Authentication Tests
# ══════════════════════════════════════════════════════════════

class TestAuthentication(unittest.TestCase):
    def test_unauthorized_request_rejected(self):
        """Requests without token should get 401 when auth is enabled."""
        node = SynapseNode(node_id="secure", auth_token="secret123")
        node.remember("protected data")
        port = 19480
        node.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            # Client without token should fail
            from urllib.request import Request, urlopen
            from urllib.error import HTTPError
            req = Request(f"http://127.0.0.1:{port}/v1/status")
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 401)
        finally:
            node.stop()

    def test_authorized_request_succeeds(self):
        """Requests with correct token should succeed."""
        node = SynapseNode(node_id="secure", auth_token="secret123")
        node.remember("protected data")
        port = 19481
        node.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            client_node = SynapseNode(node_id="client")
            client_node.add_peer(f"http://127.0.0.1:{port}", token="secret123")
            result = client_node.pull(f"http://127.0.0.1:{port}")
            self.assertEqual(result["pulled"], 1)
        finally:
            node.stop()

    def test_open_mode_still_works(self):
        """Nodes without auth_token should accept all requests."""
        node = SynapseNode(node_id="open")
        node.remember("public data")
        port = 19482
        node.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            client = SynapseNode(node_id="anon")
            result = client.pull(f"http://127.0.0.1:{port}")
            self.assertEqual(result["pulled"], 1)
        finally:
            node.stop()

    def test_wrong_token_rejected(self):
        """Requests with wrong token should get 401."""
        node = SynapseNode(node_id="secure", auth_token="correct")
        node.remember("secret")
        port = 19483
        node.listen(port=port, host="127.0.0.1")
        time.sleep(0.2)

        try:
            from urllib.request import Request, urlopen
            from urllib.error import HTTPError
            req = Request(f"http://127.0.0.1:{port}/v1/status",
                         headers={"Authorization": "Bearer wrong"})
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 401)
        finally:
            node.stop()


# ══════════════════════════════════════════════════════════════
# Wire Format Integration Tests (Phase 2 <-> Phase 3)
# ══════════════════════════════════════════════════════════════

class TestPortableWireFormat(unittest.TestCase):
    def test_synapse_binary_export_import(self):
        """Test exporting and importing via .synapse binary format."""
        from federation.sync import SyncEngine
        store1 = FederatedStore("exporter")
        engine1 = SyncEngine(store1)

        for i in range(5):
            mem = FederatedMemory(f"binary format test {i}", "fact",
                                  vclock=VectorClock().increment("exporter"))
            store1.add_memory(mem)

        # Export as binary
        binary_data = engine1.export_synapse_binary(store1.all_hashes())
        self.assertIsNotNone(binary_data)
        self.assertTrue(len(binary_data) > 0)

        # Import into another store
        store2 = FederatedStore("importer")
        engine2 = SyncEngine(store2)
        result = engine2.import_synapse_binary(binary_data)
        self.assertEqual(result["memories"], 5)
        self.assertEqual(len(store2.memories), 5)

    def test_phase2_export_phase3_import(self):
        """Export from Phase 2 format, import via Phase 3 network sync."""
        try:
            from portable import SynapseWriter, SynapseReader
        except ImportError:
            self.skipTest("Phase 2 portable module not available")

        # Create a .synapse file with Phase 2 writer containing federated memory dicts
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.synapse', delete=False) as f:
            tmp_path = f.name

        try:
            writer = SynapseWriter(tmp_path, source_agent="phase2-exporter")
            # Write memories in federated format (with hash, vclock, etc.)
            test_memories = []
            for i in range(3):
                mem = FederatedMemory(f"cross-phase memory {i}", "fact",
                                      namespaces=["public"],
                                      vclock=VectorClock().increment("phase2"))
                writer.add_memory(mem.to_dict())
                test_memories.append(mem)
            writer.write()

            # Read the binary and import into Phase 3 store
            with open(tmp_path, 'rb') as f:
                binary_data = f.read()

            store = FederatedStore("phase3-node")
            engine = SyncEngine(store)
            result = engine.import_synapse_binary(binary_data)
            self.assertEqual(result["memories"], 3)

            # Verify content
            for mem in test_memories:
                self.assertIn(mem.hash, store.memories)
                self.assertEqual(store.memories[mem.hash].content, mem.content)
        finally:
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# Persistence Tests
# ══════════════════════════════════════════════════════════════

class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        """Test that store persists and reloads correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            # Create and save
            store1 = FederatedStore("persist-test", path)
            mem = FederatedMemory("persistent memory", "fact",
                                  namespaces=["public"],
                                  vclock=VectorClock().increment("persist-test"))
            store1.add_memory(mem)
            store1.share("public")
            store1.save()

            # Reload
            store2 = FederatedStore("persist-test", path)
            self.assertEqual(len(store2.memories), 1)
            self.assertIn(mem.hash, store2.memories)
            self.assertEqual(store2.memories[mem.hash].content, "persistent memory")
            self.assertIn("public", store2.shared_namespaces)

            # Merkle tree should be rebuilt
            self.assertEqual(store1.memory_tree.root, store2.memory_tree.root)
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════
# Delta Efficiency Tests
# ══════════════════════════════════════════════════════════════

class TestDeltaEfficiency(unittest.TestCase):
    def test_large_store_small_delta(self):
        """With 1000 shared memories and 1 new one, only 1 should transfer."""
        store1 = FederatedStore("big-A")
        store2 = FederatedStore("big-B")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        # Both have 1000 identical memories
        for i in range(1000):
            mem = FederatedMemory(f"shared memory {i}", "fact")
            store1.add_memory(mem)
            store2.add_memory(FederatedMemory.from_dict(mem.to_dict()))

        # Verify in sync
        self.assertEqual(store1.memory_tree.root, store2.memory_tree.root)

        # Add 1 new memory to store1
        new_mem = FederatedMemory("brand new memory", "fact",
                                  vclock=VectorClock().increment("big-A"))
        store1.add_memory(new_mem)

        # Sync: store2 pulls from store1
        stats = engine2.sync_with(engine1, bidirectional=False)
        # Only 1 new memory should have been pulled
        self.assertEqual(stats["pulled"], 1)
        self.assertEqual(len(store2.memories), 1001)

    def test_empty_to_full_sync(self):
        """Sync from empty to full store."""
        store1 = FederatedStore("full")
        store2 = FederatedStore("empty")
        engine1 = SyncEngine(store1)
        engine2 = SyncEngine(store2)

        for i in range(50):
            mem = FederatedMemory(f"memory {i}", "fact",
                                  vclock=VectorClock().increment("full"))
            store1.add_memory(mem)

        stats = engine2.sync_with(engine1)
        self.assertEqual(stats["pulled"], 50)
        self.assertEqual(len(store2.memories), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
