import unittest

from evidence import EvidenceChain, EvidenceCompiler
from triples import Triple
from synapse import Synapse


class _DummyMemory:
    def __init__(self, memory_id, content, created_at=0.0):
        self.id = memory_id
        self.content = content
        self.created_at = created_at


class _FakeTripleIndex:
    def __init__(self, triples_by_memory):
        self.triples_by_memory = triples_by_memory

    def get_triples_for_memory(self, memory_id):
        return self.triples_by_memory.get(memory_id, [])


class TestEvidenceCompiler(unittest.TestCase):
    def test_empty_memories_returns_empty(self):
        compiler = EvidenceCompiler()
        self.assertEqual(compiler.compile([], None), [])

    def test_supporting_memories_are_recorded(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Alice likes tea", created_at=10.0),
            _DummyMemory(2, "Alice moved to paris", created_at=20.0),
        ]

        chains = compiler.compile(memories, None)

        self.assertTrue(any(chain.claim == "alice likes tea" for chain in chains))
        like_chain = next(chain for chain in chains if chain.claim == "alice likes tea")
        self.assertEqual(like_chain.supporting_memories, [1])
        self.assertFalse(like_chain.contradicting_memories)
        self.assertEqual(like_chain.first_seen, 10.0)
        self.assertEqual(like_chain.last_confirmed, 10.0)

    def test_detects_polarity_conflict(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Alice likes tea", created_at=1.0),
            _DummyMemory(2, "Alice does not like tea", created_at=2.0),
        ]

        chains = compiler.compile(memories, None)
        like = next(chain for chain in chains if chain.claim == "alice likes tea")
        no_like = next(chain for chain in chains if chain.claim == "alice not likes tea")

        self.assertIn(2, like.contradicting_memories)
        self.assertIn(1, no_like.contradicting_memories)

    def test_detects_mutual_exclusion_conflict(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Alice is vegetarian", created_at=1.0),
            _DummyMemory(2, "Alice is vegan", created_at=2.0),
        ]

        chains = compiler.compile(memories, None)
        vegan = next(chain for chain in chains if chain.claim == "alice is vegan")
        vegetarian = next(chain for chain in chains if chain.claim == "alice is vegetarian")

        self.assertIn(2, vegetarian.contradicting_memories)
        self.assertIn(1, vegan.contradicting_memories)

    def test_conflict_penalty_reduces_confidence(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Alice likes tea", created_at=1.0),
            _DummyMemory(2, "Alice does not like tea", created_at=2.0),
        ]

        chains = compiler.compile(memories, None)
        like_chain = next(chain for chain in chains if chain.claim == "alice likes tea")
        self.assertLess(like_chain.confidence, 1.0)
        self.assertGreaterEqual(like_chain.confidence, 0.5)

    def test_first_seen_last_confirmed_from_multiple_memories(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Alice likes tea", created_at=50.0),
            _DummyMemory(2, "Alice likes tea", created_at=60.0),
            _DummyMemory(3, "Alice likes tea", created_at=70.0),
        ]

        chains = compiler.compile(memories, None)
        chain = next(chain for chain in chains if chain.claim == "alice likes tea")
        self.assertEqual(chain.first_seen, 50.0)
        self.assertEqual(chain.last_confirmed, 70.0)

    def test_uses_triples_from_index(self):
        fake_triples = {
            9: [
                Triple(
                    subject="alice",
                    predicate="likes",
                    object="coffee",
                    polarity="positive",
                    tense="present",
                    confidence=0.9,
                    source_span=(0, 10),
                )
            ]
        }
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(9, "ignored content", created_at=1.0),
        ]

        chains = compiler.compile(memories, _FakeTripleIndex(fake_triples))
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].claim, "alice likes coffee")
        self.assertEqual(chains[0].supporting_memories, [9])
        self.assertEqual(chains[0].confidence, 0.9)

    def test_compile_orders_by_confidence(self):
        compiler = EvidenceCompiler()
        memories = [
            _DummyMemory(1, "Maybe Alice likes tea", created_at=1.0),
            _DummyMemory(2, "Alice likes music", created_at=2.0),
        ]

        chains = compiler.compile(memories, None)
        self.assertGreater(chains[0].confidence, chains[1].confidence)

    def test_evidence_chain_dataclass_fields(self):
        chain = EvidenceChain(
            claim="alice likes tea",
            supporting_memories=[1],
            contradicting_memories=[],
            confidence=0.85,
            first_seen=10.0,
            last_confirmed=20.0,
        )

        self.assertEqual(chain.claim, "alice likes tea")
        self.assertEqual(chain.supporting_memories, [1])
        self.assertEqual(chain.contradicting_memories, [])
        self.assertEqual(chain.confidence, 0.85)


class TestEvidenceIntegration(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def test_context_pack_exposes_claim_evidence(self):
        self.s.remember("Alice likes tea", deduplicate=False)
        self.s.remember("Alice is not vegan", deduplicate=False)
        self.s.remember("Alice is vegan", deduplicate=False)

        pack = self.s.compile_context("alice", budget=4000)
        claim_items = [item for item in pack.evidence if isinstance(item, dict) and "claim" in item]
        self.assertTrue(claim_items)
        self.assertIn("alice likes tea", {item["claim"] for item in claim_items})

    def test_recall_includes_evidence_pointers_when_explain(self):
        first = self.s.remember("Alice likes tea", deduplicate=False)
        second = self.s.remember("Alice does not like tea", deduplicate=False)

        results = self.s.recall("Alice", explain=True, limit=5)
        self.assertEqual(len(results), 2)
        ids_to_memory = {memory.id: memory for memory in results}

        for memory_id in (first.id, second.id):
            memory = ids_to_memory[memory_id]
            self.assertTrue(isinstance(memory.evidence_pointers, list))
            self.assertTrue(any(item.get("claim") == "alice likes tea" for item in memory.evidence_pointers))
