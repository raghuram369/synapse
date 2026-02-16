import unittest

from triples import ENTITY_ALIAS_MAP, Triple, TripleIndex, extract_triples, normalize_entity


class TestTripleExtraction(unittest.TestCase):

    def test_extract_is_statement(self):
        triples = extract_triples("Alice is a developer.")
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple.subject, "alice")
        self.assertEqual(triple.predicate, "is")
        self.assertEqual(triple.object, "developer")
        self.assertEqual(triple.polarity, "positive")
        self.assertEqual(triple.tense, "present")
        self.assertEqual(triple.confidence, 1.0)

    def test_extract_is_negative(self):
        triples = extract_triples("Alice is not a developer.")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].polarity, "negative")

    def test_extract_is_negative_contraction(self):
        triples = extract_triples("Alice isn't a developer.")
        self.assertEqual(triples[0].polarity, "negative")

    def test_extract_hedged_confidence(self):
        triples = extract_triples("Maybe Alice likes jazz.")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].subject, "alice")
        self.assertEqual(triples[0].predicate, "likes")
        self.assertEqual(triples[0].confidence, 0.5)

    def test_extract_likes_statement(self):
        triples = extract_triples("Alice likes jazz.")
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple.subject, "alice")
        self.assertEqual(triple.predicate, "likes")
        self.assertEqual(triple.object, "jazz")
        self.assertEqual(triple.tense, "present")
        self.assertEqual(triple.confidence, 1.0)

    def test_extract_prefers_statement(self):
        triples = extract_triples("Alice prefers tea.")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].predicate, "prefers")
        self.assertEqual(triples[0].object, "tea")

    def test_extract_wants_negated(self):
        triples = extract_triples("Alice no longer wants dessert.")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].polarity, "negative")
        self.assertEqual(triples[0].predicate, "wants")
        self.assertEqual(triples[0].object, "dessert")

    def test_extract_works_at(self):
        triples = extract_triples("Alice works at Acme.")
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple.predicate, "works_at")
        self.assertEqual(triple.object, "acme")

    def test_extract_moved_to(self):
        triples = extract_triples("Alice moved to New York.")
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple.predicate, "moved_to")
        self.assertEqual(triple.object, "new york")
        self.assertEqual(triple.tense, "past")

    def test_extract_changed_from_to(self):
        triples = extract_triples("Alice changed from intern to manager.")
        self.assertEqual(len(triples), 2)
        objects = {(triple.object, triple.predicate, triple.polarity) for triple in triples}
        self.assertIn(("intern", "changed_from", "negative"), objects)
        self.assertIn(("manager", "changed_to", "positive"), objects)
        self.assertTrue(all(triple.confidence == 0.7 for triple in triples))

    def test_extract_possessive_pattern(self):
        triples = extract_triples("Alice's laptop is red.")
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple.subject, "alice")
        self.assertEqual(triple.predicate, "laptop")
        self.assertEqual(triple.object, "red")
        self.assertEqual(triple.polarity, "positive")

    def test_tense_detection_past(self):
        triples = extract_triples("Alice was a manager.")
        self.assertEqual(triples[0].tense, "past")

    def test_tense_detection_future(self):
        triples = extract_triples("Alice will work at Acme.")
        self.assertEqual(triples[0].tense, "future")

    def test_source_span_presence(self):
        text = "Alice is a developer."
        triples = extract_triples(text)
        triple = triples[0]
        self.assertEqual(triple.source_span[0], 0)
        self.assertGreaterEqual(triple.source_span[1], len(text) - 1)

    def test_normalize_entity_articles(self):
        self.assertEqual(normalize_entity("  The Dogs  "), "dog")

    def test_normalize_entity_alias(self):
        self.assertEqual(normalize_entity("NYC"), "new york")
        self.assertIn("nyc", ENTITY_ALIAS_MAP)

    def test_normalize_entity_extension(self):
        original = dict(ENTITY_ALIAS_MAP)
        try:
            ENTITY_ALIAS_MAP["boss"] = "manager"
            self.assertEqual(normalize_entity("the boss"), "manager")
        finally:
            ENTITY_ALIAS_MAP.clear()
            ENTITY_ALIAS_MAP.update(original)


class TestTripleIndex(unittest.TestCase):

    def test_query_subject_predicate_object(self):
        index = TripleIndex()
        index.add(1, extract_triples("Alice works at Acme."))
        index.add(2, extract_triples("Bob likes cats."))
        self.assertEqual(index.query_subject("alice"), {0})
        self.assertEqual(index.query_predicate("works_at"), {0})
        self.assertEqual(index.query_object("cats"), {1})
        self.assertEqual(index.query_spo("alice", "works_at", "acme"), {0})
        self.assertEqual(index.query_spo("bob", "likes", "cats"), {1})

    def test_query_spo_with_none(self):
        index = TripleIndex()
        index.add(1, extract_triples("Alice changed from intern to manager."))
        self.assertEqual(index.query_spo(None, "changed_to", "manager"), {1})
        self.assertEqual(index.query_spo("alice", None, None), {0, 1})

    def test_get_triples_for_memory(self):
        index = TripleIndex()
        triples = extract_triples("Alice changed from intern to manager.")
        index.add(42, triples)
        self.assertEqual(len(index.get_triples_for_memory(42)), 2)
        self.assertTrue(all(isinstance(item, Triple) for item in index.get_triples_for_memory(42)))

    def test_remove_memory(self):
        index = TripleIndex()
        index.add(10, extract_triples("Alice likes jazz."))
        index.add(11, extract_triples("Bob likes tea."))
        index.remove_memory(10)
        self.assertEqual(index.get_triples_for_memory(10), [])
        self.assertEqual(index.query_subject("alice"), set())
        self.assertEqual(index.query_object("tea"), {1})


if __name__ == "__main__":
    unittest.main()
