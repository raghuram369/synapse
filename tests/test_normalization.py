import unittest

from normalization import EntityNormalizer
from synapse import Synapse
from triples import normalize_entity


class TestEntityNormalizerLemmatization(unittest.TestCase):

    def setUp(self):
        self.normalizer = EntityNormalizer()

    def test_noun_plural_cats(self):
        self.assertEqual(self.normalizer.lemmatize("cats"), "cat")

    def test_noun_plural_cities(self):
        self.assertEqual(self.normalizer.lemmatize("cities"), "city")

    def test_noun_irregular_children(self):
        self.assertEqual(self.normalizer.lemmatize("children"), "child")

    def test_verb_progressive_running(self):
        self.assertEqual(self.normalizer.lemmatize("running"), "run")

    def test_verb_past_liked(self):
        self.assertEqual(self.normalizer.lemmatize("liked"), "like")

    def test_verb_irregular_goes(self):
        self.assertEqual(self.normalizer.lemmatize("goes"), "go")

    def test_proper_nouns_keep_case(self):
        self.assertEqual(self.normalizer.lemmatize("Alice"), "Alice")


class TestEntityNormalizerCanonical(unittest.TestCase):

    def setUp(self):
        self.normalizer = EntityNormalizer()

    def test_canonical_removes_articles(self):
        self.assertEqual(self.normalizer.canonical("  The Dogs  "), "dog")

    def test_canonical_alias_nyc(self):
        self.assertEqual(self.normalizer.canonical("NYC"), "new york")

    def test_canonical_alias_with_article(self):
        self.assertEqual(self.normalizer.canonical("the ny"), "new york")

    def test_canonical_builtin_usa_and_uk(self):
        self.assertEqual(self.normalizer.canonical("usa"), "united states")
        self.assertEqual(self.normalizer.canonical("UK"), "united kingdom")

    def test_canonical_unknown_keeps_content(self):
        self.assertEqual(self.normalizer.canonical("modeling"), "modeling")

    def test_canonical_multi_word_lemma(self):
        self.assertEqual(self.normalizer.canonical("Running Models"), "run model")

    def test_register_alias_extension(self):
        self.normalizer.register_alias("the boss", "manager")
        self.assertEqual(self.normalizer.canonical("the boss"), "manager")

    def test_coref_pronouns_to_recent_entities(self):
        texts = [
            "Alice joined Synapse yesterday.",
            "She reviewed the architecture.",
        ]
        resolved = self.normalizer.coref_resolve(texts)
        self.assertEqual(resolved["she"], "Alice")

    def test_coref_it_to_latest_non_person(self):
        texts = [
            "Alice visited New York.",
            "She said it was raining.",
        ]
        resolved = self.normalizer.coref_resolve(texts)
        self.assertEqual(resolved["it"], "new york")

    def test_coref_customer_alias_without_prior_entities(self):
        self.normalizer.register_alias("the user", "Riley")
        texts = ["The customer reached out this morning."]
        resolved = self.normalizer.coref_resolve(texts)
        self.assertEqual(resolved["the customer"], "Riley")


class TestSynapseAliasRegistration(unittest.TestCase):

    def test_register_alias_from_synapse(self):
        s = Synapse(":memory:")
        try:
            s.register_alias("py", "python")
            self.assertEqual(normalize_entity("py"), "python")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
