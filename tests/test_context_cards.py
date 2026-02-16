#!/usr/bin/env python3

import argparse
import io
import os
import re
import tempfile
from contextlib import redirect_stdout
import unittest

import cli
from context_cards import CardDeck, ContextCard
from synapse import Synapse


class TestContextCards(unittest.TestCase):
    def setUp(self):
        self.s = Synapse(":memory:")

    def tearDown(self):
        self.s.close()

    def _remember(self, content: str):
        return self.s.remember(content, deduplicate=False)

    def test_create_card_from_pack(self):
        self._remember("Alice likes Python.")
        self._remember("Bob prefers coffee.")
        pack = self.s.compile_context("Alice")
        card = ContextCard(pack)

        self.assertEqual(card.query, "Alice")
        self.assertTrue(card.card_id.startswith("card-"))
        self.assertTrue(re.match(r"^card-[0-9a-f]{12}$", card.card_id))
        self.assertEqual(card.summary, pack.to_compact())
        self.assertIsInstance(card.memories, list)
        self.assertIsInstance(card.graph_slice, dict)
        self.assertIsInstance(card.evidence, list)
        self.assertEqual(card.contradictions, [])

    def test_to_markdown_renders_sections(self):
        m1 = self._remember("Feature tests passed.")
        m2 = self._remember("Python improves test speed.")
        self.s.link(m1.id, m2.id, "supports")
        pack = self.s.compile_context("feature support")
        card = ContextCard(pack)

        out = card.to_markdown()

        self.assertIn("## 🧠 Memory Card:", out)
        self.assertIn("**Query:** feature support", out)
        self.assertIn("### Summary", out)
        self.assertIn("### Evidence", out)
        self.assertIn("### Graph Context", out)
        self.assertIn("### Contradictions", out)

    def test_json_roundtrip(self):
        self._remember("The server is healthy.")
        pack = self.s.compile_context("healthy")
        card = ContextCard(pack)

        payload = card.to_json()
        restored = ContextCard.from_json(payload)

        self.assertEqual(restored.query, card.query)
        self.assertEqual(restored.card_id, card.card_id)
        self.assertEqual(restored.summary, card.summary)
        self.assertEqual(restored.created_at, card.created_at)
        self.assertEqual(restored.memories, card.memories)

    def test_binary_roundtrip(self):
        self._remember("Testing TLV serialization.")
        pack = self.s.compile_context("TLV")
        card = ContextCard(pack)

        decoded = ContextCard.from_bytes(card.to_bytes())

        self.assertEqual(decoded.card_id, card.card_id)
        self.assertEqual(decoded.query, card.query)
        self.assertEqual(decoded.summary, card.summary)

    def test_invalid_tlv_raises(self):
        self.assertRaises(ValueError, ContextCard.from_bytes, b"junk")

    def test_replay_produces_diff_metadata(self):
        self._remember("Task status is open")
        card = ContextCard(self.s.compile_context("task"))

        other = Synapse(":memory:")
        try:
            other.remember("Task status is closed")
            replayed = card.replay(other)
            diff = replayed.metadata["replay_diff"]

            self.assertIn("new_facts", diff)
            self.assertIn("removed_facts", diff)
            self.assertIn("card_id", diff)
            self.assertIn("replayed_from", replayed.metadata)
        finally:
            other.close()

    def test_diff_between_cards(self):
        first_a = self._remember("Alice likes tea.")
        first_b = self._remember("Alice likes coffee.")
        self.s.link(first_a.id, first_b.id, "supports")

        card_a = ContextCard(self.s.compile_context("Alice"))

        # Slightly different second pack to create changed/new facts.
        self._remember("Bob likes tea.")
        card_b = ContextCard(self.s.compile_context("Alice"))

        diff = card_a.diff(card_b)

        self.assertIn("new_facts", diff)
        self.assertIn("removed_facts", diff)
        self.assertIn("changed_scores", diff)

    def test_deck_add_get_search(self):
        card_pack = ContextCard(self.s.compile_context("search terms"))

        deck = CardDeck()
        deck.add(card_pack)
        self.assertIs(deck.get(card_pack.card_id), card_pack)

        results = deck.search("search")
        self.assertEqual(len(results), 1)

    def test_deck_export_import_roundtrip(self):
        card_a = ContextCard(self.s.compile_context("export"))
        if card_a.memories:
            self.s.link(card_a.memories[0].get("id", 0), card_a.memories[0].get("id", 0), "related")
        card_b = ContextCard(self.s.compile_context("import"))

        deck = CardDeck()
        deck.add(card_a)
        deck.add(card_b)

        with tempfile.TemporaryDirectory(prefix="synapse-card-test-") as temp_dir:
            path = os.path.join(temp_dir, "cards.synapse-cards")
            deck.export(path)

            imported = CardDeck()
            count = imported.import_deck(path)

            self.assertEqual(count, 2)
            self.assertIn(card_a.card_id, imported.cards)
            self.assertIn(card_b.card_id, imported.cards)

    def test_synapse_create_card_is_persistent(self):
        with tempfile.TemporaryDirectory(prefix="synapse-card-persist-") as temp_dir:
            db = os.path.join(temp_dir, "synapse")
            creator = Synapse(db)
            creator.remember("Persistence memory", deduplicate=False)
            card = creator.create_card("persistence")
            creator.close()

            reader = Synapse(db)
            try:
                loaded = reader.cards.get(card.card_id)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.query, "persistence")
            finally:
                reader.close()


class TestContextCardCli(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-card-cli-")
        self.db = os.path.join(self.tempdir.name, "synapse")

        syn = Synapse(self.db)
        syn.remember("Alice likes cherries.")
        syn.close()

    def tearDown(self):
        self.tempdir.cleanup()

    def _capture(self, fn, **kwargs):
        buf = io.StringIO()
        namespace = argparse.Namespace(
            db=self.db,
            query=None,
            card_id=None,
            path=None,
            budget=2000,
            card_action=None,
        )
        namespace.__dict__.update(kwargs)
        with redirect_stdout(buf):
            fn(namespace)
        return buf.getvalue()

    def test_cli_create_show_export(self):
        created_output = self._capture(cli.cmd_card_create, query="Alice")
        self.assertIn("## 🧠 Memory Card:", created_output)

        viewer = Synapse(self.db)
        try:
            card_id = next(iter(viewer.cards.cards))
        finally:
            viewer.close()

        shown_output = self._capture(cli.cmd_card_show, card_id=card_id)
        self.assertIn(card_id, shown_output)

        with tempfile.TemporaryDirectory(prefix="synapse-card-export-") as temp_dir:
            out_path = os.path.join(temp_dir, "cards.synapse-cards")
            export_output = self._capture(cli.cmd_card_export, path=out_path)
            self.assertIn("Exported cards to", export_output)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
