"""Tests for card_share module."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from card_share import share_card, _apply_redaction, _render_markdown, _render_html


class TestRedaction(unittest.TestCase):
    def test_pii_email(self):
        result = _apply_redaction("Contact john@example.com please", "pii")
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("john@example.com", result)

    def test_pii_phone(self):
        result = _apply_redaction("Call 555-123-4567", "pii")
        self.assertIn("[PHONE]", result)

    def test_pii_ssn(self):
        result = _apply_redaction("SSN: 123-45-6789", "pii")
        self.assertIn("[SSN]", result)

    def test_names_redaction(self):
        result = _apply_redaction("Met John Smith today", "names")
        self.assertIn("[NAME]", result)

    def test_numbers_redaction(self):
        result = _apply_redaction("Price is 42.99 dollars", "numbers")
        self.assertIn("[NUM]", result)

    def test_no_redaction(self):
        text = "Hello world"
        self.assertEqual(_apply_redaction(text, None), text)


class TestRenderMarkdown(unittest.TestCase):
    def test_basic_render(self):
        pack = MagicMock()
        pack.summary = "Test summary"
        pack.memories = [{"content": "fact 1"}, {"content": "fact 2"}]
        pack.provenance = ["source A"]

        result = _render_markdown("test query", pack, {}, None)
        self.assertIn("Memory Card: test query", result)
        self.assertIn("Test summary", result)
        self.assertIn("fact 1", result)
        self.assertIn("source A", result)

    def test_empty_memories(self):
        pack = MagicMock()
        pack.summary = "Empty"
        pack.memories = []
        pack.provenance = []

        result = _render_markdown("q", pack, {}, None)
        self.assertIn("(no memories)", result)


class TestRenderHTML(unittest.TestCase):
    def test_html_output(self):
        pack = MagicMock()
        pack.summary = "HTML test"
        pack.memories = [{"content": "mem"}]
        pack.provenance = []

        result = _render_html("test", pack, {}, None)
        self.assertIn("<!DOCTYPE html>", result)
        self.assertIn("Memory Card: test", result)
        self.assertIn("HTML test", result)


class TestShareCard(unittest.TestCase):
    def test_share_card_markdown(self):
        syn = MagicMock()
        pack = MagicMock()
        pack.summary = "compiled"
        pack.memories = [{"content": "m1"}]
        pack.provenance = []
        syn.compile_context.return_value = pack
        syn.beliefs.return_value = {}

        result = share_card(syn, "hello", format="markdown")
        self.assertIn("Memory Card: hello", result)

    def test_share_card_html(self):
        syn = MagicMock()
        pack = MagicMock()
        pack.summary = "ctx"
        pack.memories = []
        pack.provenance = []
        syn.compile_context.return_value = pack
        syn.beliefs.return_value = {}

        result = share_card(syn, "hello", format="html")
        self.assertIn("<!DOCTYPE html>", result)

    def test_share_card_with_redaction(self):
        syn = MagicMock()
        pack = MagicMock()
        pack.summary = "Email is test@example.com"
        pack.memories = [{"content": "Call 555-123-4567"}]
        pack.provenance = []
        syn.compile_context.return_value = pack
        syn.beliefs.return_value = {}

        result = share_card(syn, "contacts", redact="pii", format="markdown")
        self.assertIn("[EMAIL]", result)
        self.assertIn("[PHONE]", result)

    def test_share_card_with_beliefs(self):
        syn = MagicMock()
        pack = MagicMock()
        pack.summary = "about food"
        pack.memories = []
        pack.provenance = []
        syn.compile_context.return_value = pack

        belief = MagicMock()
        belief.value = "pizza"
        syn.beliefs.return_value = {"favorite_food": belief}

        result = share_card(syn, "food", format="markdown")
        self.assertIn("favorite_food", result)
        self.assertIn("pizza", result)


if __name__ == "__main__":
    unittest.main()
