#!/usr/bin/env python3
"""Tests for the fact extraction module."""

import unittest
from unittest.mock import patch, Mock, MagicMock
import json
import urllib.error

from extractor import FactExtractor, extract_facts


class TestFactExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = FactExtractor(model="test-model")

    def test_init(self):
        """Test extractor initialization."""
        extractor = FactExtractor(
            base_url="http://test:1234",
            model="test-model",
            timeout=60.0
        )
        
        self.assertEqual(extractor.base_url, "http://test:1234")
        self.assertEqual(extractor.model, "test-model")
        self.assertEqual(extractor.timeout, 60.0)
        self.assertEqual(extractor._generation_url, "http://test:1234/api/generate")

    def test_build_extraction_prompt(self):
        """Test prompt building."""
        content = "Test content here"
        prompt = self.extractor._build_extraction_prompt(content)
        
        self.assertIn("Extract the key facts", prompt)
        self.assertIn(content, prompt)
        self.assertIn("one per line", prompt)

    def test_parse_response_basic(self):
        """Test parsing a basic LLM response."""
        response = """Caroline is researching adoption agencies
Adoption has been on Caroline's mind"""
        
        facts = self.extractor._parse_response(response)
        
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0], "Caroline is researching adoption agencies")
        self.assertEqual(facts[1], "Adoption has been on Caroline's mind")

    def test_parse_response_with_bullets(self):
        """Test parsing response with bullet points."""
        response = """• Caroline is researching adoption agencies
- Adoption has been on Caroline's mind
* This is another fact"""
        
        facts = self.extractor._parse_response(response)
        
        self.assertEqual(len(facts), 3)
        self.assertEqual(facts[0], "Caroline is researching adoption agencies")
        self.assertEqual(facts[1], "Adoption has been on Caroline's mind")
        self.assertEqual(facts[2], "This is another fact")

    def test_parse_response_filters_short_facts(self):
        """Test that short/meaningless facts are filtered out."""
        response = """This is a good fact that should be kept
Short
Caroline is researching adoption agencies
Yes"""
        
        facts = self.extractor._parse_response(response)
        
        self.assertEqual(len(facts), 2)
        self.assertIn("This is a good fact that should be kept", facts)
        self.assertIn("Caroline is researching adoption agencies", facts)

    def test_parse_response_filters_instructions(self):
        """Test that instruction lines are filtered out."""
        response = """Here are the facts:
Caroline is researching adoption agencies
Key facts:
Adoption has been on Caroline's mind"""
        
        facts = self.extractor._parse_response(response)
        
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0], "Caroline is researching adoption agencies")
        self.assertEqual(facts[1], "Adoption has been on Caroline's mind")

    def test_extract_facts_empty_content(self):
        """Test extraction with empty content."""
        facts = self.extractor.extract_facts("")
        self.assertEqual(facts, [])
        
        facts = self.extractor.extract_facts("   ")
        self.assertEqual(facts, [])

    @patch("extractor.urllib.request.urlopen")
    def test_extract_facts_success(self, mock_urlopen):
        """Test successful fact extraction."""
        # Mock successful Ollama response
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = json.dumps(
            {
                "response": "Caroline is researching adoption agencies\nAdoption has been on Caroline's mind"
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response
        
        facts = self.extractor.extract_facts("Caroline: Yeah I've been looking into adoption agencies recently, it's been on my mind a lot")
        
        self.assertEqual(len(facts), 2)
        self.assertIn("Caroline is researching adoption agencies", facts)
        self.assertIn("Adoption has been on Caroline's mind", facts)
        
        # Verify API call
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")

    @patch("extractor.urllib.request.urlopen")
    def test_extract_facts_api_error(self, mock_urlopen):
        """Test extraction with API error."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")
        
        facts = self.extractor.extract_facts("Test content")
        
        # Should return empty list on error
        self.assertEqual(facts, [])

    @patch("extractor.urllib.request.urlopen")
    def test_extract_facts_invalid_json(self, mock_urlopen):
        """Test extraction with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = b"{not-json"
        mock_urlopen.return_value = mock_response
        
        facts = self.extractor.extract_facts("Test content")
        
        # Should return empty list on JSON error
        self.assertEqual(facts, [])

    @patch("extractor.urllib.request.urlopen")
    def test_is_available_success(self, mock_urlopen):
        """Test availability check when Ollama is available."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "test-model:latest"}, {"name": "other-model:latest"}]}
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response
        
        available = self.extractor.is_available()
        self.assertTrue(available)

    @patch("extractor.urllib.request.urlopen")
    def test_is_available_model_not_found(self, mock_urlopen):
        """Test availability check when model is not found."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "other-model:latest"}]}
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response
        
        available = self.extractor.is_available()
        self.assertFalse(available)

    @patch("extractor.urllib.request.urlopen")
    def test_is_available_connection_error(self, mock_urlopen):
        """Test availability check with connection error."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")
        
        available = self.extractor.is_available()
        self.assertFalse(available)

    @patch('extractor.get_extractor')
    def test_extract_facts_function(self, mock_get_extractor):
        """Test the convenience extract_facts function."""
        mock_extractor = Mock()
        mock_extractor.extract_facts.return_value = ["fact1", "fact2"]
        mock_get_extractor.return_value = mock_extractor
        
        facts = extract_facts("test content", model="test-model")
        
        self.assertEqual(facts, ["fact1", "fact2"])
        mock_get_extractor.assert_called_with("test-model")
        mock_extractor.extract_facts.assert_called_with("test content")


if __name__ == "__main__":
    unittest.main()
