"""Tests for the LLM enrichment module."""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from dataclasses import dataclass

from enrich import (
    OllamaClient, MemoryEnricher, EnrichmentResult,
    extract_canonical_triples, canonicalize_text, check_for_duplicates,
    suggest_promotion, enrich_memory_sync, batch_enrich_memories
)


@dataclass
class MockMemory:
    """Mock memory object for testing."""
    id: str
    content: str
    created_at: float
    metadata: dict
    memory_type: str = "fact"

    def __post_init__(self):
        if 'router_category' not in self.metadata:
            self.metadata['router_category'] = self.memory_type


class TestOllamaClient:
    """Test Ollama client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OllamaClient()
        
    @patch('urllib.request.urlopen')
    def test_is_available_success(self, mock_urlopen):
        """Test successful Ollama availability check."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "models": [{"name": "qwen2.5:1.5b"}]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        assert self.client.is_available() is True
        
    @patch('urllib.request.urlopen')
    def test_is_available_no_model(self, mock_urlopen):
        """Test availability check when model is not available."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "models": [{"name": "some-other-model"}]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        assert self.client.is_available() is False
        
    @patch('urllib.request.urlopen')
    def test_is_available_connection_error(self, mock_urlopen):
        """Test availability check when connection fails."""
        mock_urlopen.side_effect = Exception("Connection refused")
        
        assert self.client.is_available() is False
        
    @patch('urllib.request.urlopen')
    def test_generate_success(self, mock_urlopen):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "response": "Generated text response"
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock availability check
        self.client._available = True
        
        result = self.client.generate("Test prompt")
        assert result == "Generated text response"
        
    def test_generate_not_available(self):
        """Test generation when Ollama is not available."""
        self.client._available = False
        
        result = self.client.generate("Test prompt")
        assert result is None
        
    @patch('urllib.request.urlopen')
    def test_generate_with_system_prompt(self, mock_urlopen):
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "response": "System-guided response"
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        self.client._available = True
        
        result = self.client.generate("User prompt", system="System prompt")
        assert result == "System-guided response"
        
        # Check that system prompt was included in request
        call_args = mock_urlopen.call_args
        request_data = json.loads(call_args[0][0].data.decode())
        assert request_data["system"] == "System prompt"


class TestEnrichmentFunctions:
    """Test individual enrichment functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_ollama = Mock()
        self.mock_ollama.is_available.return_value = True
        
    def test_extract_canonical_triples_success(self):
        """Test successful triple extraction."""
        self.mock_ollama.generate.return_value = "Alice | works_at | Company\nBob | manages | Team"
        
        result = extract_canonical_triples(self.mock_ollama, "Alice works at Company and Bob manages the team")
        
        assert result == ["Alice | works_at | Company", "Bob | manages | Team"]
        assert self.mock_ollama.generate.called
        
    def test_extract_canonical_triples_malformed(self):
        """Test triple extraction with malformed response."""
        self.mock_ollama.generate.return_value = "Not a proper triple\nAlice works somewhere"
        
        result = extract_canonical_triples(self.mock_ollama, "Some text")
        
        assert result == []  # No valid triples
        
    def test_extract_canonical_triples_no_response(self):
        """Test triple extraction when LLM doesn't respond."""
        self.mock_ollama.generate.return_value = None
        
        result = extract_canonical_triples(self.mock_ollama, "Some text")
        
        assert result is None
        
    def test_canonicalize_text_success(self):
        """Test successful text canonicalization."""
        self.mock_ollama.generate.return_value = "Alice is a software engineer at TechCorp."
        
        result = canonicalize_text(self.mock_ollama, "alice is an eng at techcorp")
        
        assert result == "Alice is a software engineer at TechCorp."
        
    def test_canonicalize_text_too_different(self):
        """Test canonicalization rejection when result is too different."""
        # Very short response compared to input
        self.mock_ollama.generate.return_value = "Short"
        
        result = canonicalize_text(self.mock_ollama, "This is a very long input text that should not be shortened to just one word")
        
        assert result is None  # Quality check failed
        
    def test_check_for_duplicates_found(self):
        """Test duplicate detection when duplicate exists."""
        memories = [
            MockMemory("1", "Alice is a developer", time.time(), {}),
            MockMemory("2", "Bob is a manager", time.time(), {}),
        ]
        
        self.mock_ollama.generate.return_value = "DUPLICATE:1"
        
        result = check_for_duplicates(self.mock_ollama, "Alice is a software developer", memories)
        
        assert result == "1"
        
    def test_check_for_duplicates_unique(self):
        """Test duplicate detection when content is unique."""
        memories = [
            MockMemory("1", "Alice is a developer", time.time(), {}),
        ]
        
        self.mock_ollama.generate.return_value = "UNIQUE"
        
        result = check_for_duplicates(self.mock_ollama, "Bob is a designer", memories)
        
        assert result is None
        
    def test_check_for_duplicates_no_memories(self):
        """Test duplicate detection with no existing memories."""
        result = check_for_duplicates(self.mock_ollama, "Some content", [])
        
        assert result is None
        assert not self.mock_ollama.generate.called
        
    def test_suggest_promotion_promote(self):
        """Test promotion suggestion when promotion is recommended."""
        self.mock_ollama.generate.return_value = "PROMOTE:pattern:This shows recurring behavior"
        
        result = suggest_promotion(self.mock_ollama, "I always use vim for editing", "preference")
        
        assert result == "pattern:This shows recurring behavior"
        
    def test_suggest_promotion_keep(self):
        """Test promotion suggestion when no promotion is needed."""
        self.mock_ollama.generate.return_value = "KEEP:Single instance, not a pattern"
        
        result = suggest_promotion(self.mock_ollama, "I used vim today", "fact")
        
        assert result is None


class TestMemoryEnricher:
    """Test the main memory enricher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_synapse = Mock()
        self.enricher = MemoryEnricher(self.mock_synapse)
        
    @patch('enrich.OllamaClient')
    def test_init_with_ollama_available(self, mock_ollama_class):
        """Test enricher initialization when Ollama is available."""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        mock_ollama_class.return_value = mock_ollama
        
        enricher = MemoryEnricher(self.mock_synapse)
        assert enricher.ollama is not None
        
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.enricher.processed_count = 5
        self.enricher.error_count = 1
        
        stats = self.enricher.get_stats()
        
        assert stats["processed_count"] == 5
        assert stats["error_count"] == 1
        assert "ollama_available" in stats
        assert "running" in stats
        
    @patch('asyncio.sleep')
    def test_start_background_enrichment_not_available(self, mock_sleep):
        """Test background enrichment when Ollama is not available."""
        self.enricher.ollama._available = False

        asyncio.run(self.enricher.start_background_enrichment(interval=0.1))

        assert not mock_sleep.called

    def test_enrich_pending_memories_no_pending(self):
        """Test enrichment when no memories need processing."""
        self.enricher.ollama._available = False

        result = asyncio.run(self.enricher.enrich_pending_memories())

        assert result == []

    def test_enrich_batch_success(self):
        """Test successful batch enrichment."""
        # Mock memories that need enrichment
        memory1 = MockMemory("1", "Alice is a developer", time.time(), {})
        memory2 = MockMemory("2", "Bob manages the team", time.time(), {})

        self.enricher.ollama._available = True
        self.mock_synapse.list.return_value = []
        
        # Mock enrichment functions
        with patch('enrich.canonicalize_text') as mock_canonicalize, \
             patch('enrich.extract_canonical_triples') as mock_triples, \
             patch('enrich.check_for_duplicates') as mock_duplicates, \
             patch('enrich.suggest_promotion') as mock_promotion:
            
            mock_canonicalize.return_value = "Alice is a software developer."
            mock_triples.return_value = ["Alice | is_a | developer"]
            mock_duplicates.return_value = None
            mock_promotion.return_value = None
            
            results = asyncio.run(self.enricher._enrich_batch([memory1, memory2]))
            
            assert len(results) == 2
            assert all(r.confidence > 0 for r in results)
            assert all(r.error is None for r in results)
            
    def test_enrich_batch_with_errors(self):
        """Test batch enrichment with some failures."""
        memory = MockMemory("1", "Test content", time.time(), {})

        self.enricher.ollama._available = True
        self.mock_synapse.list.return_value = []

        # Mock an exception during enrichment
        with patch('enrich.canonicalize_text', side_effect=Exception("LLM error")):
            results = asyncio.run(self.enricher._enrich_batch([memory]))
            
            assert len(results) == 1
            assert results[0].error == "LLM error"
            assert results[0].confidence == 0.0


class TestSyncInterface:
    """Test synchronous enrichment interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_synapse = Mock()
        self.mock_memory = MockMemory("1", "Alice is a developer", time.time(), {"router_category": "fact"})
        
    @patch('enrich.OllamaClient')
    def test_enrich_memory_sync_success(self, mock_ollama_class):
        """Test synchronous memory enrichment."""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        mock_ollama_class.return_value = mock_ollama
        
        self.mock_synapse.list.return_value = []
        
        with patch('enrich.canonicalize_text', return_value="Alice is a software developer."), \
             patch('enrich.extract_canonical_triples', return_value=["Alice | is_a | developer"]), \
             patch('enrich.check_for_duplicates', return_value=None), \
             patch('enrich.suggest_promotion', return_value=None):
            
            result = enrich_memory_sync(self.mock_synapse, self.mock_memory)
            
            assert result.original_text == "Alice is a developer"
            assert result.canonical_form == "Alice is a software developer."
            assert result.extracted_triples == ["Alice | is_a | developer"]
            assert result.duplicate_of is None
            assert result.confidence > 0
            
    @patch('enrich.OllamaClient')
    def test_enrich_memory_sync_not_available(self, mock_ollama_class):
        """Test synchronous enrichment when Ollama is not available."""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = False
        mock_ollama_class.return_value = mock_ollama
        
        result = enrich_memory_sync(self.mock_synapse, self.mock_memory)
        
        assert result.error == "Ollama not available"
        assert result.confidence == 0.0
        
    @patch('enrich.enrich_memory_sync')
    def test_batch_enrich_memories(self, mock_enrich_sync):
        """Test batch enrichment of multiple memories."""
        self.mock_synapse.list.return_value = [
            MockMemory("1", "Content 1", time.time(), {}),
            MockMemory("2", "Content 2", time.time(), {}),
            MockMemory("3", "Content 3", time.time(), {}),
        ]

        mock_enrich_sync.side_effect = lambda s, m, *_: EnrichmentResult(
            original_text=m.content,
            confidence=0.8,
        )

        results = batch_enrich_memories(self.mock_synapse, ["1", "2", "3"])

        assert len(results) == 3
        assert all(r.confidence == 0.8 for r in results)
        assert mock_enrich_sync.call_count == 3

    @patch('enrich.enrich_memory_sync')
    def test_batch_enrich_missing_memory(self, mock_enrich_sync):
        """Test batch enrichment with missing memory."""
        self.mock_synapse.list.return_value = []

        results = batch_enrich_memories(self.mock_synapse, ["missing_id"])

        assert len(results) == 1
        assert "not found" in results[0].error
        assert results[0].confidence == 0.0


class TestIntegration:
    """Integration tests for the enrichment pipeline."""
    
    @patch('enrich.OllamaClient')
    def test_full_enrichment_pipeline(self, mock_ollama_class):
        """Test the complete enrichment pipeline."""
        # Set up mocks
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        mock_ollama.generate.side_effect = [
            None,  # check_for_duplicates (no duplicates)
            "Alice is a software engineer at TechCorp.",  # canonicalize_text
            "Alice | works_as | software_engineer\nAlice | works_at | TechCorp",  # extract_triples
            "PROMOTE:profile:Recurring professional information"  # suggest_promotion
        ]
        mock_ollama_class.return_value = mock_ollama
        
        mock_synapse = Mock()
        mock_synapse.list.return_value = [MockMemory("0", "existing baseline memory", time.time(), {})]
        
        memory = MockMemory("1", "alice is a sw eng at techcorp", time.time(), {"router_category": "fact"})
        
        result = enrich_memory_sync(mock_synapse, memory)
        
        # Verify enrichment results
        assert result.original_text == "alice is a sw eng at techcorp"
        assert result.canonical_form == "Alice is a software engineer at TechCorp."
        assert len(result.extracted_triples) == 2
        assert "Alice | works_as | software_engineer" in result.extracted_triples
        assert "Alice | works_at | TechCorp" in result.extracted_triples
        assert result.promotion_suggestion == "profile:Recurring professional information"
        assert result.duplicate_of is None
        assert result.confidence > 0
        assert result.error is None
        
    @patch('enrich.OllamaClient')
    def test_duplicate_detection_integration(self, mock_ollama_class):
        """Test enrichment with duplicate detection."""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        mock_ollama.generate.return_value = "DUPLICATE:existing_memory_id"
        mock_ollama_class.return_value = mock_ollama
        
        mock_synapse = Mock()
        existing_memory = MockMemory("existing", "Alice is a developer", time.time() - 3600, {})
        mock_synapse.list.return_value = [existing_memory]
        
        new_memory = MockMemory("new", "Alice is a software developer", time.time(), {})
        
        result = enrich_memory_sync(mock_synapse, new_memory)
        
        assert result.duplicate_of == "existing_memory_id"
        
    def test_quality_checks(self):
        """Test quality checks in enrichment functions."""
        mock_ollama = Mock()
        
        # Test canonicalization quality check
        mock_ollama.generate.return_value = "x"  # Too short
        original_text = "This is a long and detailed explanation of something important"
        
        result = canonicalize_text(mock_ollama, original_text)
        assert result is None  # Should be rejected for being too short
        
        # Test triple extraction quality check
        mock_ollama.generate.return_value = "Not | a | proper | format"
        
        result = extract_canonical_triples(mock_ollama, "Some text")
        assert result == []  # Should return empty list for malformed triples


if __name__ == "__main__":
    pytest.main([__file__])