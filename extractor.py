"""LLM-powered fact extraction using a local Ollama model.

Distils raw conversation turns into clean, searchable facts.  Each
extracted fact becomes a separate memory with richer keyword coverage.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts key facts from text using Ollama LLM."""
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model: str = "qwen2.5:14b", timeout: float = 30.0):
        """Initialize the fact extractor.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use for extraction  
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._generation_url = urljoin(self.base_url + '/', 'api/generate')
        
    def extract_facts(self, content: str) -> List[str]:
        """Extract key facts from the given content.
        
        Args:
            content: Raw text to extract facts from
            
        Returns:
            List of extracted facts as strings. Empty list if extraction fails.
        """
        if not content.strip():
            return []
            
        prompt = self._build_extraction_prompt(content)
        
        try:
            response = self._call_ollama(prompt)
            facts = self._parse_response(response)
            return [fact.strip() for fact in facts if fact.strip()]
        except Exception as e:
            logger.warning("Fact extraction failed: %s", e)
            return []
    
    def _build_extraction_prompt(self, content: str) -> str:
        """Build the extraction prompt."""
        return f"""Extract the key facts from this conversation turn as a list of concise statements.
Each fact should be self-contained and searchable.

Turn: "{content}"

Output only the facts, one per line. No bullets, no numbering."""
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
                "num_predict": 300   # Limit response length
            }
        }
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._generation_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Ollama: {e}") from e
    
    def _parse_response(self, response: str) -> List[str]:
        """Parse the LLM response into individual facts."""
        if not response:
            return []
            
        # Split by newlines and clean up
        facts = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove common prefixes/markers
            line = line.lstrip('•-*•')  # Remove bullet points
            line = line.strip()
            
            # Skip very short facts (likely noise)
            if len(line) < 10:
                continue
                
            # Skip lines that look like instructions or meta-text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'here are the facts',
                'key facts:',
                'extracted facts:',
                'the facts are:',
                'facts from the turn:'
            ]):
                continue
                
            facts.append(line)
        
        return facts
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            # Simple health check
            url = urljoin(self.base_url + "/", "api/tags")
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                models_data = json.loads(resp.read().decode("utf-8"))

            # Check if our model is available
            model_names = [model['name'] for model in models_data.get('models', [])]
            
            return any(self.model in name for name in model_names)
            
        except Exception:
            return False


# Global instance (lazy-initialized)
_extractor: Optional[FactExtractor] = None


def get_extractor(model: str = "qwen2.5:14b") -> FactExtractor:
    """Get the global fact extractor instance."""
    global _extractor
    if _extractor is None or _extractor.model != model:
        _extractor = FactExtractor(model=model)
    return _extractor


def extract_facts(content: str, model: str = "qwen2.5:14b") -> List[str]:
    """Convenience function to extract facts from content."""
    extractor = get_extractor(model)
    return extractor.extract_facts(content)
