"""Async LLM enrichment module for Synapse AI Memory.

Processes queued memories in background using Ollama for:
- Deduplication against existing memories
- Rewriting messy facts into canonical triples
- Proposing memory promotions
- Never blocks the agent response path
- Falls back gracefully if no LLM available
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse, Memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration and Types
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"  # Fast, small model for background processing
ENRICHMENT_BATCH_SIZE = 5
ENRICHMENT_TIMEOUT = 30  # seconds per request
SIMILARITY_THRESHOLD = 0.85  # For deduplication


@dataclass
class EnrichmentResult:
    """Result of enriching a memory."""
    original_text: str
    canonical_form: Optional[str] = None
    extracted_triples: Optional[List[str]] = None
    duplicate_of: Optional[str] = None  # Memory ID if duplicate
    promotion_suggestion: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None


class OllamaClient:
    """Simple Ollama client using only urllib (no external dependencies)."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        if self._available is not None:
            return self._available
            
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    # Check if our model is available
                    models = [m.get('name', '') for m in data.get('models', [])]
                    self._available = any(self.model in model for model in models)
                    return self._available
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False
        
        self._available = False
        return False
    
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.1) -> Optional[str]:
        """Generate text using Ollama API."""
        if not self.is_available():
            return None
            
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }
            
            if system:
                data["system"] = system
                
            json_data = json.dumps(data).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=ENRICHMENT_TIMEOUT) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode())
                    return result.get('response', '').strip()
                    
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            return None
        
        return None


# ---------------------------------------------------------------------------
# Enrichment Functions
# ---------------------------------------------------------------------------

def extract_canonical_triples(ollama: OllamaClient, text: str) -> Optional[List[str]]:
    """Extract structured triples from messy text."""
    system = """You are an expert at converting natural language into structured knowledge triples.
Extract factual information as simple subject-predicate-object triples.
Format: "subject | predicate | object"
Only return clear, factual triples. One per line."""
    
    prompt = f"""Convert this text to knowledge triples:

{text}

Triples:"""
    
    response = ollama.generate(prompt, system=system, max_tokens=300)
    if not response:
        return None
        
    triples = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3 and all(parts):
                triples.append(' | '.join(parts))
    
    return triples if triples else []


def canonicalize_text(ollama: OllamaClient, text: str) -> Optional[str]:
    """Rewrite messy text into a clean canonical form."""
    system = """You are an expert editor. Your job is to clean up and standardize text while preserving all important information.
- Fix typos and grammar
- Use consistent terminology
- Remove unnecessary filler words
- Preserve all factual content
- Keep it concise but complete"""
    
    prompt = f"""Clean up this text:

{text}

Canonical form:"""
    
    response = ollama.generate(prompt, system=system, max_tokens=200)
    if not response:
        return None
        
    canonical = response.strip()
    # Basic quality check - shouldn't be too different in length
    if len(canonical) < len(text) * 0.3 or len(canonical) > len(text) * 2:
        return None
        
    return canonical


def check_for_duplicates(ollama: OllamaClient, new_text: str, existing_memories: List["Memory"]) -> Optional[str]:
    """Check if new text is similar to existing memories."""
    if not existing_memories:
        return None
        
    # Compare against recent memories (last 100)
    recent_memories = existing_memories[-100:]
    
    system = """You are an expert at detecting duplicate or highly similar pieces of information.
Compare the new text against existing memories and determine if it's essentially the same information.
Return only: DUPLICATE:<memory_id> or UNIQUE"""
    
    existing_text = "\n".join([f"ID:{m.id} - {m.content[:100]}" for m in recent_memories])
    
    prompt = f"""New text: {new_text}

Existing memories:
{existing_text}

Is this new text a duplicate?"""
    
    response = ollama.generate(prompt, system=system, max_tokens=50)
    if not response:
        return None
        
    response = response.strip()
    if response.upper().startswith("DUPLICATE:"):
        memory_id = response.split(":", 1)[1].strip()
        return memory_id
        
    return None


def suggest_promotion(ollama: OllamaClient, text: str, category: str) -> Optional[str]:
    """Suggest if a memory should be promoted to a higher level."""
    system = """You are an expert at determining the importance and scope of information.
Analyze if this memory represents:
- INSTANCE: Specific one-time event or fact
- PATTERN: Recurring behavior or trend  
- PROFILE: Long-term preference or characteristic

Return only: PROMOTE:<new_level>:<reason> or KEEP:<reason>"""
    
    prompt = f"""Category: {category}
Text: {text}

Should this be promoted?"""
    
    response = ollama.generate(prompt, system=system, max_tokens=100)
    if not response:
        return None
        
    response = response.strip()
    if response.startswith("PROMOTE:"):
        parts = response.split(":", 2)
        if len(parts) >= 3:
            return f"{parts[1]}:{parts[2]}"
            
    return None


# ---------------------------------------------------------------------------
# Async Enrichment Engine
# ---------------------------------------------------------------------------

class MemoryEnricher:
    """Async background enrichment for Synapse memories."""
    
    def __init__(self, synapse: "Synapse", ollama_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.synapse = synapse
        self.ollama = OllamaClient(ollama_url, model)
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_enrichment = 0
    
    async def start_background_enrichment(self, interval: float = 60.0) -> None:
        """Start background enrichment loop."""
        if not self.ollama.is_available():
            logger.info("Ollama not available, enrichment disabled")
            return
            
        self.running = True
        logger.info(f"Starting memory enrichment (interval={interval}s)")
        
        while self.running:
            try:
                await asyncio.sleep(interval)
                await self.enrich_pending_memories()
            except Exception as e:
                logger.error(f"Enrichment loop error: {e}")
                self.error_count += 1
                
    def stop_background_enrichment(self) -> None:
        """Stop background enrichment."""
        self.running = False
        
    async def enrich_pending_memories(self) -> List[EnrichmentResult]:
        """Enrich memories that haven't been processed yet."""
        if not self.ollama.is_available():
            return []
            
        # Find memories that need enrichment (recent, not yet enriched)
        cutoff_time = time.time() - (24 * 60 * 60)  # Last 24 hours
        all_memories = self.synapse.list(limit=1000)

        pending = []
        for memory in all_memories:
            if (
                memory.created_at > cutoff_time
                and not memory.metadata.get('enriched')
                and not memory.metadata.get('router_auto_stored')
            ):
                pending.append(memory)
                
        if not pending:
            return []
            
        # Process in batches
        results = []
        for i in range(0, len(pending), ENRICHMENT_BATCH_SIZE):
            batch = pending[i:i + ENRICHMENT_BATCH_SIZE]
            batch_results = await self._enrich_batch(batch)
            results.extend(batch_results)
            
        logger.info(f"Enriched {len(results)} memories")
        return results
        
    async def _enrich_batch(self, memories: List["Memory"]) -> List[EnrichmentResult]:
        """Enrich a batch of memories."""
        results = []
        
        for memory in memories:
            start_time = time.time()
            result = EnrichmentResult(original_text=memory.content)
            
            try:
                # Check for duplicates first
                existing_memories = self.synapse.list(limit=500)
                duplicate_id = check_for_duplicates(self.ollama, memory.content, existing_memories)
                if duplicate_id:
                    result.duplicate_of = duplicate_id
                    # Could mark original for deletion/merge here
                
                # Extract canonical form
                canonical = canonicalize_text(self.ollama, memory.content)
                if canonical and canonical != memory.content:
                    result.canonical_form = canonical
                    
                # Extract triples for structured memories
                triples = extract_canonical_triples(self.ollama, memory.content)
                if triples:
                    result.extracted_triples = triples
                    
                # Suggest promotions
                category = memory.metadata.get('router_category', getattr(memory, 'memory_type', 'fact') or 'fact')
                promotion = suggest_promotion(self.ollama, memory.content, category)
                if promotion:
                    result.promotion_suggestion = promotion
                    
                result.confidence = 0.8  # Base confidence for successful enrichment
                
                # Record enrichment by appending a linked enrichment memory.
                enrichment_meta = {
                    'source_memory_id': memory.id,
                    'enriched': True,
                    'enriched_at': time.time(),
                    'canonical_form': result.canonical_form,
                    'extracted_triples': result.extracted_triples,
                    'duplicate_of': result.duplicate_of,
                    'promotion_suggestion': result.promotion_suggestion,
                }
                self.synapse.remember(
                    content=f"enrichment:{memory.id}",
                    memory_type="semantic",
                    metadata=enrichment_meta,
                    deduplicate=False,
                    extract=False,
                )

                self.processed_count += 1
                
            except Exception as e:
                result.error = str(e)
                result.confidence = 0.0
                logger.warning(f"Failed to enrich memory {memory.id}: {e}")
                self.error_count += 1
                
            finally:
                result.processing_time = time.time() - start_time
                results.append(result)
                
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        return {
            "ollama_available": self.ollama.is_available(),
            "running": self.running,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "last_enrichment": self.last_enrichment,
        }


# ---------------------------------------------------------------------------
# Synchronous Interface (for CLI/direct usage)
# ---------------------------------------------------------------------------

def enrich_memory_sync(synapse: "Synapse", memory: "Memory", ollama_url: str = OLLAMA_BASE_URL) -> EnrichmentResult:
    """Synchronously enrich a single memory."""
    ollama = OllamaClient(ollama_url)
    
    if not ollama.is_available():
        return EnrichmentResult(
            original_text=memory.content,
            error="Ollama not available",
            confidence=0.0
        )
    
    start_time = time.time()
    result = EnrichmentResult(original_text=memory.content)
    
    try:
        # Get existing memories for deduplication
        existing_memories = synapse.list(limit=200)
        
        # Run enrichment functions
        result.duplicate_of = check_for_duplicates(ollama, memory.content, existing_memories)
        result.canonical_form = canonicalize_text(ollama, memory.content)
        result.extracted_triples = extract_canonical_triples(ollama, memory.content)
        
        category = memory.metadata.get('router_category', getattr(memory, 'memory_type', 'fact') or 'fact')
        result.promotion_suggestion = suggest_promotion(ollama, memory.content, category)
        
        result.confidence = 0.8
        
    except Exception as e:
        result.error = str(e)
        result.confidence = 0.0
        
    finally:
        result.processing_time = time.time() - start_time
        
    return result


def batch_enrich_memories(synapse: "Synapse", memory_ids: List[str], ollama_url: str = OLLAMA_BASE_URL) -> List[EnrichmentResult]:
    """Synchronously enrich multiple memories by ID."""
    results = []
    
    all_memories = {str(m.id): m for m in synapse.list(limit=5000)}

    for memory_id in memory_ids:
        try:
            memory = all_memories.get(str(memory_id))
            if memory:
                result = enrich_memory_sync(synapse, memory, ollama_url)
                results.append(result)
            else:
                results.append(
                    EnrichmentResult(
                        original_text=f"Memory {memory_id} not found",
                        error=f"Memory {memory_id} not found",
                        confidence=0.0,
                    )
                )
        except Exception as e:
            results.append(
                EnrichmentResult(
                    original_text=f"Memory {memory_id}",
                    error=str(e),
                    confidence=0.0,
                )
            )
            
    return results