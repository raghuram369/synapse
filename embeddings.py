"""
Embedding index for Synapse V2 — optional semantic similarity layer.
Uses Ollama's nomic-embed-text for local embeddings (no API keys, no cloud).
"""

import json
import math
import urllib.request
from typing import Dict, List, Optional, Tuple

OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


def _embed(text: str, model: str = DEFAULT_MODEL) -> Optional[List[float]]:
    """Get embedding vector from Ollama. Returns None on failure."""
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception:
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingIndex:
    """In-memory embedding index for semantic similarity search."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.vectors: Dict[int, List[float]] = {}  # memory_id -> embedding
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if Ollama embedding model is available."""
        if self._available is not None:
            return self._available
        vec = _embed("test", self.model)
        self._available = vec is not None
        return self._available
    
    def add(self, memory_id: int, text: str) -> bool:
        """Embed and store a memory. Returns True on success."""
        vec = _embed(text, self.model)
        if vec is None:
            return False
        self.vectors[memory_id] = vec
        return True
    
    def remove(self, memory_id: int):
        """Remove a memory's embedding."""
        self.vectors.pop(memory_id, None)
    
    def search(self, query: str, limit: int = 10, 
               exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        """Find most similar memories to query. Returns [(memory_id, similarity)]."""
        query_vec = _embed(query, self.model)
        if query_vec is None:
            return []
        
        scores = []
        for mid, vec in self.vectors.items():
            if exclude and mid in exclude:
                continue
            sim = cosine_similarity(query_vec, vec)
            if sim > 0:
                scores.append((mid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
    
    def get_candidates(self, query: str, limit: int = 50) -> Dict[int, float]:
        """Get candidate memories with similarity scores (for blending with BM25)."""
        results = self.search(query, limit=limit)
        return {mid: score for mid, score in results}
    
    def __len__(self):
        return len(self.vectors)
