"""
Synapse AI Memory — A neuroscience-inspired memory database engine for AI agents.

Pure Python, zero external dependencies, no SQL.  Provides BM25 keyword
search, concept-graph expansion, optional local embeddings (Ollama), and
an append-only log + snapshot persistence layer.
"""

from __future__ import annotations

#
# NOTE: This repository historically exposed `synapse` as a single-module API
# (`synapse.py`). Some users want `synapse.compat.*` imports (Mem0 shims, etc.).
# A regular module cannot have submodules, but the import system treats any
# module with a `__path__` as a package. We set `__path__` to point at the
# adjacent `synapse/` directory (if present) so `import synapse.compat` works
# without breaking the existing `import synapse` module API.
#
import os as _os
_compat_pkg_dir = _os.path.join(_os.path.dirname(__file__), "synapse")
if _os.path.isdir(_compat_pkg_dir):
    __path__ = [_compat_pkg_dir]  # type: ignore[name-defined]

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from embeddings import EmbeddingIndex
from entity_graph import extract_concepts, expand_query
from exceptions import SynapseValidationError
from extractor import extract_facts
from indexes import (
    ConceptGraph,
    EdgeGraph,
    EpisodeIndex,
    InvertedIndex,
    TemporalIndex,
)
from storage import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_TYPES: Set[str] = {"fact", "event", "preference", "skill", "observation", "consolidated"}
MEMORY_LEVELS: Set[str] = {"instance", "pattern", "profile"}
EDGE_TYPES: Set[str] = {
    "caused_by", "contradicts", "reminds_of", "supports",
    "preceded", "followed", "supersedes", "related",
}
DECAY_HALF_LIFE: float = 86400.0 * 7  # 7 days in seconds
REINFORCE_BOOST: float = 0.05
DEFAULT_EPISODE_WINDOW_SECS: float = 1800.0


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a memory's recall score was computed."""

    bm25_score: float = 0.0
    concept_score: float = 0.0
    temporal_score: float = 0.0
    episode_score: float = 0.0
    concept_activation_score: float = 0.0
    embedding_score: float = 0.0
    match_sources: List[str] = field(default_factory=list)


@dataclass
class Memory:
    """A single memory record with temporal-decay strength."""

    id: Optional[int] = None
    content: str = ""
    memory_type: str = "fact"
    memory_level: str = "instance"
    strength: float = 1.0
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    consolidated: bool = False
    summary_of: List[int] = field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = field(default=None, repr=False)

    @property
    def effective_strength(self) -> float:
        """Strength after temporal decay and access-frequency boost."""
        age = time.time() - self.last_accessed
        decay = math.pow(0.5, age / DECAY_HALF_LIFE)
        access_boost = 1.0 + 0.1 * math.log1p(self.access_count)
        return self.strength * decay * access_boost


class Synapse:
    """Neuroscience-inspired memory engine — pure Python, zero dependencies.

    Provides BM25 keyword search, concept-graph expansion, optional local
    embeddings via Ollama, temporal decay, episode grouping, and an
    append-only-log persistence layer.

    Example::

        s = Synapse()
        s.remember("I prefer vegetarian food")
        results = s.recall("dietary preferences")
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        
        # Initialize storage
        self.store = MemoryStore(path)
        
        # Initialize indexes
        self.inverted_index = InvertedIndex()
        self.concept_graph = ConceptGraph()
        self.edge_graph = EdgeGraph()
        self.temporal_index = TemporalIndex()
        self.episode_index = EpisodeIndex()
        
        # Embedding index (optional — uses Ollama if available)
        self.embedding_index = EmbeddingIndex()
        self._use_embeddings: Optional[bool] = None  # lazy check
        
        # Episode tracking
        self._episode_window_secs = DEFAULT_EPISODE_WINDOW_SECS
        self._cached_episode_id: Optional[int] = None
        self._cached_episode_last_at: float = 0.0
        
        # Performance settings
        self._commit_batch_size = 100
        self._pending_commits = 0
        
        # Build indexes from stored data
        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from stored data."""
        # Rebuild inverted index
        for memory_id, memory_data in self.store.memories.items():
            if not memory_data.get('consolidated', False):
                self.inverted_index.add_document(memory_id, memory_data['content'])
                self.temporal_index.add_memory(memory_id, memory_data['created_at'])
        
        # Rebuild concept graph
        for memory_id, memory_data in self.store.memories.items():
            if not memory_data.get('consolidated', False):
                concepts = extract_concepts(memory_data['content'])
                for concept_name, category in concepts:
                    self.concept_graph.link_memory_concept(memory_id, concept_name, category)
        
        # Rebuild edge graph
        for edge_data in self.store.edges.values():
            self.edge_graph.add_edge(
                edge_data['source_id'],
                edge_data['target_id'], 
                edge_data['edge_type'],
                edge_data['weight']
            )
        
        # Rebuild episode index
        for episode_data in self.store.episodes.values():
            self.episode_index.add_episode(
                episode_data['id'],
                episode_data['name'],
                episode_data['started_at'],
                episode_data['ended_at']
            )
            # Link memories to episodes (would need to be tracked in storage)
    
    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        links: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        episode: Optional[str] = None,
        deduplicate: bool = True,
        extract: bool = False,
    ) -> Memory:
        """Store a new memory with optional fact extraction and deduplication.

        Args:
            content: The text content to memorise.
            memory_type: One of ``MEMORY_TYPES``.
            links: Optional edges to create (list of dicts with
                ``target_id``, ``edge_type``, and optionally ``weight``).
            metadata: Arbitrary JSON-serialisable metadata.
            episode: Optional episode name to group this memory into.
            deduplicate: If *True*, create *supersedes* edges to similar
                existing memories.
            extract: If *True*, use an LLM to split *content* into
                individual facts and store each separately.

        Returns:
            The newly created ``Memory`` (or the first extracted fact).

        Raises:
            SynapseValidationError: If *content* is empty or *memory_type*
                is not recognised.
        """
        now = time.time()

        if not content.strip():
            raise SynapseValidationError("Memory content cannot be empty")
        if memory_type not in MEMORY_TYPES:
            raise SynapseValidationError(
                f"Invalid memory_type: {memory_type}. Must be one of {MEMORY_TYPES}"
            )

        if extract:
            return self._remember_with_extraction(
                content, memory_type, links, metadata, episode, deduplicate, now,
            )
        return self._remember_single_content(
            content, memory_type, links, metadata, episode, deduplicate, now,
        )
    
    def _remember_with_extraction(self, content: str, memory_type: str, 
                                  links: Optional[List], metadata: Optional[Dict],
                                  episode: Optional[str], deduplicate: bool, now: float) -> Memory:
        """Handle memory storage with fact extraction."""
        try:
            # Extract facts using LLM
            facts = extract_facts(content)
            
            if not facts:
                logger.warning("No facts extracted from content, storing as-is")
                return self._remember_single_content(content, memory_type, links, metadata, episode, deduplicate, now)
            
            # Create memories for each extracted fact
            fact_memories = []
            fact_ids = []
            
            for i, fact in enumerate(facts):
                # Create metadata with original content reference
                fact_metadata = metadata.copy() if metadata else {}
                fact_metadata.update({
                    'original_content': content,
                    'extraction_index': i,
                    'extracted_fact': True,
                    'total_facts': len(facts)
                })
                
                # Store each fact as a separate memory
                fact_memory = self._remember_single_content(
                    fact, memory_type, None, fact_metadata, episode, deduplicate, now
                )
                fact_memories.append(fact_memory)
                fact_ids.append(fact_memory.id)
            
            # Link all extracted facts together with "related" edges
            for i, source_id in enumerate(fact_ids):
                for j, target_id in enumerate(fact_ids):
                    if i != j:  # Don't link to self
                        self.link(source_id, target_id, "related", 0.8)
            
            # Handle original links - apply to all fact memories
            if links:
                for fact_memory in fact_memories:
                    for link in links:
                        if isinstance(link, dict):
                            target_id = link.get('target_id')
                            edge_type = link.get('edge_type', 'reminds_of')
                            weight = link.get('weight', 1.0)
                            if target_id:
                                self.link(fact_memory.id, target_id, edge_type, weight)
            
            # Return the first fact memory as the primary result
            # (This maintains the API contract of returning a single Memory)
            return fact_memories[0]
            
        except Exception as e:
            logger.warning("Fact extraction failed (%s), storing content as-is", e)
            return self._remember_single_content(content, memory_type, links, metadata, episode, deduplicate, now)
    
    def _remember_single_content(self, content: str, memory_type: str, 
                                 links: Optional[List], metadata: Optional[Dict],
                                 episode: Optional[str], deduplicate: bool, now: float) -> Memory:
        """Store a single piece of content as a memory (extracted from original remember logic)."""
        # Check for duplicates if requested
        similar_memories: List[Memory] = []
        supersedes_target: Optional[Memory] = None
        if deduplicate:
            similar_memories = self._find_similar_memories(content)
            # For temporal fact chains, keep supersession linear by picking the best match.
            if similar_memories:
                supersedes_target = similar_memories[0]
        
        # Create memory data
        memory_data = {
            'content': content,
            'memory_type': memory_type,
            'memory_level': 'instance',
            'strength': 1.0,
            'access_count': 0,
            'created_at': now,
            'last_accessed': now,
            'metadata': json.dumps(metadata or {}),
            'consolidated': False,
            'summary_of': json.dumps([])
        }
        
        # Store in persistent storage
        memory_id = self.store.insert_memory(memory_data)
        
        # Update indexes
        self.inverted_index.add_document(memory_id, content)
        self.temporal_index.add_memory(memory_id, now)
        
        # Embed if available (lazy init)
        if self._use_embeddings is None:
            self._use_embeddings = self.embedding_index.is_available()
        if self._use_embeddings:
            self.embedding_index.add(memory_id, content)
        
        # Extract and index concepts
        concepts = extract_concepts(content)
        for concept_name, category in concepts:
            self.concept_graph.link_memory_concept(memory_id, concept_name, category)
            
        # Handle episode assignment
        if episode:
            episode_id = self._find_or_create_episode(episode, now)
            self.episode_index.add_memory_to_episode(memory_id, episode_id)
            
            # Also update persistent storage
            if episode_id in self.store.episodes:
                if memory_id not in self.store.episodes[episode_id].get('memory_ids', []):
                    self.store.episodes[episode_id].setdefault('memory_ids', []).append(memory_id)
        
        # Add edges if specified
        if links:
            for link in links:
                if isinstance(link, dict):
                    target_id = link.get('target_id')
                    edge_type = link.get('edge_type', 'reminds_of')
                    weight = link.get('weight', 1.0)
                    if target_id:
                        self.link(memory_id, target_id, edge_type, weight)
        
        # Handle supersession for similar memory (temporal fact chain)
        if deduplicate and supersedes_target is not None:
            self.link(memory_id, supersedes_target.id, "supersedes", 1.0)
            
            # Build temporal chain metadata
            old_meta = json.loads(self.store.memories[supersedes_target.id].get('metadata', '{}'))
            old_meta['superseded_by'] = memory_id
            old_meta['superseded_at'] = now
            self.store.update_memory(supersedes_target.id, {'metadata': json.dumps(old_meta)})
            
            # Determine chain id: extend existing chain or start a new one
            chain_id = old_meta.get('fact_chain_id', f"chain-{supersedes_target.id}")
            
            new_meta = metadata or {}
            new_meta['supersedes'] = supersedes_target.id
            new_meta['fact_chain_id'] = chain_id
            metadata = new_meta
            
            # If old memory didn't have a chain id yet, give it one
            if 'fact_chain_id' not in old_meta:
                old_meta['fact_chain_id'] = chain_id
                self.store.update_memory(supersedes_target.id, {'metadata': json.dumps(old_meta)})
            
            # Update the stored metadata for the new memory
            self.store.update_memory(memory_id, {'metadata': json.dumps(metadata)})
        
        # Periodic maintenance
        self._pending_commits += 1
        if self._pending_commits >= self._commit_batch_size:
            self.flush()
        
        # Return Memory object
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            memory_level='instance',
            strength=1.0,
            access_count=0,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
            consolidated=False
        )
        
        return memory
    
    def _find_similar_memories(self, content: str, threshold: float = 0.3) -> List[Memory]:
        """Find memories similar to the given content."""
        # Use recall to find similar memories
        similar = self.recall(content, limit=10)
        
        # Filter by similarity threshold
        content_tokens = set(self.inverted_index.tokenize_for_index(content))
        if not content_tokens:
            return []
            
        filtered = []
        for memory in similar:
            memory_tokens = set(self.inverted_index.tokenize_for_index(memory.content))
            if not memory_tokens:
                continue
                
            # Compute Jaccard similarity for word overlap
            intersection = len(content_tokens & memory_tokens)
            union = len(content_tokens | memory_tokens)
            word_similarity = intersection / union if union > 0 else 0.0
            
            # Compute concept similarity  
            content_concepts = set(c[0] for c in extract_concepts(content))
            memory_concepts = set(c[0] for c in extract_concepts(memory.content))
            concept_intersection = len(content_concepts & memory_concepts)
            concept_union = len(content_concepts | memory_concepts)
            concept_similarity = concept_intersection / concept_union if concept_union > 0 else 0.0
            
            # Combined similarity
            combined_similarity = 0.7 * word_similarity + 0.3 * concept_similarity
            
            if combined_similarity >= threshold:
                filtered.append(memory)
                
        return filtered
    
    def recall(self, context: str = "", limit: int = 10,
               memory_type: Optional[str] = None, min_strength: float = 0.01,
               temporal_boost: bool = True, explain: bool = False,
               temporal: Optional[str] = None) -> List[Memory]:
        """
        Two-Stage Recall with adaptive blending (ported from V1):
          Stage 1A: BM25 word-index candidates  
          Stage 1B: Concept-IDF candidates
          Stage 2: Normalize, blend, apply effective strength, edge/episode expansion
        """
        now = time.time()

        # Temporal fact-chain recall modes.
        # - None / "latest": default behaviour (current facts)
        # - "all": return the full fact chain for the best match (oldest -> newest)
        # - date/timestamp string: return the version that was current at that point in time
        if temporal is not None and temporal != "latest":
            # Find the best matching current memory first, then resolve within its chain.
            base = self.recall(context=context, limit=1, memory_type=memory_type,
                               min_strength=min_strength, temporal_boost=temporal_boost,
                               explain=explain, temporal=None)
            if not base:
                return []
            head = base[0]
            chain = self.history(head.id)
            if temporal == "all":
                return [entry["memory"] for entry in chain]
            at_ts = self._parse_temporal_arg(temporal)
            if at_ts is None:
                return [head]
            resolved = self._resolve_chain_at(chain, at_ts)
            return [resolved] if resolved is not None else []

        # Tokenize query
        query_tokens = self.inverted_index.tokenize_for_query(context) if context else []

        # No tokens: return by effective strength
        if not query_tokens:
            all_memories = []
            for memory_id, memory_data in self.store.memories.items():
                if memory_data.get('consolidated', False):
                    continue
                if memory_type and memory_data.get('memory_type') != memory_type:
                    continue
                    
                memory = self._memory_data_to_object(memory_data)
                if memory.effective_strength >= min_strength:
                    all_memories.append(memory)
            
            # Sort by effective strength
            all_memories.sort(key=lambda m: m.effective_strength, reverse=True)
            result = all_memories[:limit]
            
            # Reinforce accessed memories
            self._reinforce_memories([m.id for m in result])
            return result
        
        # Count total active memories
        total_memories = len([m for m in self.store.memories.values() 
                             if not m.get('consolidated', False)
                             and (not memory_type or m.get('memory_type') == memory_type)])
        
        if total_memories == 0:
            return []
        
        # ════════════════════════════════════════════════════════════
        #  STAGE 1: Parallel candidate generation
        # ════════════════════════════════════════════════════════════
        
        # Stage 1A: BM25 candidates
        bm25_scores = self.inverted_index.get_candidates(query_tokens, limit * 10)
        
        # Filter by memory_type and consolidated status
        bm25_scores = {
            mid: score for mid, score in bm25_scores.items()
            if mid in self.store.memories
            and not self.store.memories[mid].get('consolidated', False)
            and (not memory_type or self.store.memories[mid].get('memory_type') == memory_type)
        }
        
        # Stage 1B: Concept-IDF candidates  
        query_concepts = [c[0] for c in extract_concepts(context)] if context else []
        expanded_concepts = expand_query(query_concepts) if query_concepts else []
        all_concepts = list(set(query_concepts + expanded_concepts))
        
        concept_scores = self.concept_graph.get_candidates(all_concepts, limit * 10, total_memories)
        
        # Filter concept candidates too
        concept_scores = {
            mid: score for mid, score in concept_scores.items()
            if mid in self.store.memories
            and not self.store.memories[mid].get('consolidated', False)
            and (not memory_type or self.store.memories[mid].get('memory_type') == memory_type)
        }
        
        # ════════════════════════════════════════════════════════════
        # Stage 1C: Embedding candidates (if available)
        embedding_scores: Dict[int, float] = {}
        if self._use_embeddings and context:
            embedding_scores = self.embedding_index.get_candidates(context, limit=limit * 5)
            # Filter
            embedding_scores = {
                mid: score for mid, score in embedding_scores.items()
                if mid in self.store.memories
                and not self.store.memories[mid].get('consolidated', False)
                and (not memory_type or self.store.memories[mid].get('memory_type') == memory_type)
            }
        
        #  STAGE 2: Concept-Boosted BM25 + Effective Strength
        # ════════════════════════════════════════════════════════════
        
        all_ids = set(bm25_scores.keys()) | set(concept_scores.keys()) | set(embedding_scores.keys())
        if not all_ids:
            return []
        
        # Adaptive concept weighting based on query length
        # BM25 is the PRIMARY signal; concepts only boost, never inject
        n_tokens = len(query_tokens)
        if n_tokens <= 1:
            concept_weight = 0.5
        elif n_tokens <= 2:
            concept_weight = 0.3
        elif n_tokens <= 4:
            concept_weight = 0.2
        else:
            concept_weight = 0.0    # pure BM25 for long queries
        
        # Normalize concept scores
        c_max = max(concept_scores.values()) if concept_scores else 1.0
        if c_max <= 0:
            c_max = 1.0
        
        # Build per-signal components (all normalized to ~[0,1])
        lexical_scores: Dict[int, float] = {}
        concept_norms: Dict[int, float] = {}

        # Track per-memory breakdowns when explain=True
        breakdowns: Dict[int, ScoreBreakdown] = {}

        # Normalize BM25 scores to [0,1]
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
        if bm25_max <= 0:
            bm25_max = 1.0

        # Determine embedding weight based on query characteristics
        # Short queries benefit more from embeddings; long specific queries favor BM25
        if embedding_scores:
            if n_tokens <= 3:
                emb_weight = 0.4
            elif n_tokens <= 6:
                emb_weight = 0.3
            else:
                emb_weight = 0.2
            bm_weight = 1.0 - emb_weight
        else:
            bm_weight = 1.0
            emb_weight = 0.0

        for memory_id in all_ids:
            bm = bm25_scores.get(memory_id, 0.0) / bm25_max  # 0-1
            cn = concept_scores.get(memory_id, 0.0) / c_max   # 0-1
            emb = embedding_scores.get(memory_id, 0.0)         # cosine, ~0-1

            # Treat "BM25" component as lexical (BM25 + optional embedding fusion)
            lexical = bm * bm_weight + emb * emb_weight

            # Keep a *tiny* concept boost inside lexical for very short queries
            lexical = lexical + cn * concept_weight * 0.05

            if lexical > 0:
                lexical_scores[memory_id] = lexical
                concept_norms[memory_id] = cn

            if explain:
                sources = []
                if memory_id in bm25_scores:
                    sources.append("bm25")
                if memory_id in concept_scores:
                    sources.append("concept_graph")
                if memory_id in embedding_scores:
                    sources.append("embedding")
                breakdowns[memory_id] = ScoreBreakdown(
                    bm25_score=lexical,
                    concept_score=cn,
                    embedding_score=emb,
                    match_sources=sources,
                )
        
        if not lexical_scores:
            return []
        
        # Load memory objects and compute multi-signal score components
        candidates = []
        for memory_id, lexical in lexical_scores.items():
            if memory_id not in self.store.memories:
                continue

            memory_data = self.store.memories[memory_id]
            memory = self._memory_data_to_object(memory_data)
            if memory.effective_strength < min_strength:
                continue

            # Temporal component (recency): same half-life as memory decay
            age = max(0.0, now - memory.last_accessed)
            temporal_component = math.pow(0.5, age / DECAY_HALF_LIFE)

            # Concept activation component: average activation strength of this memory's concepts
            concept_activation_component = self.concept_graph.memory_concept_activation_score(memory_id, now=now)

            cn = concept_norms.get(memory_id, 0.0)

            # Provisional score (episode signal computed after anchor selection)
            provisional = (
                0.30 * lexical +
                0.25 * cn +
                0.20 * temporal_component +
                0.15 * concept_activation_component
            )

            if explain and memory_id in breakdowns:
                breakdowns[memory_id].temporal_score = temporal_component
                breakdowns[memory_id].concept_activation_score = concept_activation_component

            candidates.append((memory, provisional))
        
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:limit * 2]  # Get more for expansion

        # Episode component (0/1): boost memories in the same episode as the top result
        anchor_episode_id: Optional[int] = None
        if top_candidates:
            anchor_episode_id = self.episode_index.get_memory_episode(top_candidates[0][0].id)

        if anchor_episode_id is not None:
            adjusted = []
            for memory, score in top_candidates:
                ep = self.episode_index.get_memory_episode(memory.id)
                episode_component = 1.0 if ep == anchor_episode_id else 0.0
                score = score + 0.10 * episode_component
                if explain and memory.id in breakdowns:
                    breakdowns[memory.id].episode_score = episode_component
                adjusted.append((memory, score))
            top_candidates = adjusted

        # ════════════════════════════════════════════════════════════
        #  POST-SCORING: Edge spreading + Episode expansion + Temporal boost
        # ════════════════════════════════════════════════════════════
        
        memory_scores = {memory.id: score for memory, score in top_candidates}
        top_memory_ids = [memory.id for memory, _ in top_candidates[:limit]]
        
        # 1-hop edge spreading
        for memory_id in top_memory_ids:
            score = memory_scores.get(memory_id, 0.0)
            if score <= 0:
                continue
                
            # Spread activation to connected memories
            for target_id, edge in self.edge_graph.get_all_edges(memory_id):
                if target_id in self.store.memories and not self.store.memories[target_id].get('consolidated', False):
                    # Only boost existing candidates, don't inject via edges
                    if target_id in memory_scores:
                        memory_scores[target_id] *= 1.0 + edge.weight * 0.1
        
        # Episode siblings: only boost existing candidates, don't inject new ones
        for memory_id in top_memory_ids:
            score = memory_scores.get(memory_id, 0.0)
            if score <= 0:
                continue
                
            siblings = self.episode_index.get_episode_siblings(memory_id)
            for sibling_id in siblings:
                if (sibling_id != memory_id and 
                    sibling_id in memory_scores and  # only boost existing candidates
                    sibling_id in self.store.memories and 
                    not self.store.memories[sibling_id].get('consolidated', False)):
                    memory_scores[sibling_id] *= 1.05  # tiny boost, not additive
                    if explain and sibling_id in breakdowns:
                        breakdowns[sibling_id].episode_score = max(breakdowns[sibling_id].episode_score, 0.05)
        
        # Temporal proximity: very mild boost only for existing candidates
        if temporal_boost and top_memory_ids:
            anchor_memory = self.store.memories[top_memory_ids[0]]
            anchor_time = anchor_memory['created_at']
            
            temporal_candidates = self.temporal_index.get_memories_around_time(anchor_time, 3600.0)
            for memory_id in temporal_candidates:
                if (memory_id in memory_scores and 
                    memory_id in self.store.memories and
                    not self.store.memories[memory_id].get('consolidated', False)):
                    if not memory_type or self.store.memories[memory_id].get('memory_type') == memory_type:
                        memory_scores[memory_id] *= 1.05
                        if explain and memory_id in breakdowns:
                            breakdowns[memory_id].temporal_score = max(breakdowns[memory_id].temporal_score, 0.05)
        
        # Handle supersession (demote memories that are superseded)
        all_candidate_ids = set(memory_scores.keys())
        superseded_ids = self.edge_graph.get_superseded_memories(all_candidate_ids)
        
        # Final ranking
        final_results = []
        for memory_id, score in memory_scores.items():
            if memory_id not in self.store.memories:
                continue
                
            memory_data = self.store.memories[memory_id]
            if memory_data.get('consolidated', False):
                continue
                
            if memory_type and memory_data.get('memory_type') != memory_type:
                continue
                
            memory = self._memory_data_to_object(memory_data)
            if memory.effective_strength < min_strength:
                continue
            
            # Apply supersession penalty
            if memory_id in superseded_ids:
                score *= 0.1  # Heavily demote superseded memories

            # Memory level boost: pattern (consolidated) and profile rank higher
            level = memory_data.get('memory_level', 'instance')
            if level == 'pattern':
                score *= 1.3
            elif level == 'profile':
                score *= 1.5

            final_results.append((memory, score))
        
        # Sort and limit
        final_results.sort(key=lambda x: x[1], reverse=True)
        result_memories = [memory for memory, _ in final_results[:limit]]
        
        # Attach score breakdowns if explain=True
        if explain:
            for memory in result_memories:
                if memory.id in breakdowns:
                    memory.score_breakdown = breakdowns[memory.id]
        
        # Activate concepts referenced by returned memories (concept-level reinforcement)
        self.concept_graph.activate_memories([m.id for m in result_memories], now=now)

        # Reinforce accessed memories (per-memory reinforcement)
        self._reinforce_memories([m.id for m in result_memories])

        return result_memories
    
    def _memory_data_to_object(self, memory_data: Dict) -> Memory:
        """Convert stored memory data to Memory object."""
        return Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            memory_type=memory_data['memory_type'],
            memory_level=memory_data.get('memory_level', 'instance'),
            strength=memory_data['strength'],
            access_count=memory_data['access_count'],
            created_at=memory_data['created_at'],
            last_accessed=memory_data['last_accessed'],
            metadata=json.loads(memory_data.get('metadata', '{}')),
            consolidated=memory_data.get('consolidated', False),
            summary_of=json.loads(memory_data.get('summary_of', '[]'))
        )
    
    def _reinforce_memories(self, memory_ids: List[int]):
        """Reinforce memories by incrementing access count and updating last_accessed."""
        now = time.time()
        for memory_id in memory_ids:
            if memory_id in self.store.memories:
                updates = {
                    'access_count': self.store.memories[memory_id]['access_count'] + 1,
                    'last_accessed': now,
                    'strength': min(self.store.memories[memory_id]['strength'] + REINFORCE_BOOST, 2.0)
                }
                self.store.update_memory(memory_id, updates)
    
    def forget(self, memory_id: int) -> bool:
        """Remove a memory completely."""
        if memory_id not in self.store.memories:
            return False
            
        # Remove from indexes
        self.inverted_index.remove_document(memory_id)
        self.concept_graph.unlink_memory(memory_id)
        self.edge_graph.remove_memory_edges(memory_id)
        self.temporal_index.remove_memory(memory_id)
        self.episode_index.remove_memory(memory_id)
        self.embedding_index.remove(memory_id)
        
        # Remove from storage
        return self.store.delete_memory(memory_id)
    
    def link(self, source_id: int, target_id: int, edge_type: str, weight: float = 1.0) -> None:
        """Create a directed edge between two memories.

        Raises:
            SynapseValidationError: If *edge_type* is unrecognised or
                either memory does not exist.
        """
        if edge_type not in EDGE_TYPES:
            raise SynapseValidationError(
                f"Invalid edge_type: {edge_type}. Must be one of {EDGE_TYPES}"
            )
        if source_id not in self.store.memories or target_id not in self.store.memories:
            raise SynapseValidationError("Both source and target memories must exist")
        
        # Add to edge graph
        self.edge_graph.add_edge(source_id, target_id, edge_type, weight)
        
        # Store in persistent storage
        edge_data = {
            'source_id': source_id,
            'target_id': target_id,
            'edge_type': edge_type,
            'weight': weight,
            'created_at': time.time()
        }
        self.store.insert_edge(edge_data)
    
    def consolidate(self, min_cluster_size: int = 3, similarity_threshold: float = 0.7,
                    max_age_days: Optional[float] = None, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Consolidate clusters of similar memories into stronger summary memories.

        Algorithm:
        1. Group memories by concept overlap (sharing 2+ concepts).
        2. Within groups, compute pairwise Jaccard similarity on word shingles.
        3. Clusters >= *min_cluster_size* get merged into a single consolidated
           memory with boosted strength.

        Args:
            min_cluster_size: Minimum cluster size to trigger consolidation.
            similarity_threshold: Jaccard similarity threshold for clustering.
            max_age_days: Only consider memories older than this many days.
            dry_run: If True, return preview without modifying anything.

        Returns:
            List of consolidation results (one dict per cluster).
        """
        import re as _re
        from collections import defaultdict as _defaultdict

        now = time.time()
        age_cutoff = now - (max_age_days * 86400.0) if max_age_days else None

        # ── Collect eligible memories ──
        eligible: Dict[int, Dict] = {}
        for mid, mdata in self.store.memories.items():
            if mdata.get('consolidated', False):
                continue
            # Skip memories already consolidated into something
            meta = mdata.get('metadata', '{}')
            if isinstance(meta, str):
                meta = json.loads(meta)
            if meta.get('consolidated_into'):
                continue
            if age_cutoff and mdata['created_at'] > age_cutoff:
                continue
            eligible[mid] = mdata

        if not eligible:
            return []

        # ── Build concept → memory mapping ──
        concept_to_mids: Dict[str, Set[int]] = _defaultdict(set)
        mid_to_concepts: Dict[int, Set[str]] = _defaultdict(set)
        for mid in eligible:
            concepts = self.concept_graph.get_memory_concepts(mid)
            for c in concepts:
                concept_to_mids[c].add(mid)
                mid_to_concepts[mid].add(c)

        # ── Group by concept overlap (2+ shared concepts) OR all pairs for small sets ──
        candidate_pairs: Set[frozenset] = set()
        eligible_ids = list(eligible.keys())
        for i, mid in enumerate(eligible_ids):
            for j in range(i + 1, len(eligible_ids)):
                other_mid = eligible_ids[j]
                # Concept overlap check
                shared = mid_to_concepts.get(mid, set()) & mid_to_concepts.get(other_mid, set())
                if len(shared) >= 2:
                    candidate_pairs.add(frozenset([mid, other_mid]))
                    continue
                # Fallback: if either has few/no concepts, still consider as candidate
                # (will be filtered by Jaccard similarity below)
                if len(mid_to_concepts.get(mid, set())) < 2 or len(mid_to_concepts.get(other_mid, set())) < 2:
                    candidate_pairs.add(frozenset([mid, other_mid]))

        # ── Jaccard similarity on word shingles ──
        def _tokenize_words(text: str) -> Set[str]:
            return set(_re.findall(r'[a-zA-Z0-9]+', text.lower()))

        token_cache: Dict[int, Set[str]] = {}
        def _get_tokens(mid: int) -> Set[str]:
            if mid not in token_cache:
                token_cache[mid] = _tokenize_words(eligible[mid]['content'])
            return token_cache[mid]

        def _jaccard_sim(a: Set[str], b: Set[str]) -> float:
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        # Build adjacency for similar pairs
        adjacency: Dict[int, Set[int]] = _defaultdict(set)
        for pair in candidate_pairs:
            a, b = pair
            sim = _jaccard_sim(_get_tokens(a), _get_tokens(b))
            if sim >= similarity_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # ── Greedy clustering (connected components) ──
        visited: Set[int] = set()
        clusters: List[List[int]] = []
        for mid in adjacency:
            if mid in visited:
                continue
            # BFS
            cluster = []
            queue = [mid]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) >= min_cluster_size:
                clusters.append(sorted(cluster))

        if not clusters:
            return []

        # ── Consolidate each cluster ──
        results = []
        for cluster_ids in clusters:
            # Gather data
            contents = []
            all_concepts: Set[str] = set()
            max_strength = 0.0
            for mid in cluster_ids:
                mdata = eligible[mid]
                contents.append(mdata['content'])
                all_concepts.update(mid_to_concepts.get(mid, set()))
                max_strength = max(max_strength, mdata['strength'])

            # Merge unique sentences/facts
            seen_sentences: Set[str] = set()
            unique_parts: List[str] = []
            for content in contents:
                # Split on sentence boundaries
                sentences = _re.split(r'(?<=[.!?])\s+', content.strip())
                for sent in sentences:
                    normalized = sent.strip().lower()
                    if normalized and normalized not in seen_sentences:
                        seen_sentences.add(normalized)
                        unique_parts.append(sent.strip())

            summary = ". ".join(unique_parts)
            if summary and not summary.endswith('.'):
                summary += '.'
            boosted_strength = min(max_strength * 1.2, 2.0)

            result_entry = {
                'source_ids': cluster_ids,
                'source_count': len(cluster_ids),
                'summary': summary,
                'concepts': sorted(all_concepts),
                'strength': boosted_strength,
            }

            if dry_run:
                results.append(result_entry)
                continue

            # Create consolidated memory
            consolidated_metadata = {
                'source_ids': cluster_ids,
                'source_count': len(cluster_ids),
                'consolidated_at': now,
            }
            memory_data = {
                'content': summary,
                'memory_type': 'consolidated',
                'memory_level': 'pattern',
                'strength': boosted_strength,
                'access_count': 0,
                'created_at': now,
                'last_accessed': now,
                'metadata': json.dumps(consolidated_metadata),
                'consolidated': False,
                'summary_of': json.dumps(cluster_ids),
            }
            new_id = self.store.insert_memory(memory_data)

            # Index the consolidated memory
            self.inverted_index.add_document(new_id, summary)
            self.temporal_index.add_memory(new_id, now)
            for concept_name in all_concepts:
                category = self.concept_graph.concepts[concept_name].category if concept_name in self.concept_graph.concepts else 'general'
                self.concept_graph.link_memory_concept(new_id, concept_name, category)

            # Mark originals
            for mid in cluster_ids:
                mdata = self.store.memories[mid]
                meta = mdata.get('metadata', '{}')
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta['consolidated_into'] = new_id
                self.store.update_memory(mid, {'metadata': json.dumps(meta)})

            result_entry['consolidated_id'] = new_id
            results.append(result_entry)

        return results
    
    def _find_or_create_episode(self, episode_name: str, timestamp: float) -> int:
        """Find existing episode or create a new one."""
        # For explicit episode names, always group together regardless of time
        # (different from V1 which had temporal windows for auto episodes)
        for episode_id, episode_data in self.store.episodes.items():
            if episode_data['name'] == episode_name:
                # Update episode time range
                episode_data['started_at'] = min(episode_data['started_at'], timestamp)
                episode_data['ended_at'] = max(episode_data['ended_at'], timestamp)
                return episode_id
        
        # Create new episode
        episode_data = {
            'name': episode_name,
            'started_at': timestamp,
            'ended_at': timestamp,
            'memory_ids': []
        }
        episode_id = self.store.insert_episode(episode_data)
        self.episode_index.add_episode(episode_id, episode_name, timestamp, timestamp)
        
        return episode_id
    
    # ════════════════════════════════════════════════════════════
    #  Temporal Fact Chains
    # ════════════════════════════════════════════════════════════

    def history(self, memory_id: int) -> List[Dict[str, Any]]:
        """Return the full temporal chain for a memory, oldest first.

        Each entry is ``{"memory": Memory, "version": int, "current": bool}``.
        """
        # Walk backwards to find the chain root
        root_id = memory_id
        visited = {root_id}
        while True:
            mdata = self.store.memories.get(root_id)
            if mdata is None:
                break
            meta = json.loads(mdata.get('metadata', '{}'))
            prev_id = meta.get('supersedes')  # this memory supersedes prev_id
            if prev_id is None:
                # Check if something supersedes root_id (root_id is the OLD one)
                # Actually, "supersedes" in metadata means this memory replaced prev_id.
                # So walk via supersedes to find the oldest.
                break
            if prev_id in visited:
                break
            visited.add(prev_id)
            root_id = prev_id

        # Also check: maybe memory_id is itself superseded (i.e., it's old).
        # Walk backwards via incoming supersedes edges.
        # Re-walk: start from memory_id, follow "supersedes" metadata backwards
        # AND follow "superseded_by" metadata forwards to find chain ends.
        # Simpler: walk backwards from memory_id using the supersedes edge graph.
        
        # Walk backwards via reverse supersedes edges (incoming supersedes = I am superseded by someone)
        root_id = memory_id
        visited = {root_id}
        while True:
            mdata = self.store.memories.get(root_id)
            if mdata is None:
                break
            meta = json.loads(mdata.get('metadata', '{}'))
            supersedes_id = meta.get('supersedes')
            if supersedes_id is not None and supersedes_id not in visited:
                visited.add(supersedes_id)
                root_id = supersedes_id
            else:
                break

        # Now walk forward from root via superseded_by
        chain: List[Dict[str, Any]] = []
        current_id = root_id
        version = 1
        seen = set()
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            mdata = self.store.memories.get(current_id)
            if mdata is None:
                break
            meta = json.loads(mdata.get('metadata', '{}'))
            memory = self._memory_data_to_object(mdata)
            chain.append({
                "memory": memory,
                "version": version,
                "current": meta.get('superseded_by') is None,
            })
            version += 1
            current_id = meta.get('superseded_by')

        return chain

    def fact_history(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Find the best matching memory and return its full temporal chain.

        Args:
            query: Search query to find the relevant fact.
            limit: How many top matches to build chains for (default 1).

        Returns:
            Ordered list of chain entries (oldest → newest).
        """
        matches = self.recall(query, limit=limit)
        if not matches:
            return []
        return self.history(matches[0].id)

    def timeline(self, concept: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a timeline of fact changes, optionally filtered by concept.

        Each entry: ``{"memory": Memory, "supersedes": int|None, "timestamp": float}``.
        """
        results: List[Dict[str, Any]] = []

        for mid, mdata in self.store.memories.items():
            meta = json.loads(mdata.get('metadata', '{}'))
            if meta.get('supersedes') is None and meta.get('superseded_by') is None:
                continue  # not part of any chain

            if concept is not None:
                mem_concepts = self.concept_graph.get_memory_concepts(mid)
                if concept.lower() not in {c.lower() for c in mem_concepts}:
                    continue

            memory = self._memory_data_to_object(mdata)
            results.append({
                "memory": memory,
                "supersedes": meta.get('supersedes'),
                "superseded_by": meta.get('superseded_by'),
                "fact_chain_id": meta.get('fact_chain_id'),
                "timestamp": mdata['created_at'],
            })

        results.sort(key=lambda x: x['timestamp'])
        return results

    @staticmethod
    def _parse_temporal_arg(temporal: str) -> Optional[float]:
        """Parse a temporal argument like '2024-03' or '2024-06-15' into a timestamp."""
        import datetime
        for fmt in ("%Y-%m-%d", "%Y-%m"):
            try:
                dt = datetime.datetime.strptime(temporal, fmt)
                # For month-only, use end of month
                if fmt == "%Y-%m":
                    if dt.month == 12:
                        dt = dt.replace(year=dt.year + 1, month=1)
                    else:
                        dt = dt.replace(month=dt.month + 1)
                return dt.timestamp()
            except ValueError:
                continue
        # Try as float timestamp
        try:
            return float(temporal)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _resolve_chain_at(chain: List[Dict[str, Any]], at_ts: float) -> Optional['Memory']:
        """Given a chain and a point-in-time, return the version that was current then."""
        # Find the latest version created at or before at_ts
        best = None
        for entry in chain:
            if entry["memory"].created_at <= at_ts:
                best = entry["memory"]
        return best

    def flush(self):
        """Force write to disk."""
        self.store.flush()
        self._pending_commits = 0
    
    def close(self):
        """Close the database."""
        self.flush()
        self.store.close()
    
    def snapshot(self):
        """Create compacted snapshot."""
        self.store.create_snapshot()
    
    # ════════════════════════════════════════════════════════════
    #  Portable Format (Phase 2) — export / load / merge
    # ════════════════════════════════════════════════════════════

    def export(self, path: str, *,
               since: Optional[str] = None,
               until: Optional[str] = None,
               concepts: Optional[List[str]] = None,
               tags: Optional[List[str]] = None,
               memory_types: Optional[List[str]] = None,
               source_agent: str = "unknown") -> str:
        """Export this Synapse instance to a .synapse portable file.

        Examples:
            s.export("my_agent.synapse")
            s.export("my_agent.synapse", since="2024-01-01")
            s.export("my_agent.synapse", concepts=["food", "travel"])
        """
        from portable import export_synapse
        return export_synapse(self, path, since=since, until=until,
                              concepts=concepts, tags=tags,
                              memory_types=memory_types,
                              source_agent=source_agent)

    def load(self, path: str, *, deduplicate: bool = True,
             similarity_threshold: float = 0.85) -> Dict[str, int]:
        """Import a .synapse file into this instance.

        Example:
            s = Synapse()
            s.load("my_agent.synapse")
        """
        from portable import import_synapse
        return import_synapse(self, path, deduplicate=deduplicate,
                              similarity_threshold=similarity_threshold)

    def merge(self, path: str, *,
              conflict_resolution: str = "newer_wins",
              similarity_threshold: float = 0.85) -> Dict[str, int]:
        """Merge a .synapse file into this instance with deduplication.

        Example:
            s.merge("other_agent.synapse")
        """
        from portable import merge_synapse
        return merge_synapse(self, path, conflict_resolution=conflict_resolution,
                             similarity_threshold=similarity_threshold)

    # ════════════════════════════════════════════════════════════
    #  Federation (Phase 3) — serve / push / pull / sync / peers
    # ════════════════════════════════════════════════════════════

    def serve(
        self,
        port: int = 9470,
        host: str = "127.0.0.1",
        node_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        expose_network: bool = False,
    ):
        """Start a federation HTTP server for peer-to-peer memory sync.

        Example:
            s.serve(port=9470)

        Security:
            By default, Synapse AI Memory binds federation to localhost only. To
            bind to a non-loopback interface (including `0.0.0.0`), you must
            explicitly opt in with `expose_network=True`.
        """
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        if expose_network and host in loopback_hosts:
            host = "0.0.0.0"
        if not expose_network and host not in loopback_hosts:
            raise ValueError(
                "Refusing to bind federation server to a non-loopback host by default. "
                "Pass expose_network=True to explicitly opt in."
            )
        from federation.node import SynapseNode
        nid = node_id or f"synapse-{id(self)}"
        self._federation_node = SynapseNode(node_id=nid, synapse=self,
                                            auth_token=auth_token)
        # Seed the federated store from our local memories
        self._sync_to_federation()
        self._federation_node.listen(port=port, host=host)
        return self._federation_node

    def push(self, peer_url: str, namespaces: Optional[set] = None) -> Dict[str, int]:
        """Push memories to a remote peer.

        Example:
            s.push("http://peer:9470")
        """
        self._ensure_federation()
        self._sync_to_federation()
        return self._federation_node.push(peer_url, namespaces)

    def pull(self, peer_url: str, namespaces: Optional[set] = None) -> Dict[str, int]:
        """Pull memories from a remote peer.

        Example:
            s.pull("http://peer:9470")
        """
        self._ensure_federation()
        result = self._federation_node.pull(peer_url, namespaces)
        self._sync_from_federation()
        return result

    def sync(self, peer_url: str, namespaces: Optional[set] = None) -> Dict[str, int]:
        """Bidirectional sync with a remote peer.

        Example:
            s.sync("http://peer:9470")
        """
        self._ensure_federation()
        self._sync_to_federation()
        result = self._federation_node.sync(peer_url, namespaces)
        self._sync_from_federation()
        return result

    def add_peer(self, url: str, token: Optional[str] = None):
        """Add a known federation peer.

        Example:
            s.add_peer("http://peer:9470", token="secret")
        """
        self._ensure_federation()
        self._federation_node.add_peer(url, token)

    def share(self, namespace: str = "public"):
        """Mark a namespace as shared with federation peers.

        Example:
            s.share("public")
        """
        self._ensure_federation()
        self._federation_node.share(namespace)

    def _ensure_federation(self):
        """Lazily initialize the federation node."""
        if not hasattr(self, '_federation_node') or self._federation_node is None:
            from federation.node import SynapseNode
            nid = f"synapse-{id(self)}"
            self._federation_node = SynapseNode(node_id=nid, synapse=self)

    def _sync_to_federation(self):
        """Push local memories into the federated store."""
        from federation.memory_object import FederatedMemory
        from federation.vector_clock import VectorClock
        node = self._federation_node
        for mid, mdata in self.store.memories.items():
            content = mdata['content']
            mtype = mdata.get('memory_type', 'fact')
            meta = mdata.get('metadata', '{}')
            if isinstance(meta, str):
                meta = json.loads(meta)
            fm = FederatedMemory(
                content=content,
                memory_type=mtype,
                metadata=meta,
                created_at=mdata.get('created_at', time.time()),
                origin_node=node.node_id,
                vclock=VectorClock().increment(node.node_id),
            )
            if fm.hash not in node.store.memories:
                node.store.add_memory(fm)

    def _sync_from_federation(self):
        """Import new federated memories into local store."""
        node = self._federation_node
        local_contents = {m['content'] for m in self.store.memories.values()}
        for fm in node.store.memories.values():
            if fm.content not in local_contents:
                self.remember(fm.content, memory_type=fm.memory_type,
                              metadata=fm.metadata, deduplicate=True)
                local_contents.add(fm.content)

    def count(self) -> int:
        """Return total count of active (non-consolidated) memories."""
        return sum(1 for m in self.store.memories.values()
                   if not m.get('consolidated', False))

    def list(self, limit: int = 50, offset: int = 0,
             sort: str = "recent") -> List[Memory]:
        """List memories without a query.

        Args:
            limit: Max memories to return.
            offset: Number of memories to skip.
            sort: Sort order — ``"recent"`` (last accessed),
                ``"created"`` (creation time), or ``"access_count"``.

        Returns:
            List of ``Memory`` objects.
        """
        active = [
            self._memory_data_to_object(m)
            for m in self.store.memories.values()
            if not m.get('consolidated', False)
        ]

        sort_keys = {
            "recent": lambda m: m.last_accessed,
            "created": lambda m: m.created_at,
            "access_count": lambda m: m.access_count,
        }
        key_fn = sort_keys.get(sort, sort_keys["recent"])
        active.sort(key=key_fn, reverse=True)
        return active[offset:offset + limit]

    def browse(self, concept: str, limit: int = 50,
               offset: int = 0) -> List[Memory]:
        """Browse memories linked to a specific concept.

        Args:
            concept: Concept name to filter by.
            limit: Max results.
            offset: Skip count.

        Returns:
            List of ``Memory`` objects tagged with *concept*.
        """
        concept_node = self.concept_graph.concepts.get(concept)
        if concept_node is None:
            return []

        memory_ids = sorted(concept_node.memory_ids, reverse=True)
        selected_ids = memory_ids[offset:offset + limit]

        results = []
        for mid in selected_ids:
            if mid in self.store.memories:
                mdata = self.store.memories[mid]
                if not mdata.get('consolidated', False):
                    results.append(self._memory_data_to_object(mdata))
        return results

    def concepts(self) -> List[Dict[str, Any]]:
        """Return all concepts with their memory counts."""
        result = []
        for concept_name, concept_node in self.concept_graph.concepts.items():
            result.append({
                "name": concept_name,
                "category": concept_node.category,
                "memory_count": len(concept_node.memory_ids),
                "created_at": concept_node.created_at
            })
        
        # Sort by memory count (most used concepts first)
        result.sort(key=lambda x: x["memory_count"], reverse=True)
        return result

    def hot_concepts(self, k: int = 10) -> List[tuple[str, float]]:
        """Return the top-k most active concepts.

        Returns:
            List of (concept_name, activation_strength) tuples.
        """
        now = time.time()
        scored: List[tuple[str, float]] = []
        for name in self.concept_graph.concepts.keys():
            strength = self.concept_graph.concept_activation_strength(name, now=now)
            if strength > 0:
                scored.append((name, strength))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def prune(self, *, min_strength: float = 0.1, min_access: int = 0,
              max_age_days: float = 90, dry_run: bool = True) -> List[int]:
        """Prune (forget) weak, old, rarely accessed memories whose concepts are cold.

        Criteria (all must be true):
          - effective_strength < min_strength
          - access_count <= min_access
          - age_days >= max_age_days
          - max concept activation strength for memory's concepts < 0.1

        Returns:
            List of pruned memory IDs (or would-be pruned IDs if dry_run).
        """
        now = time.time()
        max_age_secs = float(max_age_days) * 86400.0
        concept_activation_threshold = 0.1

        to_prune: List[int] = []
        for mid in list(self.store.memories.keys()):
            mdata = self.store.memories.get(mid)
            if not mdata or mdata.get('consolidated', False):
                continue

            memory = self._memory_data_to_object(mdata)
            if now - memory.created_at < max_age_secs:
                continue
            if memory.access_count > min_access:
                continue
            if memory.effective_strength >= min_strength:
                continue

            concepts = self.concept_graph.get_memory_concepts(mid)
            if concepts:
                concept_strength = max(
                    self.concept_graph.concept_activation_strength(c, now=now)
                    for c in concepts
                )
            else:
                concept_strength = 0.0

            if concept_strength >= concept_activation_threshold:
                continue

            to_prune.append(mid)

        if dry_run:
            return sorted(to_prune)

        pruned: List[int] = []
        for mid in to_prune:
            if self.forget(mid):
                pruned.append(mid)
        return sorted(pruned)
