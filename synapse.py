"""
Synapse V2 — A native memory database engine.
Pure Python, zero external dependencies, no SQL.

Based on the original Synapse but completely rewritten with:
- Append-only log + snapshots for persistence
- Native Python indexes for BM25 + concept matching  
- Two-stage recall algorithm ported from SQL
"""

import json
import math
import time
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any
from pathlib import Path

from storage import MemoryStore
from indexes import (
    InvertedIndex, ConceptGraph, EdgeGraph, TemporalIndex, EpisodeIndex,
    STOPWORDS
)
from entity_graph import extract_concepts, expand_query
from episode_graph import close_stale_episodes, find_or_create_episode, get_episode_siblings
from embeddings import EmbeddingIndex
from extractor import extract_facts


# Constants (same as V1)
MEMORY_TYPES = {"fact", "event", "preference", "skill", "observation"}
EDGE_TYPES = {"caused_by", "contradicts", "reminds_of", "supports", "preceded", "followed", "supersedes", "related"}
DECAY_HALF_LIFE = 86400 * 7  # 7 days in seconds
REINFORCE_BOOST = 0.05
DEFAULT_EPISODE_WINDOW_SECS = 1800.0


@dataclass
class Memory:
    """Memory data structure."""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "fact"
    strength: float = 1.0
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    metadata: dict = field(default_factory=dict)
    consolidated: bool = False
    summary_of: list = field(default_factory=list)

    @property
    def effective_strength(self):
        """Strength after temporal decay and access boost."""
        age = time.time() - self.last_accessed
        decay = math.pow(0.5, age / DECAY_HALF_LIFE)
        # Access count provides a logarithmic boost
        access_boost = 1 + 0.1 * math.log1p(self.access_count)
        return self.strength * decay * access_boost


class Synapse:
    """The memory engine - pure Python, zero external dependencies."""
    
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
    
    def remember(self, content: str, memory_type: str = "fact", 
                 links: Optional[List] = None, metadata: Optional[Dict] = None,
                 episode: Optional[str] = None, deduplicate: bool = True, 
                 extract: bool = False) -> Memory:
        """Store a new memory with optional fact extraction and deduplication."""
        now = time.time()
        
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
            
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of {MEMORY_TYPES}")
        
        # Handle fact extraction
        if extract:
            return self._remember_with_extraction(content, memory_type, links, metadata, episode, deduplicate, now)
        
        # Use regular single content storage
        return self._remember_single_content(content, memory_type, links, metadata, episode, deduplicate, now)
        
        return memory
    
    def _remember_with_extraction(self, content: str, memory_type: str, 
                                  links: Optional[List], metadata: Optional[Dict],
                                  episode: Optional[str], deduplicate: bool, now: float) -> Memory:
        """Handle memory storage with fact extraction."""
        try:
            # Extract facts using LLM
            facts = extract_facts(content)
            
            if not facts:
                # If extraction fails or returns no facts, fall back to normal storage
                print(f"Warning: No facts extracted from content, storing as-is")
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
            # If extraction fails entirely, fall back to normal storage
            print(f"Warning: Fact extraction failed ({e}), storing content as-is")
            return self._remember_single_content(content, memory_type, links, metadata, episode, deduplicate, now)
    
    def _remember_single_content(self, content: str, memory_type: str, 
                                 links: Optional[List], metadata: Optional[Dict],
                                 episode: Optional[str], deduplicate: bool, now: float) -> Memory:
        """Store a single piece of content as a memory (extracted from original remember logic)."""
        # Check for duplicates if requested
        similar_memories = []
        if deduplicate:
            similar_memories = self._find_similar_memories(content)
            if similar_memories:
                # Create supersedes edges to similar memories
                pass  # Will implement after creating the memory
        
        # Create memory data
        memory_data = {
            'content': content,
            'memory_type': memory_type,
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
        
        # Handle supersession for similar memories
        if deduplicate and similar_memories:
            for similar_memory in similar_memories:
                self.link(memory_id, similar_memory.id, "supersedes", 1.0)
        
        # Periodic maintenance
        self._pending_commits += 1
        if self._pending_commits >= self._commit_batch_size:
            self.flush()
        
        # Return Memory object
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
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
               temporal_boost: bool = True) -> List[Memory]:
        """
        Two-Stage Recall with adaptive blending (ported from V1):
          Stage 1A: BM25 word-index candidates  
          Stage 1B: Concept-IDF candidates
          Stage 2: Normalize, blend, apply effective strength, edge/episode expansion
        """
        now = time.time()
        
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
        
        # Two-signal fusion: normalize BM25 and embedding to [0,1], then blend
        blended = {}
        
        # Normalize BM25 scores to [0,1]
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
        if bm25_max <= 0: bm25_max = 1.0
        
        # Determine embedding weight based on query characteristics
        # Short queries benefit more from embeddings; long specific queries favor BM25
        if embedding_scores:
            if n_tokens <= 3:
                emb_weight = 0.4   # short query: embeddings matter more
            elif n_tokens <= 6:
                emb_weight = 0.3
            else:
                emb_weight = 0.2   # long query: BM25 dominates
            bm_weight = 1.0 - emb_weight
        else:
            bm_weight = 1.0
            emb_weight = 0.0
        
        for memory_id in all_ids:
            bm = bm25_scores.get(memory_id, 0.0) / bm25_max  # normalized 0-1
            cn = concept_scores.get(memory_id, 0.0) / c_max   # normalized 0-1
            emb = embedding_scores.get(memory_id, 0.0)         # already 0-1 (cosine)
            
            # Fuse: weighted combination of BM25 and embedding
            score = bm * bm_weight + emb * emb_weight + cn * concept_weight * 0.1
            
            if score > 0:
                blended[memory_id] = score
        
        if not blended:
            return []
        
        # Load memory objects and apply effective strength
        candidates = []
        for memory_id in blended:
            if memory_id in self.store.memories:
                memory_data = self.store.memories[memory_id]
                memory = self._memory_data_to_object(memory_data)
                if memory.effective_strength >= min_strength:
                    # Pure BM25+concept score — effective strength only for filtering
                    final_score = blended[memory_id]
                    candidates.append((memory, final_score))
        
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:limit * 2]  # Get more for expansion
        
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
                
            final_results.append((memory, score))
        
        # Sort and limit
        final_results.sort(key=lambda x: x[1], reverse=True)
        result_memories = [memory for memory, _ in final_results[:limit]]
        
        # Reinforce accessed memories
        self._reinforce_memories([m.id for m in result_memories])
        
        return result_memories
    
    def _memory_data_to_object(self, memory_data: Dict) -> Memory:
        """Convert stored memory data to Memory object."""
        return Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            memory_type=memory_data['memory_type'],
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
    
    def link(self, source_id: int, target_id: int, edge_type: str, weight: float = 1.0):
        """Create a link between two memories."""
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Invalid edge_type: {edge_type}. Must be one of {EDGE_TYPES}")
            
        if source_id not in self.store.memories or target_id not in self.store.memories:
            raise ValueError("Both source and target memories must exist")
        
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
    
    def consolidate(self, threshold: float = 0.85, max_group: int = 5):
        """Consolidate similar memories (placeholder implementation)."""
        # This would implement the consolidation algorithm
        # For now, just a stub
        pass
    
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