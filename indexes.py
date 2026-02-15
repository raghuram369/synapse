"""
Native in-memory indexes for Synapse V2.
All indexes are rebuilt on startup from storage.
Pure Python, zero external dependencies.
"""

import math
import re
import time
from typing import Dict, Set, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import bisect


# Tokenization constants
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "it",
    "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "and",
    "or", "but", "not", "no", "if", "then", "so",
    "what", "how", "much", "about", "tell", "which", "where", "when", "who",
    "running", "using", "going", "looking",
    "am", "get", "got", "set", "just", "also", "being",
    "like", "need", "want", "know", "think",
    "use", "used", "any", "some", "there", "here",
    "very", "really", "quite", "more", "most", "other", "each",
    "every", "all", "both", "own", "same", "than", "too",
}


@dataclass
class ConceptNode:
    """Node in the concept graph."""
    name: str
    category: str
    memory_ids: Set[int] = field(default_factory=set)
    created_at: float = 0.0


@dataclass 
class Edge:
    """Graph edge between memories."""
    target_id: int
    edge_type: str  
    weight: float
    created_at: float


@dataclass
class Episode:
    """Episode containing related memories."""
    id: int
    name: str
    started_at: float
    ended_at: float
    memory_ids: Set[int] = field(default_factory=set)


class InvertedIndex:
    """
    Inverted index for BM25 scoring.
    Structure: word → {memory_id: term_frequency}
    """
    
    def __init__(self):
        # word -> {memory_id: tf}
        self.index: Dict[str, Dict[int, float]] = defaultdict(dict)
        # memory_id -> total_terms (document length)
        self.doc_lengths: Dict[int, int] = {}
        # total documents
        self.total_docs = 0
        # average document length
        self.avg_doc_length = 0.0
        
    def tokenize_for_index(self, text: str) -> List[str]:
        """Tokenize text for indexing (includes all words)."""
        # Split on non-alphanumeric, lowercase, keep all tokens
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens
    
    def tokenize_for_query(self, text: str) -> List[str]:
        """Tokenize text for querying (removes stopwords)."""
        tokens = self.tokenize_for_index(text)
        return [t for t in tokens if t not in STOPWORDS]
    
    def add_document(self, memory_id: int, content: str):
        """Add or update a document in the index."""
        # Remove existing document if present
        self.remove_document(memory_id)
        
        tokens = self.tokenize_for_index(content)
        if not tokens:
            return
            
        # Count term frequencies
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
            
        # Add to index
        for term, count in term_counts.items():
            self.index[term][memory_id] = float(count)
            
        # Update document length
        self.doc_lengths[memory_id] = len(tokens)
        self.total_docs = len(self.doc_lengths)
        
        # Update average document length
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
    
    def remove_document(self, memory_id: int):
        """Remove a document from the index."""
        if memory_id not in self.doc_lengths:
            return
            
        # Remove from inverted index
        terms_to_clean = []
        for term, doc_dict in self.index.items():
            if memory_id in doc_dict:
                del doc_dict[memory_id]
                if not doc_dict:  # Empty term list
                    terms_to_clean.append(term)
        
        # Clean up empty terms
        for term in terms_to_clean:
            del self.index[term]
            
        # Remove document length
        del self.doc_lengths[memory_id]
        self.total_docs = len(self.doc_lengths)
        
        # Update average document length
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0.0
    
    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term."""
        return len(self.index.get(term, {}))
    
    def get_term_frequency(self, term: str, memory_id: int) -> float:
        """Get term frequency for a term in a document."""
        return self.index.get(term, {}).get(memory_id, 0.0)
    
    def bm25_score(self, query_tokens: List[str], memory_id: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 score for a document against query tokens."""
        if memory_id not in self.doc_lengths:
            return 0.0
            
        score = 0.0
        doc_length = self.doc_lengths[memory_id]
        
        for token in query_tokens:
            tf = self.get_term_frequency(token, memory_id)
            if tf == 0:
                continue
                
            df = self.get_document_frequency(token)
            if df == 0:
                continue
                
            # IDF component with small corpus protection
            # Use max to ensure positive IDF even for common terms in small corpora
            idf_raw = (self.total_docs - df + 0.5) / (df + 0.5)
            idf = math.log(max(1.1, idf_raw))  # minimum IDF = log(1.1) ≈ 0.095
            
            # TF component with document length normalization
            if self.avg_doc_length > 0:
                normalized_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length)))
            else:
                normalized_tf = tf  # fallback for edge cases
            
            score += idf * normalized_tf
            
        return max(0.0, score)  # Ensure non-negative
    
    def get_candidates(self, query_tokens: List[str], limit: int = 1000) -> Dict[int, float]:
        """Get candidate documents with BM25 scores."""
        if not query_tokens:
            return {}
            
        # Get all documents that contain at least one query term
        candidates = set()
        for token in query_tokens:
            if token in self.index:
                candidates.update(self.index[token].keys())
        
        if not candidates:
            return {}
            
        # Score all candidates
        scores = {}
        for memory_id in candidates:
            score = self.bm25_score(query_tokens, memory_id)
            if score > 0:
                scores[memory_id] = score
        
        # Return top candidates
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_candidates[:limit])


class ConceptGraph:
    """
    Concept graph for semantic matching.
    Structure: concept_name -> ConceptNode
    """
    
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.memory_concepts: Dict[int, Set[str]] = defaultdict(set)  # memory_id -> concept_names
    
    def add_concept(self, name: str, category: str = "general") -> ConceptNode:
        """Add or get a concept."""
        if name not in self.concepts:
            self.concepts[name] = ConceptNode(
                name=name,
                category=category,
                created_at=time.time()
            )
        return self.concepts[name]
    
    def link_memory_concept(self, memory_id: int, concept_name: str, category: str = "general"):
        """Link a memory to a concept."""
        concept = self.add_concept(concept_name, category)
        concept.memory_ids.add(memory_id)
        self.memory_concepts[memory_id].add(concept_name)
    
    def unlink_memory(self, memory_id: int):
        """Remove all concept links for a memory."""
        if memory_id in self.memory_concepts:
            for concept_name in self.memory_concepts[memory_id]:
                if concept_name in self.concepts:
                    self.concepts[concept_name].memory_ids.discard(memory_id)
            del self.memory_concepts[memory_id]
    
    def get_memory_concepts(self, memory_id: int) -> Set[str]:
        """Get all concepts for a memory."""
        return self.memory_concepts.get(memory_id, set())
    
    def get_concept_memories(self, concept_name: str) -> Set[int]:
        """Get all memories for a concept."""
        if concept_name in self.concepts:
            return self.concepts[concept_name].memory_ids
        return set()
    
    def concept_idf_score(self, query_concepts: List[str], memory_id: int, total_memories: int) -> float:
        """Compute concept-IDF score for a memory."""
        if not query_concepts or memory_id not in self.memory_concepts:
            return 0.0
            
        memory_concepts = self.memory_concepts[memory_id]
        if not memory_concepts:
            return 0.0
            
        score = 0.0
        for concept in query_concepts:
            if concept in memory_concepts and concept in self.concepts:
                # Document frequency = number of memories with this concept
                df = len(self.concepts[concept].memory_ids)
                if df > 0:
                    # IDF score
                    idf = math.log(total_memories / df)
                    score += idf
                    
        return score
    
    def get_candidates(self, query_concepts: List[str], limit: int = 1000, total_memories: Optional[int] = None) -> Dict[int, float]:
        """Get candidate memories with concept-IDF scores."""
        if not query_concepts:
            return {}
            
        candidates = set()
        for concept in query_concepts:
            if concept in self.concepts:
                candidates.update(self.concepts[concept].memory_ids)
        
        if not candidates:
            return {}
            
        # Use provided total or fall back to concept-having memories
        if total_memories is None:
            total_memories = len(self.memory_concepts)
            
        if total_memories == 0:
            return {}
        
        # Score candidates
        scores = {}
        for memory_id in candidates:
            score = self.concept_idf_score(query_concepts, memory_id, total_memories)
            if score > 0:
                scores[memory_id] = score
        
        # Return top candidates
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_candidates[:limit])


class EdgeGraph:
    """
    Graph of relationships between memories.
    Structure: memory_id -> list[Edge]
    """
    
    def __init__(self):
        self.edges: Dict[int, List[Edge]] = defaultdict(list)
        self.reverse_edges: Dict[int, List[Edge]] = defaultdict(list)  # target -> sources
        self.edge_counter = 0
    
    def add_edge(self, source_id: int, target_id: int, edge_type: str, weight: float = 1.0):
        """Add an edge between memories."""
        edge = Edge(
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            created_at=time.time()
        )
        
        self.edges[source_id].append(edge)
        
        # Add reverse edge for bidirectional lookup
        reverse_edge = Edge(
            target_id=source_id,
            edge_type=edge_type,  # Could use reverse types if needed
            weight=weight,
            created_at=time.time()
        )
        self.reverse_edges[target_id].append(reverse_edge)
    
    def remove_memory_edges(self, memory_id: int):
        """Remove all edges involving a memory."""
        # Remove outgoing edges
        if memory_id in self.edges:
            del self.edges[memory_id]
            
        # Remove incoming edges
        if memory_id in self.reverse_edges:
            del self.reverse_edges[memory_id]
            
        # Remove from other memories' edge lists
        for source_id, edge_list in self.edges.items():
            self.edges[source_id] = [e for e in edge_list if e.target_id != memory_id]
            
        for target_id, edge_list in self.reverse_edges.items():
            self.reverse_edges[target_id] = [e for e in edge_list if e.target_id != memory_id]
    
    def get_outgoing_edges(self, memory_id: int) -> List[Edge]:
        """Get all outgoing edges from a memory."""
        return self.edges.get(memory_id, [])
    
    def get_incoming_edges(self, memory_id: int) -> List[Edge]:
        """Get all incoming edges to a memory."""
        return self.reverse_edges.get(memory_id, [])
    
    def get_all_edges(self, memory_id: int) -> List[Tuple[int, Edge]]:
        """Get all edges (both directions) for a memory."""
        result = []
        
        # Outgoing edges: (memory_id -> target)
        for edge in self.get_outgoing_edges(memory_id):
            result.append((edge.target_id, edge))
            
        # Incoming edges: (source -> memory_id)
        for edge in self.get_incoming_edges(memory_id):
            result.append((edge.target_id, edge))  # target_id is actually source in reverse edges
            
        return result
    
    def get_superseded_memories(self, candidate_ids: Set[int]) -> Set[int]:
        """Get memories that are superseded by others in the candidate set."""
        superseded = set()
        
        for memory_id in candidate_ids:
            for edge in self.get_incoming_edges(memory_id):
                if edge.edge_type == "supersedes" and edge.target_id in candidate_ids:
                    superseded.add(memory_id)
                    
        return superseded


class TemporalIndex:
    """
    Temporal index for time-based queries.
    Structure: sorted list of (created_at, memory_id) tuples
    """
    
    def __init__(self):
        self.time_index: List[Tuple[float, int]] = []  # (timestamp, memory_id)
        self.memory_times: Dict[int, float] = {}  # memory_id -> timestamp
    
    def add_memory(self, memory_id: int, created_at: float):
        """Add a memory to the temporal index."""
        if memory_id in self.memory_times:
            self.remove_memory(memory_id)
            
        # Insert in sorted order
        bisect.insort(self.time_index, (created_at, memory_id))
        self.memory_times[memory_id] = created_at
    
    def remove_memory(self, memory_id: int):
        """Remove a memory from the temporal index."""
        if memory_id not in self.memory_times:
            return
            
        created_at = self.memory_times[memory_id]
        self.time_index.remove((created_at, memory_id))
        del self.memory_times[memory_id]
    
    def get_memories_in_range(self, start_time: float, end_time: float) -> List[int]:
        """Get memories created within a time range."""
        if start_time > end_time:
            return []
            
        # Find insertion points for time range
        start_idx = bisect.bisect_left(self.time_index, (start_time, 0))
        end_idx = bisect.bisect_right(self.time_index, (end_time, float('inf')))
        
        return [memory_id for _, memory_id in self.time_index[start_idx:end_idx]]
    
    def get_memories_around_time(self, target_time: float, window_seconds: float) -> List[int]:
        """Get memories within a time window around a target time."""
        start_time = target_time - window_seconds
        end_time = target_time + window_seconds
        return self.get_memories_in_range(start_time, end_time)


class EpisodeIndex:
    """
    Episode index for grouping related memories.
    Structure: episode_id -> Episode, memory_id -> episode_id
    """
    
    def __init__(self):
        self.episodes: Dict[int, Episode] = {}
        self.memory_episodes: Dict[int, int] = {}  # memory_id -> episode_id
    
    def add_episode(self, episode_id: int, name: str, started_at: float, ended_at: float):
        """Add an episode."""
        self.episodes[episode_id] = Episode(
            id=episode_id,
            name=name,
            started_at=started_at,
            ended_at=ended_at
        )
    
    def add_memory_to_episode(self, memory_id: int, episode_id: int):
        """Add a memory to an episode."""
        if episode_id in self.episodes:
            self.episodes[episode_id].memory_ids.add(memory_id)
            self.memory_episodes[memory_id] = episode_id
    
    def remove_memory(self, memory_id: int):
        """Remove a memory from its episode."""
        if memory_id in self.memory_episodes:
            episode_id = self.memory_episodes[memory_id]
            if episode_id in self.episodes:
                self.episodes[episode_id].memory_ids.discard(memory_id)
            del self.memory_episodes[memory_id]
    
    def get_episode_siblings(self, memory_id: int) -> List[int]:
        """Get all memories in the same episode as the given memory."""
        if memory_id not in self.memory_episodes:
            return []
            
        episode_id = self.memory_episodes[memory_id]
        if episode_id not in self.episodes:
            return []
            
        return list(self.episodes[episode_id].memory_ids)
    
    def get_memory_episode(self, memory_id: int) -> Optional[int]:
        """Get the episode ID for a memory."""
        return self.memory_episodes.get(memory_id)