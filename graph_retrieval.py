"""Graph-based retrieval primitives used by graph recall mode."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from entity_graph import expand_query, extract_concepts


class GraphRetriever:
    """Retrieve memories by combining lexical BM25-like matching with graph spread."""

    _RELATED_CONCEPTS: Dict[str, List[str]] = {
        # Very small cross-domain bridges used to make graph recall useful even
        # when the local memory graph is sparse.
        "ai_ml": ["runtime"],
    }

    def __init__(self, concept_graph, triple_index=None):
        """Create a retriever that spreads activation on concept graph hops."""
        self.concept_graph = concept_graph
        self.triple_index = triple_index

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "it", "that", "this", "these", "those", "we",
            "our", "you", "your", "i", "me", "my", "he", "she", "they", "them",
            "and", "or", "but", "not", "no", "if", "then", "so", "what", "how",
            "which", "where", "when", "who",
        }
        return [t for t in tokens if t not in stopwords]

    def _memory_records(self, memories: List) -> Dict[int, str]:
        records: Dict[int, str] = {}
        for memory in memories:
            if memory is None:
                continue

            if isinstance(memory, dict):
                memory_id = memory.get("id")
                content = memory.get("content", "")
            else:
                memory_id = getattr(memory, "id", None)
                content = getattr(memory, "content", "")

            if memory_id is None:
                continue
            try:
                memory_id = int(memory_id)
            except (TypeError, ValueError):
                continue

            if content is None:
                content = ""
            records[memory_id] = str(content)

        return records

    def _extract_query_terms(self, query: str) -> List[str]:
        terms: List[str] = []
        seen: Set[str] = set()
        for term in self._tokenize(query):
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    def extract_query_concepts(self, query: str) -> List[str]:
        """Extract seed concepts from query text using current concept extract/expand."""
        extracted = [name for name, _ in extract_concepts(query or "")]

        expanded_from_tokens = expand_query(self._extract_query_terms(query))
        expanded_from_concepts = expand_query(extracted)

        seen: Set[str] = set()
        concepts: List[str] = []
        for item in extracted + expanded_from_tokens + expanded_from_concepts:
            if item and item not in seen:
                seen.add(item)
                concepts.append(item)

        # Apply minimal hand-authored bridges (kept intentionally small to avoid
        # turning graph mode into generic semantic recall).
        for item in list(concepts):
            for related in self._RELATED_CONCEPTS.get(item, []):
                if related and related not in seen:
                    seen.add(related)
                    concepts.append(related)
        return concepts

    def _memory_concepts(self, memory_id: int) -> Set[str]:
        method = getattr(self.concept_graph, "get_memory_concepts", None)
        if callable(method):
            return set(method(memory_id))
        return set()

    def _concept_memories(self, concept: str) -> Set[int]:
        method = getattr(self.concept_graph, "get_concept_memories", None)
        if callable(method):
            return set(method(concept))
        return set()

    def _co_occurrence_neighbors(self, concept: str) -> Dict[str, float]:
        if not concept:
            return {}
        neighbors: Dict[str, float] = defaultdict(float)

        concept_node = getattr(self.concept_graph, "concepts", {}).get(concept)
        if concept_node is None:
            return neighbors

        for memory_id in getattr(concept_node, "memory_ids", set()):
            for other in self._memory_concepts(memory_id):
                if other == concept:
                    continue
                neighbors[other] += 1.0

        return dict(neighbors)

    def _normalize_triple_neighbors(self, raw_neighbors: Any) -> List[Tuple[str, str, float]]:
        neighbors: List[Tuple[str, str, float]] = []
        if raw_neighbors is None:
            return neighbors

        def add_entry(raw_neighbor, raw_edge_type="factual", raw_weight=1.0):
            if raw_neighbor is None:
                return
            neighbor = str(raw_neighbor).strip().lower()
            if not neighbor:
                return
            edge_type = str(raw_edge_type) if raw_edge_type else "factual"
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = 1.0
            if weight <= 0:
                return
            neighbors.append((neighbor, edge_type, weight))

        if isinstance(raw_neighbors, dict):
            for neighbor, value in raw_neighbors.items():
                if isinstance(value, dict):
                    add_entry(
                        neighbor,
                        value.get("edge_type", value.get("type", value.get("relation", "factual"))),
                        value.get("weight", 1.0),
                    )
                elif isinstance(value, (int, float)):
                    add_entry(neighbor, "factual", value)
                elif isinstance(value, str):
                    add_entry(neighbor, value, 1.0)
                elif isinstance(value, (list, tuple)) and value:
                    # Allow dict-like: {'A': [('B', 'factual', 1.2)]}
                    normalized = self._normalize_triple_neighbors(value)
                    neighbors.extend(normalized)
                else:
                    add_entry(neighbor, "factual", 1.0)
            return neighbors

        if isinstance(raw_neighbors, (list, tuple, set)):
            for entry in raw_neighbors:
                if isinstance(entry, str):
                    add_entry(entry)
                    continue
                if not isinstance(entry, (list, tuple)):
                    continue
                if len(entry) == 0:
                    continue
                if len(entry) == 1:
                    add_entry(entry[0])
                    continue

                candidate = str(entry[0]).strip().lower()
                if not candidate:
                    continue

                second = entry[1]
                third = entry[2] if len(entry) > 2 else 1.0

                if isinstance(second, str):
                    if len(entry) > 2 and isinstance(third, (int, float)):
                        add_entry(candidate, second, third)
                    else:
                        add_entry(candidate, second, 1.0)
                elif isinstance(second, (int, float)):
                    if len(entry) > 2:
                        if isinstance(third, str):
                            add_entry(candidate, third, second)
                        else:
                            add_entry(candidate, "factual", second)
                    else:
                        add_entry(candidate, "factual", second)
                elif isinstance(second, dict):
                    add_entry(candidate, second.get("edge_type", second.get("type", "factual")), second.get("weight", 1.0))
                else:
                    add_entry(candidate)
            return neighbors

        if isinstance(raw_neighbors, str):
            add_entry(raw_neighbors)
            return neighbors

        return neighbors

    def _triple_neighbors(self, concept: str) -> List[Tuple[str, str, float]]:
        if self.triple_index is None or not concept:
            return []

        raw = None
        for method_name in (
            "get_neighbors",
            "get_connected_concepts",
            "neighbors",
            "related",
            "concept_neighbors",
            "get_neighbors_for",
        ):
            method = getattr(self.triple_index, method_name, None)
            if callable(method):
                raw = method(concept)
                break

        if raw is None:
            # Support direct index-like mapping access.
            raw = getattr(self.triple_index, "edges", {}).get(concept)

        neighbors: List[Tuple[str, str, float]] = []
        for neighbor, edge_type, weight in self._normalize_triple_neighbors(raw):
            if neighbor == concept:
                continue
            neighbors.append((neighbor, edge_type, weight))

        return neighbors

    def _neighbors(self, concept: str, edge_type_weights: Dict[str, float]) -> List[Tuple[str, float]]:
        weights = dict(edge_type_weights or {})
        default_weight = 1.0
        results: Dict[str, float] = defaultdict(float)

        # Co-occurrence neighbors from memory concept co-membership.
        co_weight = weights.get("co_occurrence", 1.0)
        for concept_name, count in self._co_occurrence_neighbors(concept).items():
            results[concept_name] += co_weight * float(count)

        # Triple index neighbors with relation weights.
        for target, edge_type, weight in self._triple_neighbors(concept):
            edge_weight = weights.get(edge_type, default_weight)
            results[target] += float(edge_weight) * float(weight)

        return [(neighbor, score) for neighbor, score in results.items()]

    def multi_hop_spread(
        self,
        seed_concepts: List[str],
        max_hops: int = 3,
        decay: float = 0.5,
        edge_type_weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Hop-weighted activation spreading from seed concepts.

        Returns concept -> activation_score mapping.
        """
        if not seed_concepts or max_hops < 0 or decay <= 0:
            return {}

        weights = {
            "co_occurrence": 1.0,
            "factual": 1.5,
            "derived": 0.5,
        }
        if edge_type_weights:
            weights.update({k: float(v) for k, v in edge_type_weights.items() if v is not None})

        seeds = set(str(c).strip().lower() for c in seed_concepts if c)
        if not seeds:
            return {}

        # Use a "best activation" propagation (max, not sum) to avoid cycles
        # inflating scores (e.g. alpha -> beta -> alpha).
        scores: Dict[str, float] = {seed: 1.0 for seed in seeds}
        frontier: Dict[str, float] = dict(scores)

        for _hop in range(max_hops):
            if not frontier:
                break
            next_frontier: Dict[str, float] = {}
            for concept, score in frontier.items():
                for neighbor, neighbor_weight in self._neighbors(concept, weights):
                    if not neighbor or neighbor == concept:
                        continue
                    propagated = score * decay * float(neighbor_weight)
                    if propagated <= 0:
                        continue

                    # Only keep an improvement; decay means first/best path wins in practice.
                    if propagated > scores.get(neighbor, 0.0):
                        scores[neighbor] = propagated
                        next_frontier[neighbor] = propagated
            frontier = next_frontier

        return dict(scores)

    def multi_hop_concept_expansion(
        self,
        seed_concepts: List[str],
        max_hops: int = 3,
        decay: float = 0.5,
        edge_type_weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Backward-compatible alias for multi_hop_spread."""
        return self.multi_hop_spread(seed_concepts, max_hops=max_hops, decay=decay, edge_type_weights=edge_type_weights)

    def dual_path_retrieve(
        self,
        query: str,
        memories: List,
        limit: int = 10,
        bm25_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> List[tuple]:
        """
        Two entry points:
        Path 1: BM25/lexical → candidate memories → extract their concepts → spread
        Path 2: Query → extract concepts → multi-hop spread → find memories with activated concepts
        Merge results with configurable weights.
        """
        if limit <= 0:
            return []

        records = self._memory_records(memories)
        if not records:
            return []

        query = query or ""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Path 1: lexical candidates from raw content.
        path1_base: Dict[int, float] = {}
        path1_graph: Dict[str, float] = {}
        for memory_id, content in records.items():
            content_tokens = set(self._tokenize(content))
            overlap = len(set(query_tokens) & content_tokens)
            if overlap <= 0:
                continue
            path1_base[memory_id] = float(overlap)

        if path1_base:
            top_path1 = sorted(path1_base.items(), key=lambda item: item[1], reverse=True)[: max(1, limit * 2)]
            seed_concepts: Set[str] = set()
            for memory_id, _ in top_path1:
                seed_concepts |= self._memory_concepts(memory_id)
            path1_graph = self.multi_hop_spread(list(seed_concepts), max_hops=3, decay=0.5)

            for memory_id in list(path1_base.keys()):
                concepts = self._memory_concepts(memory_id)
                if concepts:
                    path1_base[memory_id] += sum(path1_graph.get(c, 0.0) for c in concepts if c in path1_graph)
            
            for concept, concept_score in path1_graph.items():
                if concept_score <= 0:
                    continue
                for memory_id in self._concept_memories(concept):
                    if memory_id not in records or concept not in self._memory_concepts(memory_id):
                        continue
                    path1_base[memory_id] = path1_base.get(memory_id, 0.0) + concept_score

        query_seed_concepts = self.extract_query_concepts(query)

        if not query_seed_concepts:
            path2_scores = {}
        else:
            # Path 2: query concepts with graph spread.
            path2_graph = self.multi_hop_spread(query_seed_concepts, max_hops=3, decay=0.5)
            path2_scores: Dict[int, float] = {}
            if path2_graph:
                for concept, score in path2_graph.items():
                    if score <= 0:
                        continue
                    for memory_id in self._concept_memories(concept):
                        if memory_id not in records:
                            continue
                        memory_concepts = self._memory_concepts(memory_id)
                        if concept in memory_concepts:
                            path2_scores[memory_id] = (
                                path2_scores.get(memory_id, 0.0) + score
                            )
            else:
                path2_scores = {}

        # Merge two paths with configurable weights.
        if not path1_base and not path2_scores:
            return []

        def norm(scores: Dict[int, float]) -> Dict[int, float]:
            if not scores:
                return {}
            max_score = max(scores.values())
            if max_score <= 0:
                return {}
            return {memory_id: score / max_score for memory_id, score in scores.items()}

        path1_norm = norm(path1_base)
        path2_norm = norm(path2_scores)

        merged: Dict[int, float] = {}
        total = set(path1_norm) | set(path2_norm)
        for memory_id in total:
            merged[memory_id] = (
                bm25_weight * path1_norm.get(memory_id, 0.0)
                + graph_weight * path2_norm.get(memory_id, 0.0)
            )

        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:limit]
        return ranked
