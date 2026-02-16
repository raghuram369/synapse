"""Context pack model and compiler for Synapse recall results.

This module is intentionally lightweight and zero-LLM: it builds a compact
structured context package with summaries and evidence chains suitable for prompt
construction.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set, Tuple

from entity_graph import extract_concepts


_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass
class ContextPack:
    query: str
    memories: List[Dict[str, Any]]
    graph_slice: Dict[str, Any]
    summaries: List[str]
    evidence: List[Dict[str, Any]]
    budget_used: int
    budget_total: int
    metadata: Dict[str, Any]

    def to_system_prompt(self) -> str:
        """Render as a system-prompt style section for LLM context injection."""
        lines = [
            "System Context Pack",
            "= " * 10,
            f"query: {self.query}",
            f"budget: {self.budget_used}/{self.budget_total}",
            "",
            "Summaries:",
        ]

        if self.summaries:
            lines.extend(f"- {item}" for item in self.summaries)
        else:
            lines.append("- none")

        lines.append("\nTop Memories:")
        if self.memories:
            for memory in self.memories:
                score = memory.get("score")
                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                lines.append(
                    f"- [{memory.get('id')}] ({score_text}) {memory.get('content', '')[:140]}"
                )
        else:
            lines.append("- none")

        concept_names = [
            concept["name"] for concept in self.graph_slice.get("concepts", []) if isinstance(concept, dict)
        ]
        lines.append("\nGraph Concepts:")
        if concept_names:
            lines.append("- " + ", ".join(sorted(concept_names)))
        else:
            lines.append("- none")

        lines.append("\nEvidence:")
        if self.evidence:
            for item in self.evidence:
                if isinstance(item, dict) and isinstance(item.get("claim"), str):
                    supporting = item.get("supporting_memories") or []
                    contradicting = item.get("contradicting_memories") or []
                    confidence = item.get("confidence")
                    if isinstance(confidence, (int, float)):
                        confidence_text = f"{confidence:.2f}"
                    else:
                        confidence_text = "n/a"
                    lines.append(
                        f"- {item['claim']} | support: {supporting} | "
                        f"contradicts: {contradicting} | confidence: {confidence_text}"
                    )
                else:
                    lines.append(
                        f"- [{item.get('source_id')}] {item.get('relation')} [{item.get('target_id')}]"
                    )
        else:
            lines.append("- none")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return full JSON-serializable representation."""
        return {
            "query": self.query,
            "memories": self.memories,
            "graph_slice": self.graph_slice,
            "summaries": self.summaries,
            "evidence": self.evidence,
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "metadata": self.metadata,
        }

    def to_compact(self) -> str:
        """Render a compact plain-text representation within budget."""
        lines = [
            f"Query: {self.query}",
            f"Budget: {self.budget_used}/{self.budget_total}",
        ]

        if self.summaries:
            lines.append("Summaries: " + " | ".join(self.summaries[:3]))

        if self.memories:
            rendered_memories = []
            for memory in self.memories:
                score = memory.get("score")
                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                rendered_memories.append(
                    f"{memory.get('id')}:{score_text}:{memory.get('content', '')[:80]}"
                )
            lines.append("Memories: " + " | ".join(rendered_memories[:4]))

        concept_names = [
            concept["name"] for concept in self.graph_slice.get("concepts", []) if isinstance(concept, dict)
        ]
        if concept_names:
            lines.append("Concepts: " + ", ".join(sorted(concept_names[:10])))

        if self.evidence:
            evidence_lines = [
                (
                    f"{item.get('claim')}::{item.get('supporting_memories', [])}"
                    if isinstance(item, dict) and isinstance(item.get("claim"), str)
                    else f"{item.get('source_id')}->{item.get('relation')}->{item.get('target_id')}"
                )
                for item in self.evidence[:3]
            ]
            lines.append("Evidence: " + ", ".join(evidence_lines))

        compact = " | ".join(lines)
        if self.budget_total <= 0:
            return ""
        if len(compact) <= self.budget_total:
            return compact
        if self.budget_total == 1:
            return "…"
        return compact[: self.budget_total - 1] + "…"


class ContextCompiler:
    """Compile a context pack from Synapse recall and graph structures."""

    _POLICIES = {
        "balanced": {
            "limit": 10,
            "min_strength": 0.01,
            "temporal_boost": True,
            "graph_depth": 2,
            "temporal": False,
        },
        "precise": {
            "limit": 4,
            "min_strength": 0.08,
            "temporal_boost": False,
            "graph_depth": 1,
            "temporal": False,
        },
        "broad": {
            "limit": 20,
            "min_strength": 0.0,
            "temporal_boost": False,
            "graph_depth": 3,
            "temporal": False,
        },
        "temporal": {
            "limit": 12,
            "min_strength": 0.01,
            "temporal_boost": True,
            "graph_depth": 2,
            "temporal": True,
        },
    }

    def __init__(self, synapse_instance):
        self.synapse = synapse_instance

    def compile_context(
        self,
        query: str,
        budget: int = 4000,
        policy: str = "balanced",
    ) -> ContextPack:
        policy_name = (policy or "balanced").lower()
        if policy_name not in self._POLICIES:
            policy_name = "balanced"
        cfg = self._POLICIES.get(policy_name, self._POLICIES["balanced"])

        timings: Dict[str, float] = {}
        total_start = time.perf_counter()

        recall_start = time.perf_counter()
        recalled = self.synapse.recall(
            context=query,
            limit=cfg["limit"],
            min_strength=cfg["min_strength"],
            temporal_boost=cfg["temporal_boost"],
            explain=True,
        )
        timings["recall_ms"] = (time.perf_counter() - recall_start) * 1000.0

        if cfg["temporal"]:
            recalled = sorted(
                [m for m in recalled],
                key=lambda item: (item.created_at, item.id or 0),
                reverse=True,
            )

        memory_records = [self._memory_record(memory) for memory in recalled]
        memory_records = [memory for memory in memory_records if memory.get("id") is not None]

        ids = [memory["id"] for memory in memory_records]

        graph_start = time.perf_counter()
        graph_slice = self._extract_graph_slice(ids, depth=cfg["graph_depth"])
        timings["graph_ms"] = (time.perf_counter() - graph_start) * 1000.0

        concept_names = {item["name"] for item in graph_slice.get("concepts", []) if isinstance(item, dict)}

        summary_start = time.perf_counter()
        summaries = self._generate_summaries(memory_records, concept_names)
        community_summaries = self._generate_community_summaries(
            concept_names,
        )
        summaries.extend(community_summaries)
        timings["summaries_ms"] = (time.perf_counter() - summary_start) * 1000.0

        evidence_start = time.perf_counter()
        evidence = self._build_evidence_chains(recalled)
        timings["evidence_ms"] = (time.perf_counter() - evidence_start) * 1000.0

        pack = self._pack_to_budget(
            query=query,
            memories=memory_records,
            graph_slice=graph_slice,
            summaries=summaries,
            evidence=evidence,
            budget=budget,
            policy=policy_name,
        )

        pack.metadata.update(
            {
                "policy": policy_name,
                "requested_budget": budget,
                "timing_ms": {
                    "recall": timings["recall_ms"],
                    "graph": timings["graph_ms"],
                    "summaries": timings["summaries_ms"],
                    "evidence": timings["evidence_ms"],
                    "total": (time.perf_counter() - total_start) * 1000.0,
                },
                "stats": {
                    "retrieved": len(memory_records),
                    "selected_memories": len(pack.memories),
                    "graph_nodes": len(graph_slice.get("nodes", [])),
                    "graph_edges": len(graph_slice.get("edges", [])),
                    "concept_count": len(graph_slice.get("concepts", [])),
                    "evidence_count": len(pack.evidence),
                },
            }
        )

        pack.budget_used = len(pack.to_compact())
        pack.budget_total = budget
        return pack

    def _memory_record(self, memory) -> Dict[str, Any]:
        score_breakdown = asdict(memory.score_breakdown) if getattr(memory, "score_breakdown", None) else {}
        return {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type,
            "memory_level": memory.memory_level,
            "strength": memory.strength,
            "created_at": memory.created_at,
            "last_accessed": memory.last_accessed,
            "metadata": memory.metadata,
            "consolidated": memory.consolidated,
            "score": round(self._score_from_breakdown(score_breakdown), 4),
            "score_breakdown": score_breakdown,
            "concepts": [name for name, _ in extract_concepts(memory.content)],
        }

    def _score_from_breakdown(self, score_breakdown: Dict[str, Any]) -> float:
        if not score_breakdown:
            return 0.0
        return float(
            0.3 * score_breakdown.get("bm25_score", 0.0)
            + 0.25 * score_breakdown.get("concept_score", 0.0)
            + 0.2 * score_breakdown.get("temporal_score", 0.0)
            + 0.1 * score_breakdown.get("episode_score", 0.0)
            + 0.1 * score_breakdown.get("concept_activation_score", 0.0)
            + 0.05 * score_breakdown.get("embedding_score", 0.0)
        )

    def _extract_graph_slice(self, memory_ids: List[int], depth: int = 2) -> Dict[str, Any]:
        if not memory_ids:
            return {
                "nodes": [],
                "edges": [],
                "concepts": [],
                "seed_memory_ids": [],
                "depth": depth,
            }

        active_ids = [memory_id for memory_id in memory_ids if memory_id in self.synapse.store.memories]
        if not active_ids:
            return {
                "nodes": [],
                "edges": [],
                "concepts": [],
                "seed_memory_ids": [],
                "depth": depth,
            }

        frontier = set(active_ids)
        visited = set(active_ids)
        seen_edges: Set[Tuple[int, int, str]] = set()
        edges: List[Dict[str, Any]] = []

        for level in range(max(0, int(depth))):
            next_frontier = set()
            for source_id in list(frontier):
                for target_id, edge in self.synapse.edge_graph.get_all_edges(source_id):
                    if target_id not in self.synapse.store.memories:
                        continue

                    edge_key = (source_id, target_id, edge.edge_type)
                    if edge_key not in seen_edges:
                        edges.append(
                            {
                                "source_id": source_id,
                                "target_id": target_id,
                                "relation": edge.edge_type,
                                "weight": edge.weight,
                                "level": level,
                            }
                        )
                        seen_edges.add(edge_key)

                    if target_id not in visited:
                        visited.add(target_id)
                        next_frontier.add(target_id)

            frontier = next_frontier
            if not frontier:
                break

        nodes: List[Dict[str, Any]] = []
        concept_to_memory_ids: Dict[str, List[int]] = defaultdict(list)

        for memory_id in sorted(visited):
            memory_data = self.synapse.store.memories[memory_id]
            concepts = sorted(self.synapse.concept_graph.get_memory_concepts(memory_id))
            nodes.append(
                {
                    "id": memory_id,
                    "content": memory_data.get("content", "")[:160],
                    "memory_type": memory_data.get("memory_type", ""),
                    "concepts": concepts,
                    "created_at": memory_data.get("created_at", 0.0),
                }
            )
            for concept in concepts:
                concept_to_memory_ids[concept].append(memory_id)

        concepts = [
            {
                "name": concept,
                "count": len(ids),
                "memory_ids": ids,
            }
            for concept, ids in concept_to_memory_ids.items()
        ]
        concepts.sort(key=lambda item: (-item["count"], item["name"]))

        return {
            "nodes": nodes,
            "edges": edges,
            "concepts": concepts,
            "seed_memory_ids": sorted(set(active_ids)),
            "depth": depth,
        }

    def _generate_summaries(self, memories: List[Dict[str, Any]], concepts: Set[str]) -> List[str]:
        if not memories:
            return ["No memories matched this query."]

        buckets: Dict[str, List[str]] = defaultdict(list)
        loose_facts: List[str] = []

        concept_filter = set(concepts)
        for memory in memories:
            content = (memory.get("content") or "").strip()
            if not content:
                continue

            snippet = content
            if len(snippet) > 90:
                snippet = snippet[:87] + "..."

            memory_concepts = set(memory.get("concepts", []))
            years = _YEAR_RE.findall(content)
            if years:
                snippet = f"{snippet} ({years[0]})"
                memory_concepts.add("history")

            if concept_filter:
                matched = memory_concepts & concept_filter
                if years:
                    matched.add("history")
            else:
                matched = set(memory_concepts)

            if not matched:
                loose_facts.append(snippet)
                continue

            for concept in sorted(matched):
                buckets[concept].append(snippet)

        summaries: List[str] = []
        for concept, values in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
            uniq = list(dict.fromkeys(values))
            if not uniq:
                continue
            label = "History" if concept == "history" else concept.replace("_", " ").title()
            summaries.append(f"{label}: {'; '.join(uniq[:4])}")

        if loose_facts:
            summaries.append("Observed: " + "; ".join(dict.fromkeys(loose_facts)[:4]))

        if not summaries:
            return ["No stable summaries were generated from retrieved memories."]

        return summaries[:8]

    def _generate_community_summaries(
        self,
        concepts: Set[str],
    ) -> List[str]:
        if not concepts:
            return []

        summaries: List[str] = []
        seen: Set[str] = set()

        for concept in sorted(concepts):
            if not hasattr(self.synapse, "community_summary"):
                break
            summary = self.synapse.community_summary(concept)
            if not summary or summary in seen:
                continue
            if summary.startswith("No community summary available"):
                continue
            summaries.append(summary)
            seen.add(summary)
            if len(summaries) >= 3:
                break

        return summaries

    def _build_evidence_chains(self, memories: List[Any]) -> List[Dict[str, Any]]:
        memory_ids = {
            memory.id for memory in memories
            if getattr(memory, "id", None) is not None
        }
        if not memory_ids:
            return []

        compiler = self.synapse.evidence_compiler
        chains = []
        if compiler is not None:
            for chain in compiler.compile(memories, getattr(self.synapse, "triple_index", None)):
                # Keep a uniform dict shape for pack.evidence so callers can
                # always access source/target/relation without KeyError.
                payload = asdict(chain)
                payload.setdefault("source_id", None)
                payload.setdefault("target_id", None)
                payload.setdefault("relation", "claim")
                chains.append(payload)

        legacy = self._build_legacy_evidence(list(memory_ids), memories)
        return chains + legacy

    def _build_legacy_evidence(self, memory_ids: Set[int], memories: List[Any]) -> List[Dict[str, Any]]:
        id_to_memory = {memory.id: memory for memory in memories if getattr(memory, "id", None) is not None}
        evidence: List[Dict[str, Any]] = []
        seen = set()

        for source_id in memory_ids:
            for target_id, edge in self.synapse.edge_graph.get_all_edges(source_id):
                if target_id not in memory_ids or source_id == target_id:
                    continue
                key = (source_id, target_id, edge.edge_type)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation": edge.edge_type,
                        "weight": edge.weight,
                        "notes": "linked",
                    }
                )

        for memory_id in memory_ids:
            memory = id_to_memory.get(memory_id)
            if memory is None:
                continue
            metadata = getattr(memory, "metadata", {}) or {}
            for relation in ("supersedes", "superseded_by"):
                peer_id = metadata.get(relation)
                if isinstance(peer_id, int) and peer_id in memory_ids:
                    evidence.append(
                        {
                            "source_id": memory_id,
                            "target_id": peer_id,
                            "relation": relation,
                            "weight": 1.0,
                            "notes": "temporal_chain",
                        }
                    )

        evidence.sort(
            key=lambda item: (
                item.get("source_id") or 0,
                item.get("relation") or "",
                item.get("target_id") or 0,
            )
        )
        return evidence

    def _pack_to_budget(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        graph_slice: Dict[str, Any],
        summaries: List[str],
        evidence: List[Dict[str, Any]],
        budget: int,
        policy: str = "balanced",
    ) -> ContextPack:
        if budget <= 0:
            return ContextPack(
                query=query,
                memories=[],
                graph_slice={"nodes": [], "edges": [], "concepts": []},
                summaries=[],
                evidence=[],
                budget_used=0,
                budget_total=0,
                metadata={"selected_by_budget": True},
            )

        remaining = budget

        selected_summaries: List[str] = []
        for summary in summaries:
            if not summary:
                continue
            if len(summary) <= remaining:
                selected_summaries.append(summary)
                remaining -= len(summary) + 2
            elif not selected_summaries:
                selected_summaries.append(summary[: max(1, remaining - 1)] + "…")
                remaining = 0
                break

        selected_memories: List[Dict[str, Any]] = []
        ordered = self._order_memories(memories=memories, policy=policy)
        for memory in ordered:
            if remaining <= 0:
                break
            snippet = f"{memory.get('id')}:{memory.get('content', '')[:90]}"
            if len(snippet) <= remaining:
                selected_memories.append(memory)
                remaining -= len(snippet) + 2
            elif not selected_memories:
                selected_memory = dict(memory)
                selected_memory["content"] = snippet[: max(1, remaining - 1)]
                selected_memories.append(selected_memory)
                remaining = 0
                break

        selected_graph = {
            "nodes": [],
            "edges": [],
            "concepts": [],
            "seed_memory_ids": graph_slice.get("seed_memory_ids", []),
            "depth": graph_slice.get("depth"),
        }

        for node in graph_slice.get("nodes", []):
            if remaining <= 0:
                break
            node_snippet = f"{node.get('id')}:{','.join(node.get('concepts', []))}"
            if len(node_snippet) <= remaining:
                selected_graph["nodes"].append(node)
                remaining -= len(node_snippet) + 2
            elif not selected_graph["nodes"]:
                selected_graph["nodes"].append(
                    {
                        "id": node.get("id"),
                        "content": node.get("content", "")[: max(1, remaining - 1)],
                        "memory_type": node.get("memory_type", ""),
                        "concepts": node.get("concepts", []),
                        "created_at": node.get("created_at", 0.0),
                    }
                )
                remaining = 0
                break

        for edge in graph_slice.get("edges", []):
            if remaining <= 0:
                break
            edge_snippet = f"{edge.get('source_id')}->{edge.get('target_id')}:{edge.get('relation')}"
            if len(edge_snippet) <= remaining:
                selected_graph["edges"].append(edge)
                remaining -= len(edge_snippet) + 2
            elif not selected_graph["edges"]:
                remaining = 0
                break

        # Rebuild concept -> memory_ids from the selected nodes so the shape is stable.
        concept_to_memory_ids: Dict[str, Set[int]] = {}
        for node in selected_graph.get("nodes", []):
            memory_id = node.get("id")
            if memory_id is None:
                continue
            try:
                memory_id_int = int(memory_id)
            except (TypeError, ValueError):
                continue
            for concept in node.get("concepts", []) or []:
                if not concept:
                    continue
                concept_to_memory_ids.setdefault(concept, set()).add(memory_id_int)

        selected_graph["concepts"] = [
            {
                "name": concept,
                "count": len(memory_ids),
                "memory_ids": sorted(memory_ids),
            }
            for concept, memory_ids in concept_to_memory_ids.items()
        ]
        selected_graph["concepts"].sort(key=lambda item: (-item["count"], item["name"]))

        selected_evidence: List[Dict[str, Any]] = []
        for item in evidence:
            if remaining <= 0:
                break
            if isinstance(item, dict) and isinstance(item.get("claim"), str):
                supporting = ",".join(str(mid) for mid in item.get("supporting_memories", []))
                contradicting = ",".join(str(mid) for mid in item.get("contradicting_memories", []))
                confidence = item.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidence_text = f"{confidence:.2f}"
                else:
                    confidence_text = "n/a"
                evidence_snippet = (
                    f"{item.get('claim')}|s:{supporting}|c:{contradicting}|conf:{confidence_text}"
                )
            else:
                evidence_snippet = (
                    f"{item.get('source_id')}->{item.get('relation')}->{item.get('target_id')}"
                )
            if len(evidence_snippet) <= remaining:
                selected_evidence.append(item)
                remaining -= len(evidence_snippet) + 2
            elif not selected_evidence:
                break

        return ContextPack(
            query=query,
            memories=selected_memories,
            graph_slice=selected_graph,
            summaries=selected_summaries,
            evidence=selected_evidence,
            budget_used=0,
            budget_total=budget,
            metadata={"selected_by_budget": True},
        )

    @staticmethod
    def _order_memories(memories: List[Dict[str, Any]], policy: str) -> List[Dict[str, Any]]:
        if policy == "temporal":
            return sorted(
                memories,
                key=lambda item: item.get("created_at") if item.get("created_at") is not None else 0.0,
            )
        return sorted(
            memories,
            key=lambda memory: (
                -(memory.get("score", 0.0) or 0.0),
                -(memory.get("created_at") or 0.0),
                memory.get("id") or 0,
            ),
        )
