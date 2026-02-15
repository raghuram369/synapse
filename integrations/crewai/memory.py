"""
SynapseCrewMemory — CrewAI memory backend powered by Synapse.

Enables CrewAI agents to share persistent, semantically searchable memory.
Privacy-first: all data stays local. Federation for multi-crew systems.

Requirements:
    pip install crewai synapse-ai-memory
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from synapse import Synapse


class SynapseCrewMemory:
    """Synapse-backed memory for CrewAI agents and crews.

    Provides three memory layers matching CrewAI's architecture:
    - **Short-term**: Current task context (episode-based)
    - **Long-term**: Persistent facts and learnings
    - **Entity**: Structured knowledge about people, places, concepts

    Example::

        from synapse import Synapse
        from synapse.integrations.crewai import SynapseCrewMemory

        syn = Synapse("./crew_memory")
        memory = SynapseCrewMemory(synapse=syn)

        # Agents automatically share memory
        memory.save("User prefers Python", agent="researcher")
        results = memory.search("programming preferences", agent="writer")
    """

    def __init__(
        self,
        synapse: Optional[Synapse] = None,
        path: str = ":memory:",
        crew_id: str = "default",
    ):
        self.synapse = synapse or Synapse(path)
        self.crew_id = crew_id

    # ── Short-term memory (current task context) ──────────────

    def save_short_term(self, content: str, agent: str = "",
                        task: str = "", metadata: Optional[Dict] = None) -> None:
        """Save short-term memory for the current task episode."""
        meta = {
            "source": "crewai",
            "memory_layer": "short_term",
            "crew_id": self.crew_id,
            "agent": agent,
            "task": task,
            **(metadata or {}),
        }
        self.synapse.remember(
            content,
            memory_type="observation",
            metadata=meta,
            episode=f"crew-{self.crew_id}-{task}" if task else f"crew-{self.crew_id}",
        )

    def search_short_term(self, query: str, limit: int = 5,
                          agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search short-term memories."""
        return self._search("short_term", query, limit, agent)

    # ── Long-term memory (persistent facts) ───────────────────

    def save_long_term(self, content: str, agent: str = "",
                       metadata: Optional[Dict] = None) -> None:
        """Save a long-term memory (fact, learning, insight)."""
        meta = {
            "source": "crewai",
            "memory_layer": "long_term",
            "crew_id": self.crew_id,
            "agent": agent,
            **(metadata or {}),
        }
        self.synapse.remember(content, memory_type="fact", metadata=meta)

    def search_long_term(self, query: str, limit: int = 5,
                         agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search long-term memories."""
        return self._search("long_term", query, limit, agent)

    # ── Entity memory (structured knowledge) ──────────────────

    def save_entity(self, entity_name: str, entity_type: str,
                    description: str, agent: str = "",
                    metadata: Optional[Dict] = None) -> None:
        """Save structured entity knowledge."""
        content = f"{entity_name} ({entity_type}): {description}"
        meta = {
            "source": "crewai",
            "memory_layer": "entity",
            "crew_id": self.crew_id,
            "agent": agent,
            "entity_name": entity_name,
            "entity_type": entity_type,
            **(metadata or {}),
        }
        self.synapse.remember(content, memory_type="fact", metadata=meta)

    def search_entity(self, query: str, limit: int = 5,
                      entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search entity memories, optionally filtered by type."""
        results = self._search("entity", query, limit)
        if entity_type:
            results = [r for r in results if r.get("entity_type") == entity_type]
        return results

    # ── Unified interface (CrewAI compatibility) ──────────────

    def save(self, content: str, agent: str = "", task: str = "",
             metadata: Optional[Dict] = None) -> None:
        """Save memory (auto-routes to long-term storage)."""
        self.save_long_term(content, agent=agent, metadata=metadata)

    def search(self, query: str, limit: int = 5,
               agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search across all memory layers."""
        results = self.synapse.recall(query, limit=limit)

        filtered = []
        for mem in results:
            meta = mem.metadata or {}
            if meta.get("source") != "crewai":
                continue
            if meta.get("crew_id") != self.crew_id:
                continue
            if agent and meta.get("agent") != agent:
                continue

            filtered.append({
                "content": mem.content,
                "agent": meta.get("agent", ""),
                "memory_layer": meta.get("memory_layer", ""),
                "strength": mem.effective_strength,
                "created_at": mem.created_at,
                "metadata": meta,
            })

        return filtered[:limit]

    def reset(self) -> None:
        """Clear all crew memory."""
        all_memories = self.synapse.recall("", limit=10000)
        for mem in all_memories:
            meta = mem.metadata or {}
            if meta.get("source") == "crewai" and meta.get("crew_id") == self.crew_id:
                self.synapse.forget(mem.id)

    # ── Federation: Multi-crew memory sharing ─────────────────

    def share_with_crew(self, peer_url: str) -> Dict[str, int]:
        """Push this crew's memories to another Synapse instance.

        Enables multi-crew collaboration with federated memory.
        """
        return self.synapse.push(peer_url)

    def learn_from_crew(self, peer_url: str) -> Dict[str, int]:
        """Pull memories from another crew's Synapse instance."""
        return self.synapse.pull(peer_url)

    def export_crew_knowledge(self, path: str) -> str:
        """Export crew knowledge to a portable .synapse file."""
        return self.synapse.export(path, source_agent=f"crew-{self.crew_id}")

    def import_crew_knowledge(self, path: str) -> Dict[str, int]:
        """Import knowledge from a .synapse file."""
        return self.synapse.load(path)

    # ── Internal ──────────────────────────────────────────────

    def _search(self, layer: str, query: str, limit: int,
                agent: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.synapse.recall(query, limit=limit * 3)
        filtered = []
        for mem in results:
            meta = mem.metadata or {}
            if meta.get("source") != "crewai":
                continue
            if meta.get("crew_id") != self.crew_id:
                continue
            if meta.get("memory_layer") != layer:
                continue
            if agent and meta.get("agent") != agent:
                continue
            filtered.append({
                "content": mem.content,
                "agent": meta.get("agent", ""),
                "strength": mem.effective_strength,
                "created_at": mem.created_at,
                **{k: v for k, v in meta.items() if k.startswith("entity_")},
            })
        return filtered[:limit]
