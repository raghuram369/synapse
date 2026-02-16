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
import re
from datetime import datetime, timezone
from collections import defaultdict
from itertools import combinations
import time
from dataclasses import dataclass, field
from context_pack import ContextCompiler, ContextPack
from context_cards import CardDeck, ContextCard
from typing import Any, Dict, List, Optional, Set
from communities import Community, CommunityDetector

from temporal import latest_facts, memories_as_of, memories_during, parse_temporal
from embeddings import EmbeddingIndex
from sleep import SleepReport, SleepRunner
from belief import BeliefTracker, BeliefVersion
from forgetting import ForgettingPolicy
from policies import PolicyManager
from evidence import EvidenceCompiler
from normalization import ENTITY_NORMALIZER
from entity_graph import extract_concepts, expand_query
from contradictions import ContradictionDetector
from exceptions import SynapseValidationError
from extractor import extract_facts
from graph_retrieval import GraphRetriever
from scope_policy import ScopePolicy
from triples import TripleIndex, extract_triples
from checkpoints import Checkpoint, CheckpointManager
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

MEMORY_TYPES: Set[str] = {
    "fact",
    "event",
    "preference",
    "skill",
    "observation",
    "consolidated",
    "semantic",
}
MEMORY_SCOPES = ("public", "shared", "private")
_SHARED_WITH_UNSET = object()
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
    scope: str = "private"
    shared_with: Optional[List[str]] = None
    sensitive: bool = False
    strength: float = 1.0
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    observed_at: Optional[float] = None
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    consolidated: bool = False
    summary_of: List[int] = field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = field(default=None, repr=False)
    disputes: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    evidence_pointers: List[Dict[str, Any]] = field(default_factory=list, repr=False)

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

    def __init__(self, path: str = ":memory:", scope_policy: Optional[ScopePolicy] = None, 
                 vault: Optional[str] = None, vault_manager: Optional[Any] = None,
                 inbox_mode: bool = False, inbox_config: Optional[Dict[str, Any]] = None):
        self.path = path
        self.scope_policy = scope_policy
        self._entity_normalizer = ENTITY_NORMALIZER
        self._vault_id = vault
        self._user_id = None
        self._vault_manager = vault_manager
        self._inbox_mode = inbox_mode
        self._inbox: Optional[Any] = None
        self._inbox_config = inbox_config
        
        # Initialize storage
        self.store = MemoryStore(path)
        self._store = self.store
        
        # Initialize indexes
        self.inverted_index = InvertedIndex()
        self.concept_graph = ConceptGraph()
        self.edge_graph = EdgeGraph()
        self.temporal_index = TemporalIndex()
        self.episode_index = EpisodeIndex()
        self.contradiction_detector = ContradictionDetector()
        self.belief_tracker = BeliefTracker(contradiction_detector=self.contradiction_detector)
        self.evidence_compiler = EvidenceCompiler()
        self.forgetting_policy = ForgettingPolicy(self)
        self.triple_index = TripleIndex()
        self.policy_manager = PolicyManager(self)
        
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

        # Sleep mode
        self._is_sleeping = False
        self._last_sleep_at: Optional[float] = None
        self.sleep_runner = SleepRunner(self)

        # Community cache
        self._community_detector = CommunityDetector()
        self._communities: Optional[List[Community]] = None
        self._communities_dirty = True
        self._community_update_pending_nodes: Set[str] = set()
        self._community_update_pending_edges: List[tuple] = []

        # Build indexes from stored data
        self._rebuild_indexes()

        # Checkpoint support
        self.checkpoints = CheckpointManager(self)

        # Persistent context card collection
        self.cards = CardDeck(cards=getattr(self.store, "cards", {}), synapse_instance=self)
        
        # Initialize inbox mode if enabled
        self._inbox_pending_dir = None
        if self._inbox_mode:
            from memory_inbox import MemoryInbox
            import tempfile
            pending_dir = None
            if self.path == ":memory:":
                pending_dir = tempfile.mkdtemp(prefix="synapse_pending_")
                self._inbox_pending_dir = pending_dir
            self._inbox = MemoryInbox(
                self,
                pending_dir=pending_dir,
                auto_approve_config=self._inbox_config,
            )

    def _route_to_vault(self, user_id: Optional[str] = None) -> Optional["Synapse"]:
        """Route to appropriate vault based on user_id."""
        if not user_id or not self._vault_manager:
            return None
        
        if self._vault_id:  # Already in a vault
            return self if self._user_id == user_id else None
        
        # Route to user's vault
        return self._vault_manager.get_user_vault(user_id)
    
    # ─── Inbox Methods ───────────────────────────────────────────────────────
    
    def inbox(self) -> Optional[Any]:
        """Get the memory inbox instance if inbox mode is enabled."""
        return self._inbox
    
    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending memories in inbox."""
        if not self._inbox:
            return []
        return self._inbox.list_pending()
    
    def approve_memory(self, item_id: str) -> Optional[Memory]:
        """Approve a pending memory from inbox."""
        if not self._inbox:
            return None
        return self._inbox.approve(item_id)
    
    def reject_memory(self, item_id: str) -> bool:
        """Reject a pending memory from inbox."""
        if not self._inbox:
            return False
        return self._inbox.reject(item_id)
    
    def redact_memory(self, item_id: str, redacted_content: str) -> Optional[Memory]:
        """Redact and approve a pending memory from inbox."""
        if not self._inbox:
            return None
        return self._inbox.redact(item_id, redacted_content)
    
    def pin_memory(self, item_id: str) -> Optional[Memory]:
        """Pin and approve a pending memory from inbox."""
        if not self._inbox:
            return None
        return self._inbox.pin(item_id)
    
    # ─── Natural Language Forget ─────────────────────────────────────────────
    
    def natural_forget(self, command: str, confirm: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """Process a natural language forget command."""
        from natural_forget import NaturalForget
        forget_processor = NaturalForget(self)
        return forget_processor.process_forget_command(command, confirm, dry_run)

    def _reload_from_checkpoint_state(self):
        """Reload all in-memory indexes and trackers from persisted storage."""
        self.store.close()
        self.store = MemoryStore(self.path)
        self._store = self.store

        self.inverted_index = InvertedIndex()
        self.concept_graph = ConceptGraph()
        self.edge_graph = EdgeGraph()
        self.temporal_index = TemporalIndex()
        self.episode_index = EpisodeIndex()
        self.triple_index = TripleIndex()
        self.contradiction_detector = ContradictionDetector()
        self.belief_tracker = BeliefTracker(contradiction_detector=self.contradiction_detector)

        self._rebuild_indexes()
        self.cards = CardDeck(cards=getattr(self.store, "cards", {}), synapse_instance=self)
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from stored data."""
        self.triple_index = TripleIndex()
        self._invalidate_communities()

        # Rebuild inverted index
        for memory_id, memory_data in self.store.memories.items():
            if not memory_data.get('consolidated', False):
                self.inverted_index.add_document(memory_id, memory_data['content'])
                self.temporal_index.add_memory(memory_id, memory_data['created_at'])
                triples = extract_triples(memory_data['content'])
                if triples:
                    self.triple_index.add(memory_id, triples)
        
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
        memories_for_belief = [
            self._memory_data_to_object(memory_data)
            for memory_id, memory_data in self.store.memories.items()
            if not memory_data.get('consolidated', False)
            and memory_id is not None
        ]
        self.belief_tracker.rebuild(sorted(memories_for_belief, key=lambda item: item.created_at))
        
    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        scope: str = "private",
        links: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        episode: Optional[str] = None,
        observed_at: Optional[Any] = None,
        valid_from: Optional[Any] = None,
        valid_to: Optional[Any] = None,
        deduplicate: bool = True,
        shared_with: Optional[List[str]] = None,
        sensitive: bool = False,
        extract: bool = False,
        user_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory with optional fact extraction and deduplication.

        Args:
            content: The text content to memorise.
            memory_type: One of ``MEMORY_TYPES``.
            scope: One of ``MEMORY_SCOPES``.
            links: Optional edges to create (list of dicts with
                ``target_id``, ``edge_type``, and optionally ``weight``).
            metadata: Arbitrary JSON-serialisable metadata.
            episode: Optional episode name to group this memory into.
            shared_with: Optional group/principal IDs for shared memories.
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
        # Route to user vault if specified
        if user_id:
            vault = self._route_to_vault(user_id)
            if vault and vault != self:
                return vault.remember(
                    content, memory_type, scope, links, metadata, episode,
                    observed_at, valid_from, valid_to, deduplicate, 
                    shared_with, sensitive, extract, user_id
                )
        
        # Use inbox mode if enabled
        if self._inbox_mode and self._inbox:
            inbox_result = self._inbox.submit(content, memory_type, metadata, scope)
            if inbox_result["status"] == "auto_approved" and inbox_result["memory_id"]:
                # Return the auto-approved memory
                memory_data = self.store.memories.get(inbox_result["memory_id"])
                if memory_data:
                    return self._memory_data_to_object(memory_data)
            # If pending, create a placeholder memory object
            elif inbox_result["status"] == "pending":
                return Memory(
                    id=None,  # No ID yet, as it's pending
                    content=content,
                    memory_type=memory_type,
                    scope=scope,
                    metadata={**(metadata or {}), "pending": True, "item_id": inbox_result["item_id"]},
                    created_at=time.time()
                )
        
        now = time.time()
        observed_ts = self._coerce_temporal_value(observed_at, default=now)
        valid_from_ts = self._coerce_temporal_value(valid_from, default=observed_ts)
        valid_to_ts = self._coerce_temporal_value(valid_to, default=None)

        if valid_to_ts is not None and valid_to_ts < valid_from_ts:
            raise SynapseValidationError("valid_to must be greater than valid_from")

        if not content.strip():
            raise SynapseValidationError("Memory content cannot be empty")
        if memory_type not in MEMORY_TYPES:
            raise SynapseValidationError(
                f"Invalid memory_type: {memory_type}. Must be one of {MEMORY_TYPES}"
            )

        memory_type = str(memory_type)
        scope = self._normalize_scope(scope)
        content, metadata, scope, sensitive = self._apply_policy_rules(
            content=content,
            metadata=metadata,
            memory_type=memory_type,
            scope=scope,
            sensitive=bool(sensitive),
        )
        normalized_shared_with = self._normalize_shared_with(shared_with)

        if extract:
            memory = self._remember_with_extraction(
                content, memory_type, scope, links, metadata, episode, deduplicate, now,
                observed_ts, valid_from_ts, valid_to_ts,
                normalized_shared_with, sensitive,
            )
        else:
            memory = self._remember_single_content(
                content, memory_type, scope, links, metadata, episode, deduplicate, now,
                observed_ts, valid_from_ts, valid_to_ts,
                normalized_shared_with, sensitive,
            )

        if getattr(self, '_is_sleeping', False):
            self._apply_policy_retention()
            return memory

        self._apply_policy_retention()

        if self.should_sleep():
            try:
                self.sleep(verbose=False)
            except Exception:
                logger.exception("Auto-sleep failed during remember()")

        return memory

    def _apply_policy_rules(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]],
        memory_type: str,
        scope: str,
        sensitive: bool = False,
    ) -> tuple[str, Optional[Dict[str, Any]], str, bool]:
        """Apply active policy hooks before storing memory content.

        Returns sanitized content, metadata, scope and sensitive flag.
        """
        policy = self.policy_manager.get_active()
        if not policy:
            return content, metadata, scope, sensitive

        metadata_dict = dict(metadata or {}) if isinstance(metadata, dict) else {}
        tags = set(self._coerce_tags(metadata_dict.get("tags")))
        resolved_scope = self._normalize_scope(scope)

        if policy.get("redact_pii"):
            pii_patterns = policy.get("pii_patterns")
            content = self.policy_manager.auto_redact(content, pii_patterns=pii_patterns)

        if policy.get("auto_tag_project"):
            for tag in self.policy_manager.detect_project_tags(content):
                tags.add(tag)

        if policy.get("keep_tags"):
            if memory_type == "preference":
                tags.add("preference")
            if self._has_commitment_language(content):
                tags.add("commitment")

        policy_scope = policy.get("scope")
        if policy_scope is not None:
            try:
                resolved_scope = self._normalize_scope(policy_scope)
            except SynapseValidationError:
                pass

        scope_by_type = policy.get("scope_by_type")
        if isinstance(scope_by_type, dict):
            type_scope = scope_by_type.get(memory_type)
            if type_scope is not None:
                try:
                    resolved_scope = self._normalize_scope(type_scope)
                except SynapseValidationError:
                    pass

        scope_by_tag = policy.get("scope_by_tag")
        if isinstance(scope_by_tag, dict) and tags:
            tag_scopes: List[str] = []
            for tag in tags:
                if tag not in scope_by_tag:
                    continue
                try:
                    tag_scopes.append(self._normalize_scope(scope_by_tag[tag]))
                except SynapseValidationError:
                    continue
            if tag_scopes:
                resolved_scope = max(tag_scopes, key=MEMORY_SCOPES.index)

        if tags:
            metadata_dict["tags"] = sorted(set(tags))

        if metadata is not None and isinstance(metadata, dict):
            metadata.clear()
            metadata.update(metadata_dict)
        elif metadata_dict:
            metadata = metadata_dict

        # Auto-flag sensitive based on policy sensitive_tags
        sensitive_tags = set(policy.get("sensitive_tags", [
            "health", "financial", "legal", "children", "immigration",
        ]))
        if tags and sensitive_tags and tags.intersection(sensitive_tags):
            sensitive = True

        return content, metadata, resolved_scope, bool(sensitive)

    def _apply_policy_retention(self) -> Dict[str, Any]:
        """Run active policy retention rules."""
        policy = self.policy_manager.get_active()
        if not policy:
            return {}

        ttl_days = policy.get("ttl_days")
        active_name = self._active_preset_name()
        report: Dict[str, Any] = {}

        if ttl_days is not None:
            try:
                ttl_days_i = int(ttl_days)
                if ttl_days_i > 0:
                    pin_tag = self._coerce_text(policy.get("pin_tag"))
                    if pin_tag:
                        now = time.time()
                        ttl_deleted: List[int] = []
                        for memory_id, memory_data in list(self.store.memories.items()):
                            metadata = self._parse_metadata(memory_data.get("metadata"))
                            tags = set(self._coerce_tags(metadata.get("tags")))
                            if pin_tag in tags:
                                continue
                            created_at = float(memory_data.get("created_at", 0.0) or 0.0)
                            if now - created_at > ttl_days_i * 86400.0 and self.forget(memory_id):
                                ttl_deleted.append(memory_id)
                        report["ttl"] = {
                            "action": "apply_ttl",
                            "default_ttl_days": float(ttl_days_i),
                            "matched_count": len(ttl_deleted),
                            "deleted_count": len(ttl_deleted),
                            "deleted_ids": sorted(ttl_deleted),
                        }
                    else:
                        report["ttl"] = self.forgetting_policy.apply_ttl(default_ttl_days=ttl_days_i)
            except (TypeError, ValueError):
                pass

        if not policy.get("auto_prune"):
            return report

        prune_min_access = policy.get("prune_min_access")
        try:
            prune_min_access_i = int(prune_min_access) if prune_min_access is not None else 0
        except (TypeError, ValueError):
            prune_min_access_i = 0

        min_strength = 0.2 if active_name == "minimal" else 0.1
        max_age_days = 90.0
        if ttl_days is not None:
            try:
                max_age_days = float(ttl_days)
            except (TypeError, ValueError):
                max_age_days = 90.0

        keep_tags = set(self._coerce_tags(policy.get("keep_tags")))
        pin_tag = self._coerce_text(policy.get("pin_tag"))

        candidate_ids = self.prune(
            min_strength=min_strength,
            min_access=prune_min_access_i,
            max_age_days=float(max_age_days),
            dry_run=True,
        )
        candidate_set = set(candidate_ids)

        if active_name == "minimal":
            now = time.time()
            for memory_id, memory_data in list(self.store.memories.items()):
                metadata = self._parse_metadata(memory_data.get("metadata"))
                tags = set(self._coerce_tags(metadata.get("tags")))
                if keep_tags.intersection(tags):
                    continue
                access_count = int(memory_data.get("access_count", 0) or 0)
                if access_count > prune_min_access_i:
                    continue
                created_at = float(memory_data.get("created_at", 0.0) or 0.0)
                age_days = (now - created_at) / 86400.0
                if age_days < max_age_days:
                    continue
                candidate_set.add(memory_id)

        pruned: List[int] = []
        for memory_id in sorted(candidate_set):
            memory_data = self.store.memories.get(memory_id)
            if not memory_data:
                continue

            metadata = self._parse_metadata(memory_data.get("metadata"))
            tags = self._coerce_tags(metadata.get("tags"))

            if pin_tag and pin_tag in tags:
                continue

            if self._active_preset_name() == "minimal" and keep_tags:
                if keep_tags.intersection(set(tags)):
                    continue

            if self.forget(memory_id):
                pruned.append(memory_id)

        if pruned:
            report["policy_prune"] = {
                "deleted_count": len(pruned),
                "deleted_ids": sorted(pruned),
            }

        return report

    @staticmethod
    def _coerce_text(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _coerce_tags(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, (list, tuple, set)):
            normalized: List[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        return []

    @staticmethod
    def _parse_metadata(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        try:
            if isinstance(value, str) and value:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
        return {}

    @staticmethod
    def _has_commitment_language(content: str) -> bool:
        if not content:
            return False
        lowered = content.lower()
        patterns = [
            " i will ",
            "I'll ",
            "i will",
            "i am going to",
            "committed",
            "commit",
            "plan to",
            "intend",
            "must",
        ]
        for token in patterns:
            if token.strip().lower() in lowered:
                return True
        return False

    def _active_preset_name(self) -> str:
        policy = self.policy_manager.get_active() or {}
        return str(policy.get("name", "").strip())

    @staticmethod
    def _normalize_scope(scope: Any, *, allow_none: bool = False) -> Optional[str]:
        """Validate and normalize a memory scope value."""
        if scope is None:
            if allow_none:
                return None
            return "private"
        normalized = str(scope).strip().lower()
        if normalized not in MEMORY_SCOPES:
            raise SynapseValidationError(
                f"Invalid scope: {scope}. Must be one of {MEMORY_SCOPES}"
            )
        return normalized

    @staticmethod
    def _memory_scope(memory_data: Dict[str, Any]) -> str:
        """Resolve persisted memory scope with backward-compatible default."""
        raw = memory_data.get("scope", "private")
        normalized = str(raw).strip().lower() if raw is not None else "private"
        return normalized if normalized in MEMORY_SCOPES else "private"

    @staticmethod
    def _is_scope_visible(memory_scope: str, requested_scope: str) -> bool:
        """Return True when a memory is visible from requested scope."""
        return MEMORY_SCOPES.index(memory_scope) <= MEMORY_SCOPES.index(requested_scope)

    @staticmethod
    def _normalize_shared_with(shared_with: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and normalize group/principal IDs for shared memory visibility."""
        if shared_with is None:
            return None
        if not isinstance(shared_with, list):
            raise SynapseValidationError("shared_with must be a list of strings or None")
        normalized: List[str] = []
        for value in shared_with:
            if not isinstance(value, str):
                raise SynapseValidationError("shared_with entries must be strings")
            item = value.strip()
            if not item:
                continue
            normalized.append(item)
        return normalized or None

    @staticmethod
    def _parse_shared_with(raw_shared_with: Any) -> Optional[List[str]]:
        """Parse persisted shared_with payload into list form."""
        if raw_shared_with is None:
            return None
        if isinstance(raw_shared_with, list):
            normalized = [
                item.strip()
                for item in raw_shared_with
                if isinstance(item, str) and item.strip()
            ]
            return normalized or None
        if isinstance(raw_shared_with, str):
            if not raw_shared_with:
                return None
            try:
                parsed = json.loads(raw_shared_with)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, list):
                return None
            normalized = [
                item.strip()
                for item in parsed
                if isinstance(item, str) and item.strip()
            ]
            return normalized or None
        return None

    def _shared_scope_match(self, memory_data: Dict[str, Any], caller_groups: Optional[List[str]]) -> bool:
        """Apply shared-with filter for shared scope requests."""
        memory_scope = self._memory_scope(memory_data)
        if memory_scope != "shared" or caller_groups is None:
            return True
        memory_groups = self._parse_shared_with(memory_data.get("shared_with"))
        if memory_groups is None:
            return True
        caller_group_set = {group for group in caller_groups if isinstance(group, str) and group.strip()}
        if not caller_group_set:
            return False
        return any(group in caller_group_set for group in memory_groups)
    
    def bulk_set_scope(
        self,
        memory_ids: List[int],
        scope: str,
        shared_with: Any = _SHARED_WITH_UNSET,
    ) -> int:
        """Update the scope of multiple memories at once. Returns count updated."""
        target = self._normalize_scope(scope)
        normalized_shared_with = (
            self._normalize_shared_with(shared_with)
            if shared_with is not _SHARED_WITH_UNSET
            else _SHARED_WITH_UNSET
        )
        updated = 0
        for mid in memory_ids:
            if mid in self.store.memories:
                updates: Dict[str, Any] = {'scope': target}
                if normalized_shared_with is not _SHARED_WITH_UNSET:
                    updates['shared_with'] = (
                        json.dumps(normalized_shared_with)
                        if normalized_shared_with is not None
                        else None
                    )
                self.store.memories[mid].update(updates)
                self.store.update_memory(mid, updates)
                updated += 1
        if updated:
            self.flush()
        return updated

    def bulk_set_sensitive(self, memory_ids: List[int], sensitive: bool = True) -> int:
        """Update the sensitive flag of multiple memories at once. Returns count updated."""
        updated = 0
        for mid in memory_ids:
            if mid in self.store.memories:
                self.store.memories[mid]['sensitive'] = bool(sensitive)
                self.store.update_memory(mid, {'sensitive': bool(sensitive)})
                updated += 1
        if updated:
            self.flush()
        return updated

    def _remember_with_extraction(self, content: str, memory_type: str, scope: str,
                                  links: Optional[List], metadata: Optional[Dict],
                                  episode: Optional[str], deduplicate: bool, now: float,
                                  observed_at: Optional[float], valid_from: Optional[float], valid_to: Optional[float],
                                  shared_with: Optional[List[str]], sensitive: bool = False) -> Memory:
        """Handle memory storage with fact extraction."""
        try:
            # Extract facts using LLM
            facts = extract_facts(content)

            if not facts:
                logger.warning("No facts extracted from content, storing as-is")
                return self._remember_single_content(
                    content, memory_type, scope, links, metadata, episode, deduplicate, now,
                    observed_at, valid_from, valid_to,
                    shared_with, sensitive,
                )

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
                    fact, memory_type, scope, None, fact_metadata, episode, deduplicate, now,
                    observed_at, valid_from, valid_to,
                    shared_with, sensitive,
                )
                fact_memories.append(fact_memory)
                fact_ids.append(fact_memory.id)

            # Link all extracted facts together with "related" edges
            for i, source_id in enumerate(fact_ids):
                for j, target_id in enumerate(fact_ids):
                    if i != j:
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
            return fact_memories[0]

        except Exception as e:
            logger.warning("Fact extraction failed (%s), storing content as-is", e)
            return self._remember_single_content(
                content, memory_type, scope, links, metadata, episode, deduplicate, now,
                observed_at, valid_from, valid_to,
                shared_with, sensitive,
            )

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register an entity alias with the shared entity normalizer."""
        self._entity_normalizer.register_alias(alias, canonical)
    
    def _remember_single_content(self, content: str, memory_type: str, scope: str,
                                 links: Optional[List], metadata: Optional[Dict],
                                 episode: Optional[str], deduplicate: bool, now: float,
                                 observed_at: Optional[float], valid_from: Optional[float], valid_to: Optional[float],
                                 shared_with: Optional[List[str]], sensitive: bool = False) -> Memory:
        """Store a single piece of content as a memory (extracted from original remember logic)."""
        existing_memories = [
            self._memory_data_to_object(memory_data)
            for memory_data in self.store.memories.values()
            if not memory_data.get('consolidated', False)
        ]

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
            'scope': scope,
            'sensitive': bool(sensitive),
            'strength': 1.0,
            'access_count': 0,
            'created_at': now,
            'last_accessed': now,
            'observed_at': observed_at,
            'valid_from': valid_from,
            'valid_to': valid_to,
            'shared_with': json.dumps(shared_with) if shared_with is not None else None,
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
        concept_names = [concept_name for concept_name, _ in concepts]
        self._register_concept_graph_update(
            new_nodes=concept_names,
            new_edges=self._build_concept_edges(concept_names),
        )
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
            scope=scope,
            shared_with=shared_with,
            sensitive=bool(sensitive),
            strength=1.0,
            access_count=0,
            created_at=now,
            last_accessed=now,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata or {},
            consolidated=False
        )

        # Index extracted triples and feed belief tracker.
        triples = extract_triples(content)
        if triples:
            self.triple_index.add(memory_id, triples)
        self.belief_tracker.on_remember(memory)

        new_contradictions = self.contradiction_detector.check_new_memory(
            content,
            existing_memories,
            new_memory_id=memory_id,
            new_time=now,
        )
        for contradiction in new_contradictions:
            self.contradiction_detector._register(contradiction)
        
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
               retrieval_mode: str = "classic",
               temporal_boost: bool = True, explain: bool = False,
               conflict_aware: bool = False,
               show_disputes: bool = False,
               exclude_conflicted: bool = False,
               temporal: Optional[str] = None,
               scope: str = "private",
               caller_groups: Optional[List[str]] = None,
               user_id: Optional[str] = None) -> List[Memory]:
        """
        Two-Stage Recall with adaptive blending (ported from V1):
          Stage 1A: BM25 word-index candidates  
          Stage 1B: Concept-IDF candidates
          Stage 2: Normalize, blend, apply effective strength, edge/episode expansion
        """
        # Route to user vault if specified
        if user_id:
            vault = self._route_to_vault(user_id)
            if vault and vault != self:
                return vault.recall(
                    context, limit, memory_type, min_strength, retrieval_mode,
                    temporal_boost, explain, conflict_aware, show_disputes,
                    exclude_conflicted, temporal, scope, caller_groups
                )
        
        now = time.time()
        requested_scope = self._normalize_scope(scope)
        if self.scope_policy is not None:
            requested_scope = self.scope_policy.enforce(requested_scope)
        caller_groups = self._normalize_shared_with(caller_groups)

        def _scope_match(memory_data: Dict[str, Any]) -> bool:
            # Sensitive memories are ONLY visible in private scope
            if bool(memory_data.get('sensitive', False)) and requested_scope != "private":
                return False
            return (
                self._is_scope_visible(self._memory_scope(memory_data), requested_scope)
                and (
                    requested_scope != "shared"
                    or self._shared_scope_match(memory_data, caller_groups)
                )
            )

        def _id_visible(memory_id: Optional[int]) -> bool:
            if not isinstance(memory_id, int):
                return False
            memory_data = self.store.memories.get(memory_id)
            if not isinstance(memory_data, dict):
                return False
            return _scope_match(memory_data)

        need_conflicts = conflict_aware or show_disputes or exclude_conflicted
        unresolved_contradictions = self.contradictions() if need_conflicts else []
        if unresolved_contradictions:
            unresolved_contradictions = [
                contradiction for contradiction in unresolved_contradictions
                if _id_visible(getattr(contradiction, "memory_id_a", None))
                and _id_visible(getattr(contradiction, "memory_id_b", None))
            ]
        conflicted_memory_ids = {
            contradiction.memory_id_a
            for contradiction in unresolved_contradictions
        } | {
            contradiction.memory_id_b
            for contradiction in unresolved_contradictions
        }

        dispute_map: Dict[int, List[Dict[str, Any]]] = {}
        if show_disputes:
            dispute_map = self._build_dispute_map(unresolved_contradictions)

        def _finalize_recall(result_memories: List[Memory]) -> List[Memory]:
            if exclude_conflicted:
                result_memories = [
                    memory for memory in result_memories
                    if memory.id not in conflicted_memory_ids
                ]

            if show_disputes:
                for memory in result_memories:
                    memory.disputes = dispute_map.get(memory.id, [])

            if explain:
                self._attach_explain_evidence(result_memories)

            if result_memories:
                self.concept_graph.activate_memories(
                    [memory.id for memory in result_memories if memory.id is not None],
                    now=now,
                )
            self._reinforce_memories([memory.id for memory in result_memories if memory.id is not None])
            return result_memories

        if retrieval_mode not in {"classic", "graph"}:
            raise SynapseValidationError("retrieval_mode must be either 'classic' or 'graph'")

        # Temporal filtering modes.
        temporal_mode: Optional[str] = None
        temporal_start: Optional[float] = None
        temporal_end: Optional[float] = None
        if temporal is not None:
            if temporal == "all":
                # Preserve existing full-chain mode
                base = self.recall(context=context, limit=1, memory_type=memory_type,
                                   min_strength=min_strength, retrieval_mode=retrieval_mode,
                                   temporal_boost=temporal_boost, explain=explain,
                                   conflict_aware=conflict_aware,
                                   show_disputes=show_disputes,
                                   exclude_conflicted=exclude_conflicted,
                                   temporal=None,
                                   scope=requested_scope,
                                   caller_groups=caller_groups)
                if not base:
                    return []
                head = base[0]
                chain = self.history(head.id)
                visible_chain = [
                    entry["memory"]
                    for entry in chain
                    if _id_visible(getattr(entry.get("memory"), "id", None))
                ]
                return _finalize_recall(visible_chain)

            if temporal == "latest":
                temporal_mode = "latest"
            elif temporal.startswith("as_of:"):
                temporal_mode = "as_of"
                temporal_end = parse_temporal(temporal.split(":", 1)[1])
            elif temporal.startswith("during:"):
                temporal_mode = "during"
                _, _, raw = temporal.partition(":")
                parts = raw.split(":")
                if len(parts) == 2:
                    temporal_start = parse_temporal(parts[0])
                    temporal_end = parse_temporal(parts[1])
                    if temporal_start is not None and temporal_end is not None and temporal_start <= temporal_end:
                        pass
                    else:
                        temporal_mode = None
                else:
                    temporal_mode = None
            else:
                month_range = self._parse_month_range(temporal)
                if month_range is not None:
                    temporal_mode = "month"
                    temporal_start, temporal_end = month_range
                else:
                    temporal_mode = "as_of"
                    temporal_end = self._parse_temporal_arg(temporal)

        def _apply_temporal_filter(results: List[Memory]) -> List[Memory]:
            if temporal_mode == "as_of" and temporal_end is not None:
                return memories_as_of(results, temporal_end)
            if temporal_mode == "during" and temporal_start is not None and temporal_end is not None:
                return memories_during(results, temporal_start, temporal_end)
            if temporal_mode == "month" and temporal_start is not None and temporal_end is not None:
                return [
                    memory for memory in results
                    if temporal_start <= memory.created_at < temporal_end
                ]
            if temporal_mode == "latest":
                return latest_facts(results)
            return results

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
                if not _scope_match(memory_data):
                    continue
                    
                memory = self._memory_data_to_object(memory_data)
                if memory.effective_strength >= min_strength:
                    score = memory.effective_strength
                    if conflict_aware and memory.id in conflicted_memory_ids:
                        score *= 0.2
                    all_memories.append((score, memory))
            
            # Sort by effective strength
            all_memories.sort(key=lambda item: item[0], reverse=True)
            result = _apply_temporal_filter([memory for _, memory in all_memories])
            result = result[:limit]
            
            return _finalize_recall(result)
        
        # Count total active memories
        total_memories = len([m for m in self.store.memories.values() 
                             if not m.get('consolidated', False)
                             and (not memory_type or m.get('memory_type') == memory_type)
                             and _scope_match(m)])
        
        if total_memories == 0:
            return []
        
        # ════════════════════════════════════════════════════════════
        #  STAGE 1: Candidate generation
        # ════════════════════════════════════════════════════════════
        if retrieval_mode == "graph":
            active_memories = [
                mdata for mdata in self.store.memories.values()
                if (
                    not mdata.get('consolidated', False)
                    and (not memory_type or mdata.get('memory_type') == memory_type)
                    and _scope_match(mdata)
                )
            ]
            graph_retriever = GraphRetriever(self.concept_graph)
            graph_scores = dict(
                graph_retriever.dual_path_retrieve(
                    context,
                    active_memories,
                    limit=limit * 10,
                    bm25_weight=0.6,
                    graph_weight=0.4,
                )
            )

            bm25_scores = {
                int(memory_id): float(score)
                for memory_id, score in graph_scores.items()
                if float(score) > 0.0
            }
            if not bm25_scores:
                return []
            concept_scores: Dict[int, float] = {}
            embedding_scores: Dict[int, float] = {}
        else:
            # Stage 1A: BM25 candidates
            bm25_scores = self.inverted_index.get_candidates(query_tokens, limit * 10)
            
            # Filter by memory_type and consolidated status
            bm25_scores = {
                mid: score for mid, score in bm25_scores.items()
                if mid in self.store.memories
                and not self.store.memories[mid].get('consolidated', False)
                and (not memory_type or self.store.memories[mid].get('memory_type') == memory_type)
                and _scope_match(self.store.memories[mid])
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
                and _scope_match(self.store.memories[mid])
            }
            
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
                    and _scope_match(self.store.memories[mid])
                }
                # In classic mode, embeddings should not inject totally unrelated memories.
                # Gate embedding-only candidates behind lexical support.
                if retrieval_mode != "graph" and bm25_scores:
                    embedding_scores = {
                        mid: score for mid, score in embedding_scores.items()
                        if mid in bm25_scores
                    }
        
        #  STAGE 2: Concept-Boosted BM25 + Effective Strength
        # ════════════════════════════════════════════════════════════
        
        # In classic mode, do not inject pure concept-graph candidates with zero lexical/embedding support.
        # Graph mode is the intended mechanism for "concept-only" retrieval.
        if retrieval_mode == "graph":
            all_ids = set(bm25_scores.keys()) | set(concept_scores.keys()) | set(embedding_scores.keys())
        else:
            all_ids = set(bm25_scores.keys()) | set(embedding_scores.keys())
            if concept_scores:
                concept_scores = {mid: score for mid, score in concept_scores.items() if mid in all_ids}
        if not all_ids:
            return []
        
        # Adaptive concept weighting based on query length.
        # Graph mode already blends lexical/graph paths upstream.
        n_tokens = len(query_tokens)
        concept_weight = 0.0
        if retrieval_mode != "graph":
            if n_tokens <= 1:
                concept_weight = 0.5
            elif n_tokens <= 2:
                concept_weight = 0.3
            elif n_tokens <= 4:
                concept_weight = 0.2
            else:
                concept_weight = 0.0
        
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
        if retrieval_mode == "graph":
            bm_weight = 1.0
            emb_weight = 0.0
        elif embedding_scores:
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

            if retrieval_mode != "graph":
                # Keep a tiny concept boost inside lexical for very short queries.
                lexical = lexical + cn * concept_weight * 0.05

            if lexical > 0:
                lexical_scores[memory_id] = lexical
                concept_norms[memory_id] = cn

                if explain:
                    sources = []
                    if memory_id in bm25_scores:
                        sources.append("graph_retriever" if retrieval_mode == "graph" else "bm25")
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
                    if ((not memory_type or self.store.memories[memory_id].get('memory_type') == memory_type)
                        and _scope_match(self.store.memories[memory_id])):
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
            if not _scope_match(memory_data):
                continue
                
            memory = self._memory_data_to_object(memory_data)
            if memory.effective_strength < min_strength:
                continue
            
            # Apply supersession penalty
            if memory_id in superseded_ids:
                score *= 0.1  # Heavily demote superseded memories

            if conflict_aware and memory_id in conflicted_memory_ids:
                score *= 0.2

            # Memory level boost: pattern (consolidated) and profile rank higher
            level = memory_data.get('memory_level', 'instance')
            if level == 'pattern':
                score *= 1.3
            elif level == 'profile':
                score *= 1.5

            final_results.append((memory, score))
        
        # Sort and limit
        final_results.sort(key=lambda x: x[1], reverse=True)
        result_memories = _apply_temporal_filter([memory for memory, _ in final_results])
        result_memories = result_memories[:limit]
        
        # Attach score breakdowns if explain=True
        if explain:
            for memory in result_memories:
                if memory.id in breakdowns:
                    memory.score_breakdown = breakdowns[memory.id]

        return _finalize_recall(result_memories)

    def _attach_explain_evidence(self, memories: List[Memory]) -> None:
        """Attach evidence pointers to recalled memories when explain=True."""
        if not memories:
            return

        selected_ids = {memory.id for memory in memories if memory.id is not None}
        if not selected_ids:
            return

        memory_map = {memory.id: memory for memory in memories if memory.id is not None}
        chains = self.evidence_compiler.compile(memories, self.triple_index)

        pointers_by_memory: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for chain in chains:
            supporting = set(chain.supporting_memories)
            contradicting = set(chain.contradicting_memories)
            for memory_id in supporting:
                if memory_id not in selected_ids:
                    continue
                others = sorted(supporting - {memory_id})
                pointers_by_memory[memory_id].append(
                    {
                        "claim": chain.claim,
                        "relation": "supports",
                        "memory_ids": others,
                        "confidence": chain.confidence,
                    }
                )
            for memory_id in contradicting:
                if memory_id not in selected_ids:
                    continue
                others = sorted(contradicting - {memory_id})
                pointers_by_memory[memory_id].append(
                    {
                        "claim": chain.claim,
                        "relation": "contradicts",
                        "memory_ids": others,
                        "confidence": chain.confidence,
                    }
                )

        for memory_id, memory in memory_map.items():
            pointers = pointers_by_memory.get(memory_id, [])
            pointers.sort(key=lambda item: item.get("claim", ""))
            memory.evidence_pointers = pointers

    def compile_context(
        self,
        query: str,
        budget: int = 4000,
        policy: str = "balanced",
        scope: str = "private",
        caller_groups: Optional[List[str]] = None,
    ) -> ContextPack:
        """Compile retrieved memories, graph context and summaries into a ContextPack."""
        resolved_scope = self._normalize_scope(scope)
        if self.scope_policy is not None:
            resolved_scope = self.scope_policy.enforce(resolved_scope)
        return ContextCompiler(self).compile_context(
            query=query,
            budget=budget,
            policy=policy,
            scope=resolved_scope,
            caller_groups=caller_groups,
        )

    def create_card(self, query: str, budget: int = 2000, policy: str = "balanced") -> ContextCard:
        """Create and store a replayable ContextCard for a query."""
        card = ContextCard(self.compile_context(query=query, budget=budget, policy=policy))
        self.cards.add(card)
        return card
    
    def _memory_data_to_object(self, memory_data: Dict) -> Memory:
        """Convert stored memory data to Memory object."""
        return Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            memory_type=memory_data['memory_type'],
            memory_level=memory_data.get('memory_level', 'instance'),
            scope=self._memory_scope(memory_data),
            shared_with=self._parse_shared_with(memory_data.get("shared_with")),
            sensitive=bool(memory_data.get('sensitive', False)),
            strength=memory_data['strength'],
            access_count=memory_data['access_count'],
            created_at=memory_data['created_at'],
            last_accessed=memory_data['last_accessed'],
            observed_at=memory_data.get('observed_at'),
            valid_from=memory_data.get('valid_from'),
            valid_to=memory_data.get('valid_to'),
            metadata=json.loads(memory_data.get('metadata', '{}')),
            consolidated=memory_data.get('consolidated', False),
            summary_of=json.loads(memory_data.get('summary_of', '[]'))
        )

    def contradictions(self) -> List:
        """Return unresolved contradictions detected for this Synapse."""
        return self.contradiction_detector.unresolved_contradictions()

    def _build_dispute_map(self, contradictions: List) -> Dict[int, List[Dict[str, Any]]]:
        """Build per-memory dispute payloads for unresolved contradictions."""
        # ContradictionDetector can yield multiple kinds for the same memory pair
        # (e.g., mutual_exclusion plus a derived temporal_conflict). For recall
        # disputes we want at most one dispute entry per (memory, other_memory).
        kind_priority = {
            "mutual_exclusion": 0,
            "polarity": 1,
            "numeric_range": 2,
            "temporal_conflict": 3,
        }

        best: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        def _maybe_set(owner_id: int, other_id: int, other_text: str, kind: str, confidence: float) -> None:
            existing = best[owner_id].get(other_id)
            candidate = {
                "memory_id": other_id,
                "text": str(other_text or ""),
                "kind": str(kind),
                "confidence": float(confidence),
            }
            if existing is None:
                best[owner_id][other_id] = candidate
                return

            existing_pri = kind_priority.get(existing.get("kind"), 99)
            cand_pri = kind_priority.get(candidate.get("kind"), 99)
            if cand_pri < existing_pri:
                best[owner_id][other_id] = candidate
                return
            if cand_pri == existing_pri and candidate["confidence"] > float(existing.get("confidence", 0.0)):
                best[owner_id][other_id] = candidate

        for contradiction in contradictions:
            a_id = contradiction.memory_id_a
            b_id = contradiction.memory_id_b

            a_data = self.store.memories.get(a_id, {})
            b_data = self.store.memories.get(b_id, {})

            _maybe_set(
                owner_id=a_id,
                other_id=b_id,
                other_text=b_data.get("content", ""),
                kind=contradiction.kind,
                confidence=contradiction.confidence,
            )
            _maybe_set(
                owner_id=b_id,
                other_id=a_id,
                other_text=a_data.get("content", ""),
                kind=contradiction.kind,
                confidence=contradiction.confidence,
            )

        disputes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for owner_id, by_other in best.items():
            disputes[owner_id] = sorted(by_other.values(), key=lambda item: item.get("memory_id", 0))
        return disputes

    def resolve_contradiction(self, contradiction_id: int, winner_memory_id: int):
        """Resolve a contradiction by winner memory id; mark loser as superseded."""
        unresolved = self.contradictions()
        if contradiction_id < 0 or contradiction_id >= len(unresolved):
            raise SynapseValidationError("Invalid contradiction_id")

        target = unresolved[contradiction_id]
        if winner_memory_id not in {target.memory_id_a, target.memory_id_b}:
            raise SynapseValidationError("winner_memory_id must be part of contradiction pair")

        loser_memory_id = target.memory_id_b if winner_memory_id == target.memory_id_a else target.memory_id_a
        if loser_memory_id not in self.store.memories:
            raise SynapseValidationError("Loser memory no longer exists")

        if not any(
            edge.edge_type == "supersedes" and edge.target_id == loser_memory_id
            for edge in self.edge_graph.get_outgoing_edges(winner_memory_id)
        ):
            self.link(winner_memory_id, loser_memory_id, "supersedes", 1.0)

        loser_data = self.store.memories[loser_memory_id]
        loser_meta = json.loads(loser_data.get('metadata', '{}'))
        loser_meta['superseded_by'] = winner_memory_id
        loser_meta['superseded_at'] = time.time()
        self.store.update_memory(loser_memory_id, {'metadata': json.dumps(loser_meta)})

        self.contradiction_detector.resolve_contradiction(contradiction_id)

    def set_retention_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply declarative retention rules."""
        return self.forgetting_policy.retention_rules(rules)

    def policy(self, preset: str = None):
        """Apply a named policy preset and return active policy metadata."""
        if preset is None:
            return self.policy_manager.get_active()
        return self.policy_manager.apply(preset)

    def list_presets(self) -> Dict[str, Any]:
        """List available memory policy presets."""
        return self.policy_manager.list_presets()

    def get_active_policy(self) -> Optional[Dict[str, Any]]:
        """Return active policy metadata."""
        return self.policy_manager.get_active()

    def forget_topic(self, topic: str):
        """Forget all memories related to a topic/concept."""
        return self.forgetting_policy.forget_topic(topic)

    def redact(self, memory_id: int, fields: List[str] = None):
        """Redact a memory while keeping provenance."""
        return self.forgetting_policy.redact(memory_id, fields=fields)

    def gdpr_delete(self, user_id: str = None, concept: str = None):
        """Full deletion by user tag or concept."""
        return self.forgetting_policy.gdpr_delete(user_id=user_id, concept=concept)

    def _coerce_temporal_value(self, value: Optional[Any], default: Optional[float] = None) -> Optional[float]:
        """Coerce an optional temporal value to a float timestamp."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parsed = parse_temporal(value)
            if parsed is not None:
                return parsed
            raise SynapseValidationError(f"Invalid temporal value: {value!r}")
        raise SynapseValidationError(f"Unsupported temporal type: {type(value)!r}")

    @staticmethod
    def _parse_month_range(value: str) -> Optional[tuple]:
        """Return a month interval for an ISO month string or 'March 2024'."""
        normalized = value.strip()
        if not re.fullmatch(r"\d{4}-\d{1,2}", normalized) and not re.fullmatch(r"^[a-zA-Z]+ \d{4}$", normalized):
            return None
        start = parse_temporal(normalized)
        if start is None:
            return None
        dt = datetime.fromtimestamp(start, tz=timezone.utc)
        if dt.month == 12:
            end = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
        else:
            end = datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc).timestamp()
        return (start, end)
    
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

    def _build_concept_edges(self, concepts: List[str]) -> List[tuple]:
        normalized = sorted({self._normalize_concept_for_community(concept) for concept in concepts})
        if len(normalized) < 2:
            return []

        return [
            (left, right, 1.0)
            for left, right in combinations(normalized, 2)
        ]

    def _normalize_concept_for_community(self, concept: str) -> str:
        if not concept:
            return ""
        return self._entity_normalizer.canonical(str(concept), keep_proper_nouns=False)

    def _register_concept_graph_update(self, new_nodes: List[str], new_edges: List[tuple]) -> None:
        if not new_nodes and not new_edges:
            return
        # Keep community recomputation lazy: mark dirty and queue updates.
        # This matches the expectation that communities refresh during sleep
        # (or on-demand via `communities()`), not on every remember().
        self._communities_dirty = True
        self._community_update_pending_nodes.update(
            self._normalize_concept_for_community(node) for node in new_nodes
        )
        self._community_update_pending_edges.extend(new_edges)

    def _invalidate_communities(self) -> None:
        self._communities_dirty = True
        self._community_update_pending_nodes = set()
        self._community_update_pending_edges = []

    def _refresh_communities(self, min_size: int = 3, force: bool = False) -> List[Community]:
        if not force and self._communities is not None and not self._communities_dirty:
            return self._communities

        if self._communities_dirty and self._community_update_pending_nodes:
            try:
                communities = self._community_detector.incremental_update(
                    communities=self._communities or [],
                    new_nodes=sorted(self._community_update_pending_nodes),
                    new_edges=self._community_update_pending_edges,
                )
            except Exception:
                communities = self._community_detector.detect_communities(
                    self.concept_graph,
                    min_size=min_size,
                )
        else:
            communities = self._community_detector.detect_communities(
                self.concept_graph,
                min_size=min_size,
            )

        self._communities = communities
        self._communities_dirty = False
        self._community_update_pending_nodes = set()
        self._community_update_pending_edges = []
        return self._communities

    def communities(self) -> List[Community]:
        """Return detected communities for concept co-occurrence."""
        return self._refresh_communities()

    def community_summary(self, concept: str) -> str:
        """Return a zero-LLM summary for a concept's community."""
        normalized = self._normalize_concept_for_community(concept)
        if not normalized:
            return "No community summary available"

        communities = self.communities()
        if not communities:
            return "No community summary available"

        target = None
        for community in communities:
            if normalized in community.concepts:
                target = community
                break

        if target is None:
            return "No community summary available"

        memory_ids = set()
        for concept_name in target.concepts:
            memory_ids.update(self.concept_graph.get_concept_memories(concept_name))

        memories = [
            self._memory_data_to_object(memory_data)
            for memory_id, memory_data in self.store.memories.items()
            if memory_id in memory_ids and not memory_data.get('consolidated', False)
        ]
        if not memories:
            return "No community summary available"

        return target.summary(memories)

    def should_sleep(self) -> bool:
        """Return whether periodic maintenance should run."""
        if self.count() >= self.sleep_runner._memory_threshold:
            return True
        if self._last_sleep_at is None:
            return False
        return (time.time() - self._last_sleep_at) >= self.sleep_runner.sleep_interval_seconds

    def sleep(self, verbose: bool = False) -> SleepReport:
        """Run sleep maintenance and refresh communities."""
        if self.checkpoints is not None:
            self.checkpoints.checkpoint_before_sleep()
        result = self.sleep_runner.sleep(verbose=verbose)
        self._refresh_communities(force=True)
        return result

    def checkpoint(self, name: str, desc: str = '') -> Checkpoint:
        """Create a named checkpoint from current memory state."""
        return self.checkpoints.create(name=name, description=desc)
    
    def forget(self, memory_id: int) -> bool:
        """Remove a memory completely."""
        if memory_id not in self.store.memories:
            return False
            
        # Remove from indexes
        self.inverted_index.remove_document(memory_id)
        self.concept_graph.unlink_memory(memory_id)
        self._invalidate_communities()
        self.edge_graph.remove_memory_edges(memory_id)
        self.temporal_index.remove_memory(memory_id)
        self.episode_index.remove_memory(memory_id)
        self.embedding_index.remove(memory_id)
        self.triple_index.remove_memory(memory_id)
        
        # Remove from storage
        deleted = self.store.delete_memory(memory_id)
        if deleted:
            self.contradiction_detector.remove_memory(memory_id)
            self.belief_tracker.remove_memory(memory_id)
            memories_for_belief = [
                self._memory_data_to_object(memory_data)
                for memory_data in self.store.memories.values()
                if not memory_data.get('consolidated', False)
                and memory_data.get('id') is not None
            ]
            self.belief_tracker.rebuild(
                sorted(memories_for_belief, key=lambda item: item.created_at)
            )
        return deleted
    
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

    def _link_memory_to_pattern(
        self,
        source_id: int,
        target_id: int,
        edge_type: str = "reminds_of",
        weight: float = 1.0,
    ) -> None:
        """Create internal graph edges for memory provenance links."""
        if source_id is None or target_id is None:
            return
        if source_id == target_id:
            return
        if source_id not in self.store.memories or target_id not in self.store.memories:
            return
        try:
            self.link(source_id, target_id, edge_type, weight)
        except SynapseValidationError:
            # Defensive: keep sleep-mode maintenance running even if provenance
            # links fail due to invalid edge metadata.
            return
    
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
                'scope': 'private',
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
            new_concept_names = sorted(all_concepts)
            self._register_concept_graph_update(
                new_nodes=new_concept_names,
                new_edges=self._build_concept_edges(new_concept_names),
            )
            for concept_name in new_concept_names:
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

    def beliefs(self) -> Dict[str, BeliefVersion]:
        """Return all currently valid belief versions."""
        return self.belief_tracker.get_all_current()

    def belief_history(self, topic: str) -> List[BeliefVersion]:
        """Return versions for facts matching a topic-like filter."""
        return self.belief_tracker.get_matching_history(topic)

    @staticmethod
    def _parse_temporal_arg(temporal: str) -> Optional[float]:
        """Parse a temporal argument like '2024-03' or '2024-06-15' into a timestamp."""
        return parse_temporal(temporal)

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
        # Close vault manager if present
        if getattr(self, '_vault_manager', None):
            try:
                self._vault_manager.close()
            except Exception:
                pass
        # Clean up temporary inbox pending dir for :memory: mode
        if getattr(self, '_inbox_pending_dir', None):
            import shutil
            try:
                shutil.rmtree(self._inbox_pending_dir, ignore_errors=True)
            except Exception:
                pass
            self._inbox_pending_dir = None

    def command(self, text: str) -> str:
        """Execute a /mem command and return a human-readable response."""
        parser = getattr(self, "_command_parser", None)
        if parser is None:
            from command_dsl import MemoryCommandParser

            parser = MemoryCommandParser(self)
            self._command_parser = parser
        return parser.parse_and_execute(text)

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

    # --------------------------------------------------------------------------
    # Memory Router Integration
    # --------------------------------------------------------------------------

    def ingest(self, text: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, policy: str = "auto"):
        """Ingest text through the memory router pipeline.
        
        Args:
            text: Text to process
            source: Source identifier (e.g., "cli", "api", "clipboard")
            metadata: Additional metadata to attach
            policy: Routing policy ("auto", "minimal", "review", "off")
            
        Returns:
            IngestResult indicating what happened
        """
        from capture import ingest
        from review_queue import ReviewQueue
        
        review_queue = ReviewQueue(self) if policy in ['review', 'auto'] else None
        
        return ingest(
            text=text,
            synapse=self,
            review_queue=review_queue,
            source=source,
            meta=metadata,
            policy=policy
        )
    
    def watch(self, stream_type: str = "stdin", **kwargs):
        """Watch a stream and feed to memory router.
        
        Args:
            stream_type: "stdin", "file", or "callback"
            **kwargs: Additional configuration (file_path, policy, batch_size, etc.)
        """
        from watch import watch_stdin, watch_file, create_callback_watcher
        from review_queue import ReviewQueue
        
        review_queue = kwargs.get('review_queue')
        if review_queue is None:
            policy = kwargs.get('policy', 'auto')
            review_queue = ReviewQueue(self) if policy in ['review', 'auto'] else None
            kwargs['review_queue'] = review_queue
        
        if stream_type == "stdin":
            watch_stdin(
                self,
                review_queue=kwargs.get('review_queue'),
                policy=kwargs.get('policy', 'auto'),
                batch_size=kwargs.get('batch_size', 5),
                batch_timeout=kwargs.get('batch_timeout', 30.0),
            )
        elif stream_type == "file":
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path required for file stream type")
            watch_file(
                file_path,
                self,
                review_queue=kwargs.get('review_queue'),
                policy=kwargs.get('policy', 'auto'),
                batch_size=kwargs.get('batch_size', 5),
                batch_timeout=kwargs.get('batch_timeout', 30.0),
            )
        elif stream_type == "callback":
            return create_callback_watcher(
                self,
                review_queue=kwargs.get('review_queue'),
                policy=kwargs.get('policy', 'auto'),
                batch_size=kwargs.get('batch_size', 5),
                batch_timeout=kwargs.get('batch_timeout', 30.0),
            )
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


# Add methods to existing Synapse class via the integration function in watch.py
try:
    from watch import add_watch_methods
    # This would add the methods, but since we defined them directly above, we skip this
    # add_watch_methods(Synapse) 
except ImportError:
    pass  # watch.py not available yet


# ─── Vault Management ────────────────────────────────────────────────────────

def create_vault_manager(base_path: str = "./synapse_vaults") -> "VaultManager":
    """Create a vault manager for multi-user Synapse instances."""
    from vault_manager import VaultManager
    return VaultManager(base_path)


def create_synapse_with_vaults(base_path: str = "./synapse_vaults") -> Synapse:
    """Create a Synapse instance with vault management support."""
    from vault_manager import VaultManager
    vault_manager = VaultManager(base_path)
    main_synapse = Synapse(path=":memory:", vault_manager=vault_manager)
    return main_synapse
