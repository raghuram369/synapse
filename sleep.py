"""Sleep mode maintenance utilities for periodic memory consolidation and cleanup."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class SleepReport:
    consolidated: int
    promoted: int
    patterns_found: int
    contradictions: int
    pruned: int
    graph_cleaned: int
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class SleepRunner:
    def __init__(self, synapse_instance):
        self.synapse = synapse_instance

        # Defaults for sleep scheduling and pipeline heuristics.
        # Backward/forward compatible attribute naming:
        # - older code expects `_memory_threshold`
        # - newer code expects `memory_threshold`
        self._memory_threshold = 100
        self.sleep_interval_seconds = 24 * 3600

        self.promotion_access_threshold = 3
        self.pattern_min_occurrences = 3
        self.streak_min_episodes = 3

        self._max_links_per_provenance = 8

    @property
    def memory_threshold(self) -> int:
        return int(getattr(self, "_memory_threshold", 0))

    @memory_threshold.setter
    def memory_threshold(self, value: int) -> None:
        try:
            self._memory_threshold = int(value)
        except (TypeError, ValueError):
            self._memory_threshold = 0

    def _should_run(self) -> bool:
        return self.synapse.should_sleep()

    def should_sleep(self) -> bool:
        return self._should_run()

    def schedule_hook(self) -> Dict[str, Any]:
        """Return scheduling recommendations for sleep maintenance."""
        now = time.time()
        last_sleep = self.synapse._last_sleep_at
        elapsed_since_last = now - last_sleep if last_sleep is not None else None
        active_count = self.synapse.count()

        memory_due = active_count >= self.memory_threshold
        time_due = elapsed_since_last is not None and elapsed_since_last >= self.sleep_interval_seconds

        hook: Dict[str, Any] = {
            "enabled": True,
            "memory_threshold": self.memory_threshold,
            "sleep_interval_seconds": self.sleep_interval_seconds,
            "active_memory_count": active_count,
            "last_sleep_at": last_sleep,
            "seconds_since_last_sleep": elapsed_since_last,
            "memory_due": memory_due,
            "time_due": time_due,
            "should_sleep": bool(memory_due or time_due),
        }

        if last_sleep is None:
            hook["next_due_at"] = now if memory_due else None
        else:
            hook["next_due_at"] = last_sleep + self.sleep_interval_seconds

        return hook

    def _active_memories(self) -> List[Any]:
        return [
            self.synapse._memory_data_to_object(memory_data)
            for memory_data in self.synapse.store.memories.values()
            if not memory_data.get('consolidated', False)
        ]

    @staticmethod
    def _load_metadata(memory_data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = memory_data.get('metadata', '{}')
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            pass
        return {}

    def _canonical_concept(self, concept: str) -> str:
        if not concept:
            return ""
        return re.sub(r"[^a-z0-9]+", "", concept.lower())

    def _upsert_metadata_list(self, metadata: Dict[str, Any], key: str,
                             values: Sequence[int]) -> None:
        current = set(int(v) for v in metadata.get(key, []) if isinstance(v, int) or str(v).isdigit())
        current.update(int(v) for v in values)
        metadata[key] = sorted(current)

    def _record_pattern_memory(
        self,
        content: str,
        pattern_type: str,
        source_memories: Sequence[int],
        source_episodes: Sequence[int],
    ) -> bool:
        """Store a semantic pattern memory if it does not already exist.

        Returns:
            True if a new memory is created.
        """
        normalized = (content or "").strip().lower()
        for memory_data in self.synapse.store.memories.values():
            if memory_data.get('consolidated', False):
                continue
            if memory_data.get('memory_type') != 'semantic':
                continue
            existing_content = (memory_data.get('content') or "").strip().lower()
            if existing_content != normalized:
                continue

            meta = self._load_metadata(memory_data)
            self._upsert_metadata_list(meta, 'source_memories', source_memories)
            self._upsert_metadata_list(meta, 'source_episodes', source_episodes)
            meta.setdefault('pattern_type', pattern_type)
            meta.setdefault('pattern_hits', 0)
            meta['pattern_hits'] = int(meta.get('pattern_hits') or 0) + 1
            self.synapse.store.update_memory(memory_data['id'], {'metadata': json.dumps(meta)})
            return False

        payload = {
            'pattern_type': pattern_type,
            'source_memories': sorted(set(int(v) for v in source_memories if isinstance(v, int)),),
            'source_episodes': sorted(set(int(v) for v in source_episodes if isinstance(v, int)),),
            'pattern_hits': 1,
            'generated_at': time.time(),
        }
        pattern_memory = self.synapse.remember(
            content,
            memory_type='semantic',
            deduplicate=False,
            metadata=payload,
        )

        for source_memory_id in sorted(set(pattern_memory.metadata.get('source_memories', []))):
            self.synapse._link_memory_to_pattern(
                source_memory_id,
                pattern_memory.id,
                edge_type='reminds_of',
                weight=0.3,
            )

        return True

    def _find_semantic_topic_memory(self, topic: str) -> Optional[int]:
        target = (topic or "").lower()
        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get('consolidated', False):
                continue
            if memory_data.get('memory_type') != 'semantic':
                continue
            meta = self._load_metadata(memory_data)
            topics = [t.lower() for t in meta.get('semantic_topics', []) if isinstance(t, str)]
            if target in topics:
                return memory_id
        return None

    def _promote_to_semantic(self) -> int:
        """Promote frequently recalled episodic memories into semantic memories."""
        topic_stats: Dict[str, Dict[str, Any]] = {}

        for memory in self._active_memories():
            memory_type = memory.memory_type
            if memory_type in {'semantic', 'consolidated'}:
                continue

            concepts = self.synapse.concept_graph.get_memory_concepts(memory.id)
            if not concepts:
                continue

            episode_id = self.synapse.episode_index.get_memory_episode(memory.id)
            for concept in sorted(set(concepts)):
                entry = topic_stats.setdefault(
                    concept,
                    {
                        'access': 0,
                        'memory_ids': set(),
                        'episodes': set(),
                    },
                )
                entry['access'] += int(memory.access_count)
                entry['memory_ids'].add(memory.id)
                if episode_id is not None:
                    entry['episodes'].add(int(episode_id))

        created = 0
        for topic, info in topic_stats.items():
            if info['access'] < self.promotion_access_threshold:
                continue

            source_memories = sorted(info['memory_ids'])
            source_episodes = sorted(info['episodes'])
            existing_id = self._find_semantic_topic_memory(topic)
            if existing_id is not None:
                existing_data = self.synapse.store.memories.get(existing_id)
                if existing_data is None:
                    continue
                meta = self._load_metadata(existing_data)
                self._upsert_metadata_list(meta, 'source_memories', source_memories)
                self._upsert_metadata_list(meta, 'source_episodes', source_episodes)
                meta.setdefault('semantic_topics', [])
                meta['semantic_topics'] = sorted(set(meta['semantic_topics']) | {topic})
                meta['updated_at'] = time.time()
                self.synapse.store.update_memory(existing_id, {'metadata': json.dumps(meta)})
                for source_memory_id in source_memories[:self._max_links_per_provenance]:
                    self.synapse._link_memory_to_pattern(
                        source_memory_id,
                        existing_id,
                        edge_type='supports',
                        weight=0.2,
                    )
                continue

            payload = {
                'semantic_topics': [topic],
                'source_memories': source_memories,
                'source_episodes': source_episodes,
                'promoted_at': time.time(),
            }
            semantic = self.synapse.remember(
                f"User typically does {topic}",
                memory_type='semantic',
                deduplicate=False,
                metadata=payload,
            )
            created += 1
            for source_memory_id in source_memories[:self._max_links_per_provenance]:
                self.synapse._link_memory_to_pattern(
                    source_memory_id,
                    semantic.id,
                    edge_type='supports',
                    weight=0.2,
                )

        return created

    def _mine_patterns(self) -> int:
        """Discover repeated co-occurrences, streaks, and seasonal behavior."""
        occurrences: Dict[str, List[Tuple[int, float, Optional[int]]]] = defaultdict(list)
        pair_memories: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        for memory in self._active_memories():
            concepts = sorted(self.synapse.concept_graph.get_memory_concepts(memory.id))
            if not concepts:
                continue
            episode_id = self.synapse.episode_index.get_memory_episode(memory.id)
            ts = float(memory.created_at)

            for concept in concepts:
                occurrences[concept].append((memory.id, ts, episode_id))

            if len(concepts) >= 2:
                for i, concept_a in enumerate(concepts):
                    for concept_b in concepts[i + 1:]:
                        if concept_a == concept_b:
                            continue
                        key = (concept_a, concept_b)
                        pair_memories[key].add(memory.id)

        patterns_created = 0
        patterns_created += self._mine_co_occurrence_patterns(pair_memories)
        patterns_created += self._mine_streak_patterns(occurrences)
        patterns_created += self._mine_seasonal_patterns(occurrences)

        return patterns_created

    def _mine_co_occurrence_patterns(self, pair_memories: Dict[Tuple[str, str], Set[int]]) -> int:
        created = 0
        for (left, right), source_memories in pair_memories.items():
            if len(source_memories) < self.pattern_min_occurrences:
                continue
            source_memories_sorted = sorted(source_memories)
            content = f"User typically does {left} on {right}."
            source_episodes = self._source_episodes_for_memories(source_memories)
            if self._record_pattern_memory(
                content,
                pattern_type='co_occurrence',
                source_memories=source_memories_sorted,
                source_episodes=source_episodes,
            ):
                created += 1
        return created

    def _source_episodes_for_memories(self, memory_ids: Sequence[int]) -> List[int]:
        episodes = set()
        for memory_id in memory_ids:
            episode_id = self.synapse.episode_index.get_memory_episode(int(memory_id))
            if episode_id is not None:
                episodes.add(int(episode_id))
        return sorted(episodes)

    def _episode_order(self) -> Dict[int, int]:
        episodes = [
            (episode_id, episode_data.get('started_at', 0.0))
            for episode_id, episode_data in self.synapse.store.episodes.items()
        ]
        episodes.sort(key=lambda item: item[1])
        return {episode_id: index for index, (episode_id, _) in enumerate(episodes)}

    def _mine_streak_patterns(self, occurrences: Dict[str, List[Tuple[int, float, Optional[int]]]]) -> int:
        created = 0
        episode_order = self._episode_order()

        for concept, values in occurrences.items():
            seen_episodes = sorted({
                episode_id
                for _, _, episode_id in values
                if episode_id is not None and episode_id in episode_order
            })
            if len(seen_episodes) < self.streak_min_episodes:
                continue

            ordered_indices = [episode_order[eid] for eid in seen_episodes]
            ordered_indices.sort()

            streak_len = 1
            max_streak = 1
            for i in range(1, len(ordered_indices)):
                if ordered_indices[i] == ordered_indices[i - 1] + 1:
                    streak_len += 1
                    max_streak = max(max_streak, streak_len)
                else:
                    streak_len = 1

            if max_streak < self.streak_min_episodes:
                continue

            source_memories = sorted({mid for mid, _, _ in values})
            source_episodes = self._source_episodes_for_memories(source_memories)
            if self._record_pattern_memory(
                f"User typically does {concept} on consecutive episodes.",
                pattern_type='streak',
                source_memories=source_memories,
                source_episodes=source_episodes,
            ):
                created += 1

        return created

    def _season_label_for_timestamp(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%A")

    def _mine_seasonal_patterns(self, occurrences: Dict[str, List[Tuple[int, float, Optional[int]]]]) -> int:
        created = 0

        for concept, values in occurrences.items():
            if len(values) < self.pattern_min_occurrences:
                continue

            buckets = Counter()
            for _, ts, _ in values:
                buckets[self._season_label_for_timestamp(ts)] += 1

            total = len(values)
            top_bucket, top_count = buckets.most_common(1)[0]
            if top_count < self.pattern_min_occurrences:
                continue

            ratio = top_count / float(total)
            if ratio < 0.66:
                continue

            source_memories = sorted({mid for mid, _, _ in values})
            source_episodes = self._source_episodes_for_memories(source_memories)
            if self._record_pattern_memory(
                f"User typically does {concept} on {top_bucket}s.",
                pattern_type='seasonal',
                source_memories=source_memories,
                source_episodes=source_episodes,
            ):
                created += 1

        return created

    def _cleanup_graph(self) -> int:
        """Remove orphan concepts and merge near-duplicate concept nodes."""
        removed = 0
        merged = 0

        cg = self.synapse.concept_graph
        # drop stale memory references and empty concepts.
        for mid in list(cg.memory_concepts.keys()):
            if mid not in self.synapse.store.memories:
                del cg.memory_concepts[mid]

        for concept_name, node in list(cg.concepts.items()):
            if not node.memory_ids:
                del cg.concepts[concept_name]
                removed += 1
                continue

            node.memory_ids = {
                memory_id for memory_id in set(node.memory_ids)
                if memory_id in self.synapse.store.memories
            }
            if not node.memory_ids:
                del cg.concepts[concept_name]
                removed += 1
                continue

            # keep per-memory reverse index tidy
            for memory_id in node.memory_ids:
                cg.memory_concepts[memory_id].add(concept_name)

        # merge normalized near-duplicates (e.g., "py-thon" and "python")
        buckets: Dict[str, List[str]] = defaultdict(list)
        for concept_name in list(cg.concepts.keys()):
            canonical = self._canonical_concept(concept_name)
            if canonical:
                buckets[canonical].append(concept_name)

        canonical_map: Dict[str, str] = {}
        for canonical, names in buckets.items():
            if len(names) <= 1:
                continue

            keep = max(
                names,
                key=lambda item: (len(cg.concepts[item].memory_ids), len(item)),
            )

            for candidate in names:
                canonical_map[candidate] = keep
                if candidate == keep:
                    continue

                node = cg.concepts.pop(candidate, None)
                if node is None:
                    continue
                merged += 1
                cg.concepts[keep].memory_ids.update(node.memory_ids)
                for memory_id in node.memory_ids:
                    if memory_id not in cg.memory_concepts:
                        continue
                    concepts = cg.memory_concepts[memory_id]
                    concepts.discard(candidate)
                    concepts.add(keep)

        for memory_id, concepts in cg.memory_concepts.items():
            remapped = {canonical_map.get(c, c) for c in set(concepts)}
            if remapped != concepts:
                concepts.clear()
                concepts.update(remapped)

        for concept_name, node in list(cg.concepts.items()):
            empty_node = True
            for memory_id in set(node.memory_ids):
                if memory_id in cg.memory_concepts and concept_name in cg.memory_concepts[memory_id]:
                    empty_node = False
                    break
            if empty_node:
                del cg.concepts[concept_name]
                removed += 1

        return removed + merged

    def sleep(self, verbose: bool = False) -> SleepReport:
        if getattr(self.synapse, '_is_sleeping', False):
            return SleepReport(
                consolidated=0,
                promoted=0,
                patterns_found=0,
                contradictions=0,
                pruned=0,
                graph_cleaned=0,
                duration_ms=0.0,
                details={"status": "skipped", "reason": "sleep_in_progress"},
            )

        self.synapse._is_sleeping = True
        start = time.time()
        details: Dict[str, Any] = {
            "verbose": verbose,
        }

        try:
            consolidation = self.synapse.consolidate()
            consolidated = sum(
                int(item.get('source_count', 0))
                for item in consolidation
                if isinstance(item, dict)
            )
            details['consolidation'] = {
                'groups': len(consolidation),
                'memos_merged': consolidated,
            }

            promoted = self._promote_to_semantic()
            details['semantic'] = {
                'promoted': promoted,
            }

            patterns_found = self._mine_patterns()
            details['patterns'] = {
                'found': patterns_found,
            }

            contradiction_results = self.synapse.contradiction_detector.scan_memories(
                self._active_memories(),
            )
            details['contradictions'] = {
                'count': len(contradiction_results),
            }

            pruned_ids = self.synapse.prune(
                min_strength=0.1,
                min_access=0,
                max_age_days=90,
                dry_run=False,
            )
            details['prune'] = {
                'pruned': len(pruned_ids),
            }

            graph_cleaned = self._cleanup_graph()
            details['graph_cleanup'] = {
                'items': graph_cleaned,
            }

            self.synapse._last_sleep_at = time.time()
            return SleepReport(
                consolidated=consolidated,
                promoted=promoted,
                patterns_found=patterns_found,
                contradictions=len(contradiction_results),
                pruned=len(pruned_ids),
                graph_cleaned=graph_cleaned,
                duration_ms=(time.time() - start) * 1000.0,
                details=details,
            )
        finally:
            self.synapse._is_sleeping = False
