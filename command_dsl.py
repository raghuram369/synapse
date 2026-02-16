"""Tiny command parser for chat-native /mem commands."""

from __future__ import annotations

import difflib
import shlex
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from temporal import parse_temporal


class MemoryCommandParser:
    """Parse and render ``/mem`` commands for chat agents."""

    _COMMANDS: Dict[str, str] = {
        "help": "Show this command list",
        "remember": "Store a new memory",
        "recall": "Recall top matching memories",
        "pack": "Build a context pack for a query",
        "rewind": "Show changed memories in a time window",
        "contradict": "Show unresolved contradictions",
        "history": "Show belief history for a fact",
        "timeline": "Show memory timeline",
        "sleep": "Run sleep cycle",
        "stats": "Show store statistics",
        "forget": "Forget memories by topic",
        "search": "Recall with explain mode",
        "export": "Export the store to .synapse",
    }

    _DURATION_UNITS = {
        "d": 86400,
        "w": 86400 * 7,
    }

    def __init__(self, synapse_instance):
        self.synapse = synapse_instance

    def is_memory_command(self, text: str) -> bool:
        """Check if text is a /mem command."""
        return text.strip().startswith('/mem ')

    def command(self, text: str) -> str:
        """Backward-compatible alias for parse_and_execute."""
        return self.parse_and_execute(text)

    def parse_and_execute(self, command: str) -> str:
        """Parse a /mem command and return a human-readable response."""
        if not isinstance(command, str):
            return "⚠️ Command must be a string."

        if not command.strip():
            return "⚠️ Empty command. Did you mean: /mem help"

        if not self.is_memory_command(command) and command.strip() != '/mem':
            return "⚠️ Not a /mem command. Use /mem help for available commands."

        parts = self._split_command(command)
        if not parts:
            return "⚠️ Empty command. Did you mean /mem help?"

        if len(parts) == 1:
            return self._format_help()

        if parts[0] != '/mem':
            return "⚠️ Invalid /mem command format. Use /mem help."

        action = (parts[1] or '').lower()
        args = parts[2:]

        if action == 'help':
            return self._format_help()
        if action == 'remember':
            return self._handle_remember(args)
        if action == 'recall':
            return self._handle_recall(args)
        if action == 'pack':
            return self._handle_pack(args)
        if action == 'rewind':
            return self._handle_rewind(args)
        if action == 'contradict':
            return self._handle_contradict(args)
        if action == 'history':
            return self._handle_history(args)
        if action == 'timeline':
            return self._handle_timeline(args)
        if action == 'sleep':
            return self._handle_sleep(args)
        if action == 'stats':
            return self._handle_stats(args)
        if action == 'forget':
            return self._handle_forget(args)
        if action == 'search':
            return self._handle_search(args)
        if action == 'export':
            return self._handle_export(args)

        suggestions = difflib.get_close_matches(action, self._COMMANDS.keys(), n=1, cutoff=0.65)
        if suggestions:
            return f"⚠️ Unknown command `/mem {action}`. Did you mean /mem {suggestions[0]}?"

        return "⚠️ Unknown command. Use /mem help to see all commands."

    def format_recall_results(self, results, limit: int = 5) -> str:
        """Format recall results as clean text for chat display."""
        entries = [result for result in (results or []) if result is not None][:limit]
        if not entries:
            return "🔍 No matches found."

        lines = [f"🔎 Recall (showing {len(entries)})"]
        for idx, memory in enumerate(entries, start=1):
            text = self._clean_text(getattr(memory, 'content', ''))
            created = self._format_datetime(getattr(memory, 'created_at', 0.0))
            score = getattr(memory, 'effective_strength', getattr(memory, 'strength', 0.0))
            disputes = getattr(memory, 'disputes', None)
            conflict_note = " ⚠️ conflicts" if disputes else ""
            lines.append(
                f"{idx}. #{getattr(memory, 'id', '?')} "
                f"[{getattr(memory, 'memory_type', 'memory')}] "
                f"({created}) score={score:.2f}{conflict_note}"
            )
            lines.append(f"   {text}")
            if getattr(memory, 'score_breakdown', None) is not None:
                breakdown = memory.score_breakdown
                lines.append(
                    "   Breakdown: "
                    f"bm25={breakdown.bm25_score:.2f} "
                    f"concept={breakdown.concept_score:.2f} "
                    f"temporal={breakdown.temporal_score:.2f} "
                    f"episode={breakdown.episode_score:.2f} "
                    f"activation={breakdown.concept_activation_score:.2f} "
                    f"embedding={breakdown.embedding_score:.2f}"
                )
        return "\n".join(lines)

    def format_context_pack(self, pack) -> str:
        """Format ContextPack as readable summary for chat."""
        if pack is None:
            return "⚠️ Empty context pack."

        summaries = pack.summaries or []
        evidence = pack.evidence or []
        concept_names = [
            item.get("name")
            for item in pack.graph_slice.get("concepts", [])
            if isinstance(item, dict) and item.get("name")
        ]

        lines = [
            "🧠 Context Pack",
            f"Query: {pack.query}",
            f"Budget: {pack.budget_used}/{pack.budget_total}",
            "Top memories:",
        ]

        for memory in pack.memories[:4]:
            memory_text = self._clean_text(memory.get('content', ''), max_len=120)
            lines.append(f"- #{memory.get('id', '?')} [{memory.get('memory_type', 'memory')}] {memory_text}")

        if summaries:
            lines.append("Summaries:")
            for summary in summaries:
                lines.append(f"- {summary}")
        else:
            lines.append("Summaries: none")

        if concept_names:
            lines.append("Concepts: " + ", ".join(sorted(concept_names)))

        if evidence:
            lines.append("Evidence:")
            for item in evidence[:3]:
                if isinstance(item, dict) and item.get("claim"):
                    lines.append(f"- {item.get('claim')}")
                else:
                    lines.append(
                        f"- {item.get('source_id')} -> {item.get('relation')} -> "
                        f"{item.get('target_id')}"
                    )

        if not lines:
            lines.append("No context available")

        return "\n".join(lines)

    def format_timeline(self, memories: List[Dict[str, Any]], title: str = "📅 Timeline") -> str:
        """Format memories as timeline."""
        if not memories:
            return "📭 No timeline entries in this range."

        entries = sorted(memories, key=lambda item: item.get("timestamp", 0.0))
        visible_ids = {
            getattr(entry.get("memory"), "id", None)
            for entry in entries
            if entry.get("memory") is not None
        }
        lines = [f"{title} ({len(entries)} changes)"]

        for entry in entries:
            memory = entry.get("memory")
            if memory is None:
                continue
            ts = self._format_datetime(entry.get("timestamp", 0.0))
            delta = ""
            supersedes = entry.get("supersedes")
            superseded_by = entry.get("superseded_by")
            if supersedes is not None and supersedes in visible_ids:
                delta += f" ⇐ supersedes #{supersedes}"
            if superseded_by is not None and superseded_by in visible_ids:
                delta += f" ⇒ superseded by #{superseded_by}"
            if not delta:
                delta = ""
            lines.append(
                f"{ts}  #{getattr(memory, 'id', '?')}"
                f" ({getattr(memory, 'memory_type', 'memory')}){delta}"
            )
            lines.append(f"  {self._clean_text(getattr(memory, 'content', ''), max_len=130)}")

        if not lines:
            return "📭 No timeline entries in this range."
        return "\n".join(lines)

    def _handle_remember(self, args: List[str]) -> str:
        text = self._join_args(args)
        if not text:
            return "⚠️ Usage: /mem remember <text>"

        memory = self.synapse.remember(text)
        return (
            "✅ Stored memory\n"
            f"#{memory.id} ({memory.memory_type}) {self._clean_text(memory.content, max_len=200)}"
        )

    def _handle_recall(self, args: List[str]) -> str:
        query = self._join_args(args)
        if not query:
            return "⚠️ Usage: /mem recall <query>"

        results = self.synapse.recall(query, limit=5)
        return self.format_recall_results(results, limit=5)

    def _handle_pack(self, args: List[str]) -> str:
        if not args:
            return "⚠️ Usage: /mem pack <query> [budget]"

        budget = self._parse_optional_budget(args[-1])
        if budget is not None and len(args) > 1:
            query = self._join_args(args[:-1])
            if not query:
                return "⚠️ Usage: /mem pack <query> [budget]"
            if budget < 0:
                return "⚠️ Budget must be non-negative."
            pack = self.synapse.compile_context(query, budget=budget)
            return self.format_context_pack(pack)

        query = self._join_args(args)
        pack = self.synapse.compile_context(query)
        return self.format_context_pack(pack)

    def _handle_rewind(self, args: List[str]) -> str:
        if not args:
            return "⚠️ Usage: /mem rewind <range> [topic]"

        range_text = args[0]
        topic = self._join_args(args[1:])
        now = time.time()
        parsed_range = self._parse_rewind_range(range_text, now)

        if parsed_range is None:
            return "⚠️ Invalid range. Use days (7 or 7d), timestamps, or start:end."

        start_ts, end_ts = parsed_range
        entries = self._filter_timeline(topic=topic or None, start_ts=start_ts, end_ts=end_ts)
        return self.format_timeline(entries, title=f"🧭 Rewind: {range_text}")

    def _handle_contradict(self, args: List[str]) -> str:
        topic_filter = self._join_args(args).lower()
        conflicts = self.synapse.contradictions()

        if topic_filter:
            filtered = []
            for conflict in conflicts:
                a = self.synapse.store.memories.get(conflict.memory_id_a, {}).get('content', '')
                b = self.synapse.store.memories.get(conflict.memory_id_b, {}).get('content', '')
                if topic_filter in (a or '').lower() or topic_filter in (b or '').lower():
                    filtered.append(conflict)
            conflicts = filtered

        if not conflicts:
            return "✅ No active contradictions found."

        lines = [f"⚠️ Active contradictions ({len(conflicts)})"]
        for idx, conflict in enumerate(conflicts, start=1):
            lines.append(f"{idx}. [{conflict.kind}] confidence={conflict.confidence:.2f}")
            lines.append(
                f"   - #{conflict.memory_id_a}: "
                f"{self._clean_text(self.synapse.store.memories.get(conflict.memory_id_a, {}).get('content', '<missing>'))}"
            )
            lines.append(
                f"   - #{conflict.memory_id_b}: "
                f"{self._clean_text(self.synapse.store.memories.get(conflict.memory_id_b, {}).get('content', '<missing>'))}"
            )
        return "\n".join(lines)

    def _handle_history(self, args: List[str]) -> str:
        query = self._join_args(args)
        if not query:
            return "⚠️ Usage: /mem history <subject>"

        chain = self.synapse.fact_history(query)
        if not chain:
            return f"🔍 No fact history found for '{query}'."

        lines = [f"📜 Fact history for '{query}' ({len(chain)} versions):"]
        for entry in chain:
            memory = entry.get('memory')
            if memory is None:
                continue
            ts = self._format_datetime(getattr(memory, 'created_at', 0.0))
            marker = " (current)" if entry.get('current') else ""
            lines.append(
                f"v{entry.get('version', '?')}: #{getattr(memory, 'id', '?')} "
                f"[{ts}] {marker}\n   {self._clean_text(memory.content, max_len=160)}"
            )
        return "\n".join(lines)

    def _handle_timeline(self, args: List[str]) -> str:
        days = None
        if args and self._parse_days(args[0]) is not None:
            days = self._parse_days(args[0])
            topic = self._join_args(args[1:])
        else:
            topic = self._join_args(args)

        now = time.time()
        if days is not None:
            start_ts = now - (days * 86400)
        else:
            start_ts = None

        entries = self._filter_timeline(
            topic=topic or None,
            start_ts=start_ts,
            end_ts=now,
        )

        title = "📅 Timeline"
        if days is not None:
            title += f" (last {days} day{'s' if days != 1 else ''})"
        if topic:
            title += f" for '{topic}'"

        return self.format_timeline(entries, title=title)

    def _handle_sleep(self, args: List[str]) -> str:
        if args:
            return "⚠️ Usage: /mem sleep"

        report = self.synapse.sleep(verbose=False)
        lines = [
            "😴 Sleep cycle complete",
            f"🧱 Consolidated: {report.consolidated}",
            f"📈 Promoted: {report.promoted}",
            f"🧠 Patterns found: {report.patterns_found}",
            f"⚠️ Contradictions: {report.contradictions}",
            f"🗑️ Pruned: {report.pruned}",
            f"🧹 Graph cleaned: {report.graph_cleaned}",
            f"⏱ {report.duration_ms:.2f}ms",
        ]
        return "\n".join(lines)

    def _handle_stats(self, args: List[str]) -> str:
        if args:
            return "⚠️ Usage: /mem stats"

        active = [
            m for m in self.synapse.store.memories.values()
            if not m.get('consolidated', False)
        ]
        contradictions = self.synapse.contradictions()

        lines = [
            "📊 Store statistics",
            f"🧠 Memories: {len(active)}",
            f"🏷️ Concepts: {len(self.synapse.concept_graph.concepts)}",
            f"🔗 Edges: {len(self.synapse.store.edges)}",
            f"⚠️ Contradictions: {len(contradictions)}",
        ]

        last_sleep = self.synapse._last_sleep_at
        if last_sleep is None:
            lines.append("🛌 Sleep: never")
        else:
            lines.append(f"🛌 Last sleep: {self._format_datetime(last_sleep)}")

        return "\n".join(lines)

    def _handle_forget(self, args: List[str]) -> str:
        topic = self._join_args(args)
        if not topic:
            return "⚠️ Usage: /mem forget <topic>"

        report = self.synapse.forget_topic(topic)
        return (
            "🧹 Forget complete\n"
            f"Topic: {topic}\n"
            f"Matched: {report.get('matched_count', 0)}\n"
            f"Deleted: {report.get('deleted_count', 0)}"
        )

    def _handle_search(self, args: List[str]) -> str:
        query = self._join_args(args)
        if not query:
            return "⚠️ Usage: /mem search <query>"

        results = self.synapse.recall(query, limit=5, explain=True)
        return self.format_recall_results(results, limit=5)

    def _handle_export(self, args: List[str]) -> str:
        if not args:
            path = 'synapse_export.synapse'
        else:
            path = self._join_args(args)
        if not path:
            return "⚠️ Usage: /mem export [path]"

        if not path.endswith('.synapse'):
            path = f"{path}.synapse"

        exported = self.synapse.export(path)
        return f"📤 Exported memory store to: {exported}"

    def _format_help(self) -> str:
        lines = ["🧭 /mem commands:"]
        for name, description in self._COMMANDS.items():
            lines.append(f"/mem {name} - {description}")
        return "\n".join(lines)

    def _split_command(self, text: str) -> List[str]:
        try:
            return shlex.split(text.strip())
        except ValueError:
            return text.strip().split()

    def _join_args(self, args: List[str]) -> str:
        return " ".join(arg.strip() for arg in args).strip()

    def _parse_optional_budget(self, value: str) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_days(self, value: str) -> Optional[int]:
        text = (value or '').strip().lower()
        if not text:
            return None

        if text.isdigit():
            return int(text)

        unit = text[-1:]
        if unit in self._DURATION_UNITS and text[:-1].isdigit():
            return int(text[:-1])

        return None

    def _parse_rewind_range(self, raw_range: str, now: float):
        if ':' in raw_range:
            left, right = raw_range.split(':', 1)
            start = parse_temporal(left.strip())
            end = parse_temporal(right.strip())
            if start is None and end is None:
                return None
            if start is not None and end is not None and start > end:
                start, end = end, start
            if start is not None and end is not None:
                return start, end
            if start is None:
                return max(now - 30 * 86400, 0.0), end
            return start, min(end, now)

        parsed_days = self._parse_days(raw_range)
        if parsed_days is not None:
            if parsed_days < 0:
                return None
            return now - (parsed_days * 86400), now

        start = parse_temporal(raw_range)
        if start is None:
            return None

        return start, now

    def _filter_timeline(self, topic: Optional[str], start_ts: Optional[float], end_ts: Optional[float]):
        entries = self.synapse.timeline(concept=None)
        topic_text = (topic or "").strip().lower()
        filtered = []
        for entry in entries:
            timestamp = entry.get('timestamp')
            if timestamp is None:
                continue
            if start_ts is not None and timestamp < start_ts:
                continue
            if end_ts is not None and timestamp > end_ts:
                continue
            if topic_text:
                memory = entry.get("memory")
                content = (getattr(memory, "content", "") or "").lower()
                memory_id = getattr(memory, "id", None)
                concept_hits = set()
                if memory_id is not None:
                    concept_hits = {
                        str(name).lower()
                        for name in (self.synapse.concept_graph.get_memory_concepts(memory_id) or set())
                    }
                if topic_text not in content and topic_text not in concept_hits:
                    continue
            filtered.append(entry)
        return sorted(filtered, key=lambda item: item.get('timestamp', 0.0))

    @staticmethod
    def _format_datetime(timestamp: float) -> str:
        try:
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except (TypeError, OSError, OverflowError, ValueError):
            return "unknown"

    @staticmethod
    def _clean_text(value: Any, max_len: int = 100) -> str:
        if value is None:
            return ""
        text = str(value).replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return f"{text[:max_len - 3]}..."
