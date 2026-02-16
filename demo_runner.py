from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, List

from synapse import Synapse


class DemoRunner:
    SCENARIOS = {
        'diet': {
            'description': 'Food preferences, allergies, and dietary memory',
            'memories': [
                'User is vegetarian and allergic to shellfish',
                'User loves Italian food, especially pasta',
                'User recently tried sushi and liked it',
                'User is lactose intolerant',
                'User prefers organic produce',
            ],
            'queries': ['What should I eat?', 'Any food allergies?', 'Restaurant suggestions?'],
        },
        'travel': {
            'description': 'Location history and travel preferences',
            'memories': [
                'User lived in Austin TX from 2020 to 2023',
                'User moved to Denver CO in January 2024',
                'User prefers window seats on flights',
                'User has Global Entry for customs',
                'User visited Tokyo in March 2024 and loved it',
            ],
            'queries': ['Where does user live?', 'Travel preferences?', 'Where did user live in 2021?'],
        },
        'project': {
            'description': 'Project management and work context',
            'memories': [
                'Project Alpha deadline is March 1 2026',
                'Team uses Python and FastAPI for backend',
                'Alice is the project lead, Bob handles frontend',
                'Sprint reviews happen every other Friday',
                'Migrated from PostgreSQL to SQLite last month',
            ],
            'queries': ['When is the deadline?', 'Who leads the project?', 'What tech stack?'],
        },
    }

    _ANSI_BOLD = "\033[1m"
    _ANSI_CYAN = "\033[36m"
    _ANSI_GREEN = "\033[32m"
    _ANSI_YELLOW = "\033[33m"
    _ANSI_BLUE = "\033[34m"
    _ANSI_RESET = "\033[0m"

    def _color(self, text: str, code: str, enabled: bool) -> str:
        if not enabled:
            return text
        return f"{code}{text}{self._ANSI_RESET}"

    def _snip(self, text: str, max_len: int = 110) -> str:
        value = (text or "").replace("\n", " ").strip()
        if len(value) <= max_len:
            return value
        return f"{value[:max_len-3]}..."

    def _score_lines(self, memory) -> list[str]:
        breakdown = memory.score_breakdown
        if breakdown is None:
            return [
                "BM25: 0.0000",
                "Concept: 0.0000",
                "Temporal: 0.0000",
                "Episode: 0.0000",
                "Concept Activation: 0.0000",
                "Embedding: 0.0000",
                "Match Sources: []",
            ]

        sources = breakdown.match_sources or []
        return [
            f"BM25: {breakdown.bm25_score:.4f}",
            f"Concept: {breakdown.concept_score:.4f}",
            f"Temporal: {breakdown.temporal_score:.4f}",
            f"Episode: {breakdown.episode_score:.4f}",
            f"Concept Activation: {breakdown.concept_activation_score:.4f}",
            f"Embedding: {breakdown.embedding_score:.4f}",
            f"Match Sources: {sources}",
        ]

    def _format_sleep_digest(self, sleep_report) -> list[str]:
        lines = [
            f"consolidated: {sleep_report.consolidated}",
            f"promoted: {sleep_report.promoted}",
            f"patterns_found: {sleep_report.patterns_found}",
            f"contradictions: {sleep_report.contradictions}",
            f"pruned: {sleep_report.pruned}",
            f"graph_cleaned: {sleep_report.graph_cleaned}",
            f"duration_ms: {sleep_report.duration_ms:.2f}",
        ]
        return lines

    def _format_contradictions(self, synapse: Synapse) -> list[str]:
        contradictions = synapse.contradictions()
        if not contradictions:
            return ["No contradictions detected."]

        lines = []
        for idx, conflict in enumerate(contradictions, start=1):
            left = self._snip(synapse.store.memories.get(conflict.memory_id_a, {}).get('content', '<missing>'))
            right = self._snip(synapse.store.memories.get(conflict.memory_id_b, {}).get('content', '<missing>'))
            lines.append(
                f"{idx}. #{conflict.memory_id_a} vs #{conflict.memory_id_b} "
                f"({conflict.kind}, confidence={conflict.confidence:.2f})"
            )
            lines.append(f"   - {left}")
            lines.append(f"   - {right}")
            lines.append(f"   - {conflict.description}")
        return lines

    def _collect_results(self, synapse: Synapse, scenario: dict[str, Any]) -> tuple[
        list[Any],
        list[tuple[str, float, list[Any]]],
        Any,
        list[str],
    ]:
        memories = []
        for item in scenario['memories']:
            memories.append(synapse.remember(item, deduplicate=False))

        query_runs = []
        for query in scenario['queries']:
            started = time.perf_counter()
            results = synapse.recall(query, limit=5, explain=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            query_runs.append((query, elapsed_ms, results))

        sleep_report = synapse.sleep(verbose=False)
        contradictions = self._format_contradictions(synapse)
        return memories, query_runs, sleep_report, contradictions

    def run(self, scenario='diet', output='terminal') -> str:
        '''
        Run a demo scenario:
        1. Create temp Synapse instance
        2. Store memories
        3. Run queries, show results with score breakdowns
        4. Run sleep(), show digest
        5. Show contradictions if any
        6. If output='markdown': return formatted markdown
        7. If output='terminal': print colored terminal output
        '''
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if output not in {"terminal", "markdown"}:
            raise ValueError("output must be 'terminal' or 'markdown'")

        config = self.SCENARIOS[scenario]
        isatty = output == "terminal" and bool(getattr(sys.stdout, "isatty", lambda: False)())

        with tempfile.TemporaryDirectory(prefix="synapse-demo-") as tmp:
            db_path = os.path.join(tmp, "demo_store")
            synapse = Synapse(db_path)
            try:
                memories, query_runs, sleep_report, contradictions = self._collect_results(synapse, config)
            finally:
                synapse.close()

        if output == "markdown":
            lines: List[str] = []
            label = scenario.capitalize()
            lines.append(f"# Demo Scenario: {label}")
            lines.append("")
            lines.append(f"**Description:** {config['description']}")
            lines.append("")
            lines.append("## Stored memories")
            for idx, memory in enumerate(memories, start=1):
                lines.append(f"{idx}. #{memory.id}: {self._snip(memory.content)}")
            lines.append("")

            lines.append("## Queries")
            for query, elapsed_ms, results in query_runs:
                lines.append(f"### {query}")
                lines.append(f"- time_ms: {elapsed_ms:.2f}")
                if not results:
                    lines.append("- No matching memories.")
                else:
                    lines.append(f"- matched: {len(results)}")
                    for result in results:
                        lines.append(f"  - #{result.id}: {self._snip(result.content)}")
                        lines.append("    - Score breakdown:")
                        for row in self._score_lines(result):
                            lines.append(f"      - {row}")
                lines.append("")

            lines.append("## Sleep digest")
            for row in self._format_sleep_digest(sleep_report):
                lines.append(f"- {row}")
            lines.append("")

            lines.append("## Contradictions")
            lines.extend(contradictions if contradictions else ["None"])
            text = "\n".join(lines)
            return text

        # Terminal output
        terminal_lines: List[str] = []
        label = scenario.capitalize()
        terminal_lines.append(self._color(f"Demo Scenario: {label}", self._ANSI_BOLD, isatty))
        terminal_lines.append(self._color(config['description'], self._ANSI_CYAN, isatty))
        terminal_lines.append(self._color("Stored memories:", self._ANSI_BLUE, isatty))
        for idx, memory in enumerate(memories, start=1):
            terminal_lines.append(
                self._color(
                    f"{idx}. #{memory.id}: {self._snip(memory.content)}",
                    self._ANSI_GREEN,
                    isatty,
                )
            )
        terminal_lines.append("")
        terminal_lines.append(self._color("Queries:", self._ANSI_BLUE, isatty))
        for query, elapsed_ms, results in query_runs:
            terminal_lines.append(self._color(f"Query: {query}", self._ANSI_YELLOW, isatty))
            terminal_lines.append(f"  latency_ms: {elapsed_ms:.2f}")
            if not results:
                terminal_lines.append(self._color("  No matching memories.", self._ANSI_YELLOW, isatty))
            else:
                terminal_lines.append(f"  matched: {len(results)}")
                for result in results:
                    terminal_lines.append(f"  - #{result.id}: {self._snip(result.content)}")
                    terminal_lines.append("    Score breakdown:")
                    for row in self._score_lines(result):
                        terminal_lines.append(f"      {row}")
            terminal_lines.append("")

        terminal_lines.append(self._color("Sleep digest:", self._ANSI_BLUE, isatty))
        for row in self._format_sleep_digest(sleep_report):
            terminal_lines.append(f"  {row}")
        terminal_lines.append("")
        terminal_lines.append(self._color("Contradictions:", self._ANSI_BLUE, isatty))
        for row in contradictions:
            terminal_lines.append(f"  {row}")
        text = "\n".join(terminal_lines)
        print(text)
        return text

    def run_all(self, output: str = 'terminal') -> str:
        outputs = []
        for scenario in self.SCENARIOS:
            outputs.append(self.run(scenario=scenario, output=output))
        return "\n\n".join(outputs)
