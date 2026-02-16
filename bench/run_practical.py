#!/usr/bin/env python3
"""Practical benchmark over realistic multi-session agent memories (Synapse AI Memory)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse, Memory


RANDOM_SEED = 2026
K = 10

PRACTICAL_SESSIONS = [
    {
        "session": "s1",
        "date": "2024-01-06",
        "messages": [
            "Hi, I'm Jordan Patel and I work as a data analyst at Brightline.",
            "I currently live in Seattle and bike to work in the mornings.",
            "I have a peanut allergy, so I avoid peanut-containing meals.",
            "My favorite coffee drink is a double espresso.",
            "I'm helping the platform team build a new observability dashboard.",
        ],
    },
    {
        "session": "s2",
        "date": "2024-02-14",
        "messages": [
            "I moved from Seattle to Denver on March 1st after transferring teams.",
            "My manager at Brightline is Dana for the observability project.",
            "My preferred lunch spots are places with Mediterranean options.",
            "I take Tuesday nights for pottery class and Sunday for hiking.",
        ],
    },
    {
        "session": "s3",
        "date": "2024-05-09",
        "messages": [
            "I switched to a more plant-forward diet, but still eat fish.",
            "I started doing office hours at 10am instead of 9am.",
            "I'm leading a migration from Airflow to Dagster this quarter.",
            "My favorite music style is lo-fi jazz while working.",
        ],
    },
    {
        "session": "s4",
        "date": "2024-08-22",
        "messages": [
            "I moved again, now living in Austin.",
            "I changed roles to senior software engineer on the data platform team.",
            "My new manager is Samira.",
            "My preferred first meeting location is in-person at the cafe next to the office.",
        ],
    },
    {
        "session": "s5",
        "date": "2024-10-11",
        "messages": [
            "I changed employers to Nova Health and now support agent operations.",
            "I still avoid peanuts and I no longer drink espresso; I switched to iced chai.",
            "Jordan's travel plans this quarter are for a trip to Kyoto in November.",
            "I want reminders to prioritize sleep before Thursday calls.",
        ],
    },
]

PRACTICAL_QUERIES = [
    {"query": "What is the user's current employer?", "category": "facts", "expected_keywords": ["Nova", "Nova Health"]},
    {"query": "What is Jordan's current title?", "category": "facts", "expected_keywords": ["senior", "software engineer"]},
    {"query": "Who is the current manager?", "category": "facts", "expected_keywords": ["Samira"]},

    {"query": "When did Jordan move to Denver?", "category": "temporal", "expected_keywords": ["March", "2024"]},
    {"query": "Where did Jordan move in August 2024?", "category": "temporal", "expected_keywords": ["Austin"]},
    {"query": "What time did Jordan shift office hours to?", "category": "temporal", "expected_keywords": ["10", "10am"]},

    {"query": "What food options should a planner avoid when suggesting dinner for Jordan?", "category": "preference", "expected_keywords": ["peanut", "avoid"]},
    {"query": "What is Jordan's current drink preference?", "category": "preference", "expected_keywords": ["iced chai", "chai"]},
    {"query": "What type of music does Jordan prefer while working?", "category": "preference", "expected_keywords": ["lo-fi", "lofi", "jazz"]},

    {"query": "Would Kyoto vacation require dietary planning because of any allergies?", "category": "multi-hop", "expected_keywords": ["peanut", "allergy"]},
    {"query": "Which work change happened after moving to Austin?", "category": "multi-hop", "expected_keywords": ["Nova Health", "data platform", "senior software engineer"]},
    {"query": "If booking lunch for Jordan in Denver, what food preference should be considered and what job schedule change matters?", "category": "multi-hop", "expected_keywords": ["Mediterranean", "10am", "Denver"]},

    {"query": "What city does Jordan live in now?", "category": "correction", "expected_keywords": ["Austin"]},
    {"query": "What role and office do they hold now?", "category": "correction", "expected_keywords": ["software engineer", "Austin"]},
    {"query": "Which manager is current versus previous?", "category": "correction", "expected_keywords": ["Samira"]},
    {"query": "What coffee drink is currently preferred?", "category": "correction", "expected_keywords": ["chai", "iced"]},
]


@dataclass
class Metrics:
    total: int = 0
    hits_1: int = 0
    hits_5: int = 0
    hits_10: int = 0
    mrr_sum: float = 0.0

    def add(self, hit_rank: Optional[int]) -> None:
        self.total += 1
        if hit_rank is None:
            return
        if hit_rank <= 1:
            self.hits_1 += 1
        if hit_rank <= 5:
            self.hits_5 += 1
        if hit_rank <= K:
            self.hits_10 += 1
        self.mrr_sum += 1.0 / hit_rank

    def recall(self, k: int) -> float:
        if self.total == 0:
            return 0.0
        if k == 1:
            return self.hits_1 / self.total
        if k == 5:
            return self.hits_5 / self.total
        if k == 10:
            return self.hits_10 / self.total
        raise ValueError("k must be 1, 5, or 10")

    def mrr(self) -> float:
        return self.mrr_sum / self.total if self.total else 0.0


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def query_hit(memory: Memory, expected_keywords: List[str]) -> bool:
    text = memory.content.lower()
    return any(word.lower() in text for word in expected_keywords)


def run_benchmark(seed: int, sessions: List[dict], queries: List[dict], save: bool = True) -> Dict[str, object]:
    set_seed(seed)
    synapse = Synapse(":memory:")
    synapse._use_embeddings = False

    all_memories: List[Memory] = []
    for session in sessions:
        date = session["date"]
        for msg in session["messages"]:
            all_memories.append(synapse.remember(f"[{date}] {msg}"))

    by_category = defaultdict(Metrics)
    overall = Metrics()

    results_for_log = []
    start = time.time()
    for q in queries:
        query = q["query"]
        category = q["category"]
        expected = q["expected_keywords"]

        recalled = synapse.recall(query, limit=K)
        hit_rank: Optional[int] = None
        for rank, mem in enumerate(recalled, start=1):
            if query_hit(mem, expected):
                hit_rank = rank
                break

        by_category[category].add(hit_rank)
        overall.add(hit_rank)

        results_for_log.append({
            "query": query,
            "category": category,
            "rank": hit_rank,
            "top_match": recalled[0].content if recalled else None,
        })

    elapsed = time.time() - start
    output = {
        "config": {"seed": seed, "limit": K, "num_memories": len(all_memories), "num_queries": len(queries)},
        "overall": {
            "n": overall.total,
            "recall_1": overall.recall(1),
            "recall_5": overall.recall(5),
            "recall_10": overall.recall(10),
            "mrr": overall.mrr(),
            "elapsed_seconds": elapsed,
        },
        "by_category": {
            key: {
                "n": bucket.total,
                "recall_1": bucket.recall(1),
                "recall_5": bucket.recall(5),
                "recall_10": bucket.recall(10),
                "mrr": bucket.mrr(),
            }
            for key, bucket in sorted(by_category.items())
        },
        "queries": results_for_log,
    }

    print_results(output)
    if save:
        out_path = ROOT / "bench" / f"practical_benchmark_{seed}_{int(time.time())}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {out_path}")
    return output


def print_results(output: Dict[str, object]) -> None:
    overall = output["overall"]
    by_category = output["by_category"]

    print("\n" + "=" * 72)
    print("Practical Memory Benchmark (Synapse AI Memory)")
    print("=" * 72)
    print(f"Seed: {output['config']['seed']}")
    print(f"Memories inserted: {output['config']['num_memories']}")
    print(f"Queries: {output['config']['num_queries']}")

    print(f"\n{'Metric':<10}{'Synapse AI Memory':>18}")
    print("-" * 22)
    print(f"{'Recall@10':<10}{overall['recall_10'] * 100:>11.2f}%")
    print(f"{'Recall@5':<10}{overall['recall_5'] * 100:>11.2f}%")
    print(f"{'Recall@1':<10}{overall['recall_1'] * 100:>11.2f}%")
    print(f"{'MRR':<10}{overall['mrr'] * 100:>11.2f}%")
    print(f"Runtime: {overall['elapsed_seconds']:.2f}s")

    print("\nPer-category results (Recall@10, Recall@5, Recall@1, MRR):")
    print(f"{'Category':<14}{'R@10':>10}{'R@5':>10}{'R@1':>10}{'MRR':>10}{'N':>6}")
    print("-" * 70)
    for category in ["facts", "temporal", "preference", "multi-hop", "correction"]:
        if category not in by_category:
            continue
        row = by_category[category]
        print(
            f"{category:<14}{row['recall_10'] * 100:>9.2f}%{row['recall_5'] * 100:>10.2f}%"
            f"{row['recall_1'] * 100:>10.2f}%{row['mrr'] * 100:>10.2f}%{row['n']:>6}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run practical memory benchmark.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Pinned seed for reproducible behavior.")
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        seed=args.seed,
        sessions=PRACTICAL_SESSIONS,
        queries=PRACTICAL_QUERIES,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
