#!/usr/bin/env python3
"""Reproducible LOCOMO benchmark runner for Synapse AI Memory."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure Synapse AI Memory import works when running from bench/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Memory, Synapse


RANDOM_SEED = 2026
DEFAULT_URL = ""
LOCAL_DATA_PATH = ROOT / "locomo_data.json"

LOCOMO_EVAL_LIMIT = 10
CATEGORY_TO_METHOD = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "open-domain",
}
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "direct retrieval / open-domain",
    5: "adversarial open-domain",
}


@dataclass
class BenchmarkMetrics:
    total: int = 0
    recall_1: int = 0
    recall_5: int = 0
    recall_10: int = 0
    mrr_sum: float = 0.0

    def add(self, rank: Optional[int]) -> None:
        self.total += 1
        if rank is not None:
            if rank <= 1:
                self.recall_1 += 1
            if rank <= 5:
                self.recall_5 += 1
            if rank <= 10:
                self.recall_10 += 1
            self.mrr_sum += 1.0 / rank

    def recall(self, k: int) -> float:
        if self.total == 0:
            return 0.0
        if k == 1:
            return self.recall_1 / self.total
        if k == 5:
            return self.recall_5 / self.total
        if k == 10:
            return self.recall_10 / self.total
        raise ValueError("k must be 1, 5, or 10")

    def mrr(self) -> float:
        return self.mrr_sum / self.total if self.total else 0.0


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def _safe_read_json_text(raw: str) -> Sequence[dict]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Expected list of dialogues")
    return data


def _download_dataset(url: str) -> Sequence[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "synapse-ai-memory-bench/locomo"})
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to download dataset: HTTP {response.status}")
        return _safe_read_json_text(response.read().decode("utf-8"))


def _load_from_file(path: Path) -> Sequence[dict]:
    with path.open("r", encoding="utf-8") as f:
        return _safe_read_json_text(f.read())


def _build_fallback_sample() -> Sequence[dict]:
    return [
        {
            "sample_id": "fallback-1",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Riley",
                "session_1_date_time": "2024-01-03T10:00:00",
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Caroline", "text": "Hi, I'm Caroline and I live in Austin, Texas with my cat Juniper."},
                    {"dia_id": "D1:2", "speaker": "Riley", "text": "What did we decide for your wedding date?"},
                    {"dia_id": "D1:3", "speaker": "Caroline", "text": "Let's set the wedding for July 2025."},
                    {"dia_id": "D1:4", "speaker": "Caroline", "text": "I moved to Austin from Boston in March 2024."}
                ],
                "session_2_date_time": "2024-04-12T16:22:00",
                "session_2": [
                    {"dia_id": "D1:5", "speaker": "Caroline", "text": "Our wedding venue is Sunset Terrace and they'll need it by 4pm for a rain backup date."},
                    {"dia_id": "D1:6", "speaker": "Riley", "text": "Budget note: they asked for a vegetarian option too."},
                    {"dia_id": "D1:7", "speaker": "Caroline", "text": "I work at the city design lab and I'm focused on urban mobility dashboards."},
                    {"dia_id": "D1:8", "speaker": "Riley", "text": "She really likes warm weather and trail runs."}
                ]
            },
            "qa": [
                {"question": "Where does Caroline live now?", "answer": "Austin", "evidence": ["D1:1"], "category": 1},
                {"question": "When did Caroline move to Austin?", "answer": "March 2024", "evidence": ["D1:4"], "category": 2},
                {"question": "Where is Caroline's wedding venue and what happened in case of rain?", "answer": "Sunset Terrace and rain backup time at 4pm", "evidence": ["D1:5"], "category": 3},
                {"question": "What are the dietary needs mentioned for the wedding menu?", "answer": "vegetarian option", "evidence": ["D1:6"], "category": 4},
                {"question": "If the user prefers warm weather, where are they planning a hike?", "answer": "trail runs", "evidence": ["D1:8", "D1:1"], "category": 5}
            ],
        },
        {
            "sample_id": "fallback-2",
            "conversation": {
                "speaker_a": "Eli",
                "speaker_b": "Nora",
                "session_1_date_time": "2024-02-11T09:00:00",
                "session_1": [
                    {"dia_id": "D2:1", "speaker": "Eli", "text": "My name is Eli and I teach machine learning at a community college."},
                    {"dia_id": "D2:2", "speaker": "Nora", "text": "Good news: Eli passed the cloud certification on Friday."},
                    {"dia_id": "D2:3", "speaker": "Eli", "text": "I live in Portland, Oregon and bike to campus."},
                    {"dia_id": "D2:4", "speaker": "Eli", "text": "I am allergic to shellfish."}
                ],
                "session_2_date_time": "2024-07-01T14:00:00",
                "session_2": [
                    {"dia_id": "D2:5", "speaker": "Eli", "text": "I now work at Azure Learning Studio as an assistant principal."},
                    {"dia_id": "D2:6", "speaker": "Eli", "text": "I moved to Seattle in June 2024."},
                    {"dia_id": "D2:7", "speaker": "Eli", "text": "Ask me for seafood recommendations only if they are shellfish-free."}
                ]
            },
            "qa": [
                {"question": "What city does Eli live in after June 2024?", "answer": "Seattle", "evidence": ["D2:6"], "category": 2},
                {"question": "What did Eli do on weekdays before moving to Seattle?", "answer": "taught machine learning at a community college", "evidence": ["D2:1"], "category": 1},
                {"question": "What role did Eli move into after his career change?", "answer": "assistant principal", "evidence": ["D2:5"], "category": 3},
                {"question": "What food should be avoided in recommendations for Eli?", "answer": "shellfish", "evidence": ["D2:4", "D2:7"], "category": 4},
                {"question": "Given Eli moved cities and changed jobs, what two major facts changed?", "answer": "Seattle and assistant principal", "evidence": ["D2:5", "D2:6"], "category": 5}
            ],
        }
    ]


def load_locomo_dataset(path: Optional[Path], data_url: str, seed: int, use_download: bool = False) -> Tuple[Sequence[dict], str]:
    _ = seed
    data_path = path if path else LOCAL_DATA_PATH
    if data_path and data_path.exists():
        return _load_from_file(data_path), f"local file: {data_path}"
    if use_download:
        if not data_url:
            print("[WARN] No --dataset-url provided; using reproducible fallback sample.")
            return _build_fallback_sample(), "embedded fallback sample"
        try:
            return _download_dataset(data_url), f"url: {data_url}"
        except (urllib.error.URLError, RuntimeError, ValueError) as error:
            print(f"[WARN] Could not download LoCoMo data ({error}); using reproducible fallback sample.")
    return _build_fallback_sample(), "embedded fallback sample"


def _conversation_turns(dialogue: dict) -> List[Tuple[str, str, str]]:
    turns: List[Tuple[str, str, str]] = []
    conversation = dialogue.get("conversation", {})
    if not isinstance(conversation, dict):
        return turns

    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(value, list):
            continue
        for turn in value:
            if not isinstance(turn, dict):
                continue
            dia_id = str(turn.get("dia_id", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()
            if not dia_id or not text:
                continue
            if not speaker:
                speaker = "speaker"
            turns.append((dia_id, speaker, text))
    return turns


def _normalize_dialogue(dialogue: dict, idx: int) -> Optional[dict]:
    qa = dialogue.get("qa", [])
    if not isinstance(qa, list):
        return None
    if not qa:
        return None
    sample_id = str(dialogue.get("sample_id", f"dialogue-{idx}"))
    conversation = dialogue.get("conversation", {})
    if not isinstance(conversation, dict):
        return None
    return {"sample_id": sample_id, "conversation": conversation, "qa": qa}


def run_locomo(dialogues: Sequence[dict], seed: int) -> Tuple[dict, int, int]:
    _ = seed
    total_qa = sum(len(d["qa"]) for d in dialogues)
    method_metrics: Dict[str, BenchmarkMetrics] = defaultdict(BenchmarkMetrics)
    raw_category_metrics: Dict[int, BenchmarkMetrics] = defaultdict(BenchmarkMetrics)
    overall = BenchmarkMetrics()

    processed_dialogues = 0
    start = time.time()
    for dialogue in dialogues:
        sample_id = dialogue["sample_id"]
        turns = _conversation_turns(dialogue)
        if not turns:
            print(f"[WARN] {sample_id}: no turns found; skipping dialogue.")
            continue

        synapse = Synapse(":memory:")
        synapse._use_embeddings = False

        memory_by_id: Dict[int, str] = {}
        for dia_id, speaker, text in turns:
            memory = synapse.remember(f"{speaker}: {text}")
            memory_by_id[memory.id] = dia_id

        for qa in dialogue["qa"]:
            question = str(qa.get("question", "")).strip()
            evidence = qa.get("evidence", [])
            try:
                category = int(qa.get("category", 0))
            except (TypeError, ValueError):
                category = 0
            if not question:
                continue
            evidence_set = {str(item).strip() for item in evidence if str(item).strip()}
            if not evidence_set:
                continue

            recalled = synapse.recall(question, limit=LOCOMO_EVAL_LIMIT)
            hit_rank: Optional[int] = None
            for rank, mem in enumerate(recalled, start=1):
                dia_id = memory_by_id.get(mem.id)
                if dia_id is not None and dia_id in evidence_set:
                    hit_rank = rank
                    break

            overall.add(hit_rank)
            raw_category_metrics[category].add(hit_rank)
            method = CATEGORY_TO_METHOD.get(category, "other")
            method_metrics[method].add(hit_rank)
        processed_dialogues += 1

    elapsed = time.time() - start
    return {
        "source": {},
        "overall": {
            "n": overall.total,
            "r@1": overall.recall(1),
            "r@5": overall.recall(5),
            "r@10": overall.recall(10),
            "mrr": overall.mrr(),
        },
        "method_breakdown": {
            method: {
                "n": stats.total,
                "r@1": stats.recall(1),
                "r@5": stats.recall(5),
                "r@10": stats.recall(10),
                "mrr": stats.mrr(),
            }
            for method, stats in sorted(method_metrics.items())
        },
        "category_breakdown": {
            category: {
                "name": CATEGORY_NAMES.get(category, f"category-{category}"),
                "n": stats.total,
                "r@1": stats.recall(1),
                "r@5": stats.recall(5),
                "r@10": stats.recall(10),
                "mrr": stats.mrr(),
            }
            for category, stats in sorted(raw_category_metrics.items())
        },
        "elapsed_seconds": elapsed,
        "num_dialogues": processed_dialogues,
    }, total_qa, overall.total


def print_results(metrics: dict, dataset_source: str, total_dialogues: int, total_qa: int, seed: int) -> None:
    print("\n" + "=" * 84)
    print("LOCOMO BENCHMARK (Synapse)")
    print("=" * 84)
    print(f"Seed: {seed}")
    print(f"Dataset source: {dataset_source}")
    print(f"Loaded: {total_dialogues} dialogues, {total_qa} QA pairs")
    print(f"Processed: {metrics['overall']['n']} QA pairs")
    print(f"Evaluated with limit k = {LOCOMO_EVAL_LIMIT}")
    print("\nOverall (Synapse @k):")
    print(f"{'Metric':<10}{'Value':>12}")
    print("-" * 22)
    print(f"{'R@1':<10}{metrics['overall']['r@1'] * 100:>11.2f}%")
    print(f"{'R@5':<10}{metrics['overall']['r@5'] * 100:>11.2f}%")
    print(f"{'R@10':<10}{metrics['overall']['r@10'] * 100:>11.2f}%")
    print(f"{'MRR':<10}{metrics['overall']['mrr'] * 100:>11.2f}%")
    print(f"Runtime: {metrics['elapsed_seconds']:.2f}s")

    print("\nPer-methodology breakdown (LOCOMO categories requested):")
    print(f"{'Method':<14}{'n':>6}{'R@1':>10}{'R@5':>10}{'R@10':>10}{'MRR':>10}")
    print("-" * 60)
    for method in ["single-hop", "multi-hop", "temporal", "open-domain", "other"]:
        if method not in metrics["method_breakdown"]:
            continue
        row = metrics["method_breakdown"][method]
        print(f"{method:<14}{row['n']:>6}{row['r@1'] * 100:>9.2f}%{row['r@5'] * 100:>10.2f}%{row['r@10'] * 100:>10.2f}%{row['mrr'] * 100:>10.2f}%")

    print("\nPer-LOCOMO category IDs:")
    print(f"{'Category':<28}{'n':>6}{'R@1':>10}{'R@5':>10}{'R@10':>10}{'MRR':>10}")
    print("-" * 74)
    for category, row in metrics["category_breakdown"].items():
        label = f"{category}: {row['name']}"
        print(f"{label:<28}{row['n']:>6}{row['r@1'] * 100:>9.2f}%{row['r@5'] * 100:>10.2f}%{row['r@10'] * 100:>10.2f}%{row['mrr'] * 100:>10.2f}%")

    print("\nProtocol:")
    print("- Fresh in-memory Synapse AI Memory per dialogue")
    print("- Embeddings explicitly disabled for deterministic lexical+index path")
    print("- One pass over all QA pairs; no shuffling")
    print("- Hit@k is true if any evidence dia_id appears in top-k recall")
    print("- MRR uses reciprocal of first hit rank among evidence items")


def write_results(metrics: dict, out_dir: Path, prefix: str = "locomo_results") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{int(time.time())}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


def sanitize_id_for_output(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark against Synapse AI Memory recall.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Pinned seed for reproducibility.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to a LoCoMo dataset json file.")
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_URL,
        help="Optional URL for locomo_data.json when local file is missing (requires --download).",
    )
    parser.add_argument("--download", action="store_true", help="Opt-in: allow remote dataset download when local file is missing.")
    parser.add_argument("--no-save", action="store_true", help="Do not write benchmark JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset, source = load_locomo_dataset(args.dataset_path, args.dataset_url, args.seed, use_download=args.download)
    normalized: List[dict] = []
    for i, dialogue in enumerate(dataset):
        normalized_dialogue = _normalize_dialogue(dialogue, i)
        if normalized_dialogue is not None:
            normalized.append(normalized_dialogue)

    if not normalized:
        raise RuntimeError("No valid dialogues found in dataset.")

    results, total_qa, processed = run_locomo(normalized, args.seed)
    results["source"] = {
        "dataset_source": source,
        "seed": args.seed,
        "limit": LOCOMO_EVAL_LIMIT,
    }
    print_results(results, source, len(normalized), total_qa, args.seed)

    if not args.no_save:
        out_file = write_results(
            results,
            ROOT / "bench",
            f"locomo_benchmark_{sanitize_id_for_output(args.seed)}",
        )
        print(f"\nSaved results to {out_file}")
    else:
        print("\nSkipping JSON output.")

    if processed == 0:
        print("[WARN] No QA pairs were processed. Check dataset schema.")


if __name__ == "__main__":
    main()
