# Synapse AI Memory Reproducible Benchmarks

This `bench/` suite contains two standalone benchmark runners:

- `run_locomo.py` for LOCOMO-style long-context recall
- `run_practical.py` for practical multi-session memory behavior

Both scripts are designed to be run from the `bench/` directory.

## Setup

```bash
cd bench
pip install -r requirements.txt
```

No API keys, network model calls, or embedding services are used.

## Reproducibility

Both scripts are pinned to a deterministic seed by default (`--seed 2026`) and use
an in-memory Synapse AI Memory store where possible.

- `PYTHONHASHSEED` is set inside each runner.
- Python `random` is seeded.
- Evaluation loops are deterministic (no shuffle, fixed query order, fixed `k=10`).
- Synapse AI Memory embeddings are disabled in benchmark runs: `synapse._use_embeddings = False`.

You can still override the seed:

```bash
python run_locomo.py --seed 12345
python run_practical.py --seed 12345
```

## run_locomo.py

### What it evaluates

LOCOMO-style long-horizon memory recall over dialogue turns.

### Dataset

The script supports three modes:

1. **Load local LOCOMO file**: if `../locomo_data.json` exists, it is loaded.
2. **Download from URL** (opt-in): if local file is missing and `--download` is set, it downloads
   `locomo_data.json` from `--dataset-url`.
3. **Fallback sample**: if both local and download are unavailable (or download is not enabled), it uses a tiny reproducible
   built-in sample dataset.

### Evaluation protocol

- One Synapse AI Memory instance per dialogue, created fresh from scratch.
- Every available turn from `session_*` keys is inserted in order.
- Each QA is answered via `Synapse.recall(query, limit=10)`.
- A QA is counted as a hit if any returned result contains evidence `dia_id` from the official `evidence` list.
- Metrics reported:
  - `R@1`, `R@5`, `R@10`
  - `MRR`
  - per-category and per-methodology breakdowns

LOCOMO category mapping used for reporting:

- `1`: single-hop
- `2`: temporal
- `3`: multi-hop
- `4` + `5`: open-domain

### Run

```bash
cd bench
python run_locomo.py
```

Helpful flags:

- `--dataset-path PATH`: explicit dataset file
- `--download`: opt-in to remote fetch fallback
- `--dataset-url URL`: download URL (used only with `--download`)
- `--no-save`: skip JSON artifact

Default artifact: `bench/locomo_benchmark_<seed>_<timestamp>.json`

## run_practical.py

### What it evaluates

A practical synthetic benchmark with realistic conversations across sessions:

- factual memories
- temporal recall
- multi-hop reasoning
- preference tracking
- correction/supersession behavior

### Evaluation protocol

- Insert all inline session memories into a fresh in-memory Synapse AI Memory instance.
- Query with the inline practical dataset questions.
- Recall metric is based on first hit rank in `top-10`.
- Metrics reported:
  - `R@1`, `R@5`, `R@10`, `MRR`
  - by category (`facts`, `temporal`, `preference`, `multi-hop`, `correction`)

### Run

```bash
cd bench
python run_practical.py
```

Defaults to saving JSON to:

- `bench/practical_benchmark_<seed>_<timestamp>.json`

## Quick checks

```bash
cd bench
python run_practical.py
python run_locomo.py
```
