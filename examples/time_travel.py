"""Synapse AI Memory example: time-travel queries and fact evolution chains."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse


def remember_at(synapse, content, iso, memory_type="event"):
    memory = synapse.remember(content, memory_type=memory_type)
    ts = int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())
    synapse.store.update_memory(memory.id, {"created_at": ts, "last_accessed": ts})
    return memory


def show_chain(synapse, query):
    print(f"\nHistory: {query}")
    for step in synapse.fact_history(query):
        print(f"  - {step['memory'].content}")


def main():
    s = Synapse(":memory:")

    # changing facts over time
    remember_at(s, "User lives in Boston", "2021-06-01")
    remember_at(s, "User lives in Seattle", "2024-03-12")
    remember_at(s, "User lives in Berlin", "2024-12-10")
    remember_at(s, "User worked as junior developer", "2022-01-15", memory_type="fact")
    remember_at(s, "User became staff engineer", "2024-08-01", memory_type="fact")

    show_chain(s, "where does the user live")
    show_chain(s, "what is the user's job")

    past = s.recall("where did user live in March 2024?", memory_type="event", temporal="2024-03", limit=1)
    if past:
        print(f"\nMarch 2024 location: {past[0].content}")

    # consolidate repeated preferences
    for line in [
        "User likes dark coffee every morning.",
        "User takes dark coffee in the morning every day.",
        "User starts mornings with dark coffee.",
    ]:
        s.remember(line, memory_type="preference")
    clusters = s.consolidate(min_cluster_size=3, similarity_threshold=0.35)
    for c in clusters:
        print(f"\nConsolidated {c['source_count']} repeated preference patterns")


if __name__ == "__main__":
    main()
