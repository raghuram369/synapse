"""Synapse AI Memory example: personal preference recall with score breakdown."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse


def main():
    print("Synapse AI Memory Example: Personal Preferences")
    syn = Synapse(":memory:")

    syn.remember("User prefers dark mode in UI", memory_type="preference")
    syn.remember("User prefers French as the interface language", memory_type="preference")
    syn.remember("User has lactose and peanut dietary restrictions", memory_type="preference")
    syn.remember("User prefers jazz and lo-fi music for focus", memory_type="preference")

    for prompt in [
        "What does the user prefer for display?",
        "What food rules apply?",
        "Which music does the user like?",
    ]:
        print(f"\nQ: {prompt}")
        hits = syn.recall(prompt, limit=2, memory_type="preference", explain=True)
        if not hits:
            print("  no result")
            continue
        for mem in hits:
            b = mem.score_breakdown
            print(f"- #{mem.id}: {mem.content}")
            print(
                f"  breakdown: lex={b.bm25_score:.2f}, concept={b.concept_score:.2f}, "
                f"temporal={b.temporal_score:.2f}, episode={b.episode_score:.2f}, "
                f"src={','.join(b.match_sources)}"
            )


if __name__ == "__main__":
    main()
