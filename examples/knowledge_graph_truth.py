"""Synapse AI Memory example: triples (KG) + contradictions + beliefs + GraphRAG recall."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse


def main() -> None:
    s = Synapse(":memory:")

    # Triples are extracted automatically on remember() (zero-LLM).
    m1 = s.remember("Alice moved to New York. Alice works at Acme Corp.", memory_type="fact")
    m2 = s.remember("Alice does not work at Acme Corp.", memory_type="fact")  # contradiction

    print("Triples for memory", m1.id)
    for t in s.triple_index.get_triples_for_memory(m1.id):
        print(
            f"- ({t.subject}, {t.predicate}, {t.object}) "
            f"polarity={t.polarity} tense={t.tense} conf={t.confidence:.2f}"
        )

    print("\nContradictions (unresolved):")
    for c in s.contradictions():
        print(
            f"- kind={c.kind} a={c.memory_id_a} b={c.memory_id_b} "
            f"conf={c.confidence:.2f} desc={c.description}"
        )

    print("\nConflict-aware recall (with disputes):")
    hits = s.recall("Where does Alice work?", retrieval_mode="graph", show_disputes=True, limit=5)
    for mem in hits:
        disputes = getattr(mem, "disputes", []) or []
        print(f"- #{mem.id}: {mem.content}")
        for d in disputes:
            print(f"  dispute -> {d.get('kind')} with #{d.get('memory_id')}: {d.get('text')}")

    print("\nBeliefs (current worldview):")
    beliefs = s.beliefs()
    for fact_key in sorted(beliefs):
        v = beliefs[fact_key]
        print(
            f"- {v.fact_key} = {v.value} (mem #{v.memory_id}) "
            f"reason={v.reason} conf={v.confidence:.2f}"
        )

    # Keep m2 referenced so linters don't complain in some environments.
    _ = m2.id


if __name__ == "__main__":
    main()
