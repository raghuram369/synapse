"""Synapse AI Memory example: compile ContextPack for LLM integration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse


def main() -> None:
    s = Synapse(":memory:")

    s.remember("User is vegetarian and allergic to shellfish", memory_type="preference")
    s.remember("User lives in Denver, CO", memory_type="fact", valid_from="2024-06-01")
    s.remember("User lived in Austin, TX", memory_type="fact", valid_from="2024-01-01", valid_to="2024-06-01")
    s.remember("User likes spicy ramen", memory_type="preference")

    pack = s.compile_context(
        "Recommend a safe restaurant and remind me where the user lives.",
        budget=1400,
        policy="balanced",
    )

    print("\nContextPack.to_compact():")
    print(pack.to_compact())

    print("\nContextPack.to_system_prompt():")
    print(pack.to_system_prompt())

    print("\nContextPack.to_dict() keys:", sorted(pack.to_dict().keys()))


if __name__ == "__main__":
    main()

