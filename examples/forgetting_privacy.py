"""Synapse AI Memory example: TTL, forget_topic, redaction, and GDPR delete."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse


def main() -> None:
    s = Synapse(":memory:")

    pii = s.remember(
        "User SSN is 123-45-6789",
        memory_type="fact",
        metadata={"tags": ["user:42", "pii", "temporary"], "ttl_days": 7},
    )
    s.remember("User likes hiking in Colorado", memory_type="preference", metadata={"tags": ["user:42"]})

    print("Before redaction:", s.recall("ssn", limit=1)[0].content)
    print("Redact:", s.redact(memory_id=pii.id, fields=["content"]))
    print("After redaction:", s.recall("ssn", limit=1)[0].content)

    # Declarative retention rules (example: delete temporary after 7 days).
    rules_report = s.set_retention_rules([{"tag": "temporary", "ttl_days": 7, "action": "delete"}])
    print("Retention rules report:", rules_report)

    # Topic/concept deletion.
    print("Forget topic 'pii':", s.forget_topic("pii"))

    # GDPR-style delete for a user tag.
    print("GDPR delete user 42:", s.gdpr_delete(user_id="42"))


if __name__ == "__main__":
    main()

