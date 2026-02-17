from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


RECEIPT_FILE = "permit_receipts.jsonl"
RECEIPTS_SUBDIR = "receipts"
DEFAULT_RECEIPT_ROOT = os.path.expanduser("~/.synapse")


def _to_text(value: Any, *, null_if_missing: bool = True) -> str | None:
    if value is None:
        return None if null_if_missing else ""
    if value == "":
        return ""
    return str(value)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_receipt_root(db_path: str | None) -> str:
    """Resolve receipt storage root for a Synapse DB path."""
    if not db_path or db_path == ":memory:":
        return DEFAULT_RECEIPT_ROOT

    expanded = os.path.expanduser(db_path)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(expanded))

    if candidate.is_dir():
        return str(candidate)
    return str(candidate.parent)


def resolve_receipt_path(db_path: str | None = None) -> str:
    root = Path(resolve_receipt_root(db_path))
    return str(root / RECEIPTS_SUBDIR / RECEIPT_FILE)


def _candidate_receipt_paths(db_path: str | None) -> list[str]:
    root = Path(resolve_receipt_root(db_path))
    return [
        str(root / RECEIPT_FILE),
        str(root / RECEIPTS_SUBDIR / RECEIPT_FILE),
    ]


def write_receipt(payload: Dict[str, Any], db_path: str | None = None) -> None:
    path = Path(resolve_receipt_path(db_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    if not record.get("timestamp"):
        record["timestamp"] = _timestamp_utc()
    if not record.get("receipt_id"):
        record["receipt_id"] = f"permit-{int(datetime.now(timezone.utc).timestamp() * 1000)}"

    try:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")
    except OSError:
        return


def load_receipts(last: int = 3, db_path: str | None = None) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    source = ""

    for path in _candidate_receipt_paths(db_path):
        if not path or not os.path.exists(path):
            continue
        source = path
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    records.append(raw)
        break

    if last > 0:
        records = records[-int(last) :]
    return records, source


def normalize_receipt(record: Dict[str, Any]) -> Dict[str, Any]:
    counts = record.get("memory_counts")
    if not isinstance(counts, dict):
        counts = {}

    matched = record.get("matched_rules")
    if not isinstance(matched, list):
        matched = []

    reasons = record.get("block_reasons")
    if not isinstance(reasons, list):
        reasons = []

    return {
        "receipt_id": _to_text(record.get("receipt_id")),
        "decision": _to_text(record.get("decision"), null_if_missing=True) or "unknown",
        "actor_id": _to_text(record.get("actor_id")),
        "app_id": _to_text(record.get("app_id")),
        "purpose": _to_text(record.get("purpose")),
        "scope_requested": _to_text(record.get("scope_requested")),
        "scope_applied": _to_text(record.get("scope_applied")),
        "policy_id": _to_text(record.get("policy_id")),
        "matched_rules": [str(item) for item in matched],
        "memory_counts": {
            "considered": int(counts.get("considered", 0) or 0),
            "returned": int(counts.get("returned", 0) or 0),
            "blocked": int(counts.get("blocked", 0) or 0),
        },
        "block_reasons": [str(item) for item in reasons],
        "timestamp": _to_text(record.get("timestamp")),
    }
