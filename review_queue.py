"""Memory review queue — ask-before-saving mode."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse

PENDING_DIR = os.path.expanduser("~/.synapse/pending")


class ReviewQueue:
    """Wraps a Synapse instance with a pending review queue."""

    def __init__(self, synapse: "Synapse", pending_dir: str = PENDING_DIR):
        self.synapse = synapse
        self.pending_dir = pending_dir
        os.makedirs(self.pending_dir, exist_ok=True)

    _counter = 0

    def submit(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save content to pending as a JSON file. Returns the item id."""
        ts = time.time()
        ReviewQueue._counter += 1
        item_id = f"{int(ts * 1000000)}_{ReviewQueue._counter}"
        item = {
            "id": item_id,
            "content": content,
            "metadata": metadata or {},
            "submitted_at": ts,
        }
        path = os.path.join(self.pending_dir, f"{item_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        return item_id

    def list_pending(self) -> List[Dict[str, Any]]:
        """Return list of pending items."""
        items = []
        for fname in sorted(os.listdir(self.pending_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.pending_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
                items.append(item)
            except (json.JSONDecodeError, OSError):
                continue
        return items

    def approve(self, item_id: str) -> Optional[Any]:
        """Move from pending → synapse.remember(). Returns Memory or None."""
        path = os.path.join(self.pending_dir, f"{item_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        metadata = item.get("metadata") or {}
        metadata["reviewed"] = True
        metadata["review_approved_at"] = time.time()
        memory = self.synapse.remember(item["content"], metadata=metadata)
        os.remove(path)
        return memory

    def reject(self, item_id: str) -> bool:
        """Delete the pending file."""
        path = os.path.join(self.pending_dir, f"{item_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def approve_all(self) -> List[Any]:
        """Approve everything pending."""
        results = []
        for item in self.list_pending():
            memory = self.approve(item["id"])
            if memory is not None:
                results.append(memory)
        return results

    def count(self) -> int:
        """Number of pending items."""
        return len([f for f in os.listdir(self.pending_dir) if f.endswith(".json")])
