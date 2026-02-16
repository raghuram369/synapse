"""Memory Inbox - Enhanced review queue with auto-approve rules and integration."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse, Memory

from review_queue import ReviewQueue


class AutoApproveRules:
    """Configurable auto-approve rules for memory inbox."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._default_config()
        if config:
            self._merge_config(config)

    def _merge_config(self, custom: Dict[str, Any]):
        """Deep-merge custom configuration into defaults."""
        for key, value in custom.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _default_config(self) -> Dict[str, Any]:
        """Default auto-approve rules."""
        return {
            "enabled": True,
            "preferences": {
                "enabled": True,
                "keywords": ["prefer", "like", "favorite", "love", "hate", "dislike"]
            },
            "goals": {
                "enabled": True,
                "keywords": ["goal", "want to", "plan to", "intend to", "aim to"]
            },
            "facts": {
                "enabled": True,
                "min_confidence": 0.8,
                "exclude_patterns": ["maybe", "might", "possibly", "perhaps", "unclear", "not sure"]
            },
            "personal_info": {
                "enabled": False,  # Require explicit approval for sensitive data
                "patterns": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{3}-\d{3}-\d{4}\b"]  # SSN, phone
            }
        }
    
    def should_auto_approve(self, content: str, memory_type: str = "fact", 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a memory should be auto-approved."""
        if not self.config.get("enabled", True):
            return False
        
        metadata = metadata or {}
        content_lower = content.lower()
        
        # Check for sensitive personal info patterns (blocking)
        patterns = self.config.get("personal_info", {}).get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False  # Always block potential PII
        
        # Check for uncertainty patterns first (blocking)
        exclude_patterns = self.config.get("facts", {}).get("exclude_patterns", [])
        if any(pattern in content_lower for pattern in exclude_patterns):
            return False
        
        # Auto-approve preferences
        if self.config.get("preferences", {}).get("enabled", True):
            keywords = self.config.get("preferences", {}).get("keywords", [])
            if memory_type == "preference" or any(keyword in content_lower for keyword in keywords):
                return True

        # Auto-approve goals
        if self.config.get("goals", {}).get("enabled", True):
            keywords = self.config.get("goals", {}).get("keywords", [])
            if any(keyword in content_lower for keyword in keywords):
                return True

        # For facts, stay conservative: require manual review unless clearly
        # tagged as preferences/goals above.
        if memory_type == "fact" and self.config.get("facts", {}).get("enabled", True):
            return False

        return False


class MemoryInbox:
    """Enhanced memory inbox with auto-approve rules and better integration."""
    
    def __init__(self, synapse: "Synapse", pending_dir: Optional[str] = None, 
                 auto_approve_config: Optional[Dict[str, Any]] = None):
        self.synapse = synapse
        self.review_queue = ReviewQueue(synapse, pending_dir) if pending_dir else ReviewQueue(synapse)
        self.auto_approve_rules = AutoApproveRules(auto_approve_config)
        self._pending_cache: Optional[List[Dict[str, Any]]] = None
        self._bypass_lock = threading.Lock()

    def _remember_direct(self, **kwargs):
        """Store memory bypassing inbox routing to avoid re-queue loops."""
        with self._bypass_lock:
            original_mode = getattr(self.synapse, "_inbox_mode", False)
            original_inbox = getattr(self.synapse, "_inbox", None)
            try:
                self.synapse._inbox_mode = False
                self.synapse._inbox = None
                return self.synapse.remember(**kwargs)
            finally:
                self.synapse._inbox_mode = original_mode
                self.synapse._inbox = original_inbox
    
    def submit(self, content: str, memory_type: str = "fact", 
               metadata: Optional[Dict[str, Any]] = None,
               scope: str = "private", force_review: bool = False) -> Dict[str, Any]:
        """Submit a memory to the inbox, potentially auto-approving it."""
        
        # Check if should auto-approve
        if not force_review and self.auto_approve_rules.should_auto_approve(content, memory_type, metadata):
            # Auto-approve: store directly
            try:
                memory = self._remember_direct(
                    content=content,
                    memory_type=memory_type,
                    metadata={**(metadata or {}), "auto_approved": True, "inbox_processed": True},
                    scope=scope
                )
                return {
                    "status": "auto_approved",
                    "memory_id": memory.id,
                    "item_id": None,
                    "reason": "matched_auto_approve_rules"
                }
            except Exception as e:
                # Fall back to inbox if auto-approve fails
                pass
        
        # Submit to pending inbox
        enriched_metadata = {
            **(metadata or {}),
            "memory_type": memory_type,
            "scope": scope,
            "submitted_at": time.time(),
            "requires_approval": True,
            "auto_approve_checked": True
        }
        
        item_id = self.review_queue.submit(content, enriched_metadata)
        self._invalidate_cache()
        
        return {
            "status": "pending",
            "memory_id": None,
            "item_id": item_id,
            "reason": "requires_manual_review"
        }
    
    def list_pending(self, include_queryable: bool = False) -> List[Dict[str, Any]]:
        """List all pending memories in inbox."""
        if self._pending_cache is None:
            self._pending_cache = self.review_queue.list_pending()
        
        pending = self._pending_cache.copy()
        
        # Add queryable flag if requested
        if include_queryable:
            for item in pending:
                item["queryable"] = True
                item["status"] = "pending"
        
        return pending
    
    def approve(self, item_id: str) -> Optional["Memory"]:
        """Approve a pending memory and store it permanently."""
        path = os.path.join(self.review_queue.pending_dir, f"{item_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)

        metadata = item.get("metadata") or {}
        metadata["reviewed"] = True
        metadata["review_approved_at"] = time.time()

        memory = self._remember_direct(
            content=item["content"],
            memory_type=metadata.get("memory_type", "fact"),
            scope=metadata.get("scope", "private"),
            metadata=metadata,
        )
        os.remove(path)
        self._invalidate_cache()
        return memory
    
    def reject(self, item_id: str) -> bool:
        """Reject and delete a pending memory."""
        result = self.review_queue.reject(item_id)
        self._invalidate_cache()
        return result
    
    def redact(self, item_id: str, redacted_content: str) -> Optional["Memory"]:
        """Redact content of a pending memory and then approve it."""
        # Get the original item
        path = os.path.join(self.review_queue.pending_dir, f"{item_id}.json")
        if not os.path.exists(path):
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        
        # Update content and approve — never persist original content
        original_metadata = item.get("metadata", {})
        redaction_metadata = {
            **original_metadata,
            "redacted": True,
            "redacted_at": time.time(),
            "reviewed": True,
            "review_approved_at": time.time()
        }
        
        memory = self._remember_direct(
            content=redacted_content,
            memory_type=original_metadata.get("memory_type", "fact"),
            scope=original_metadata.get("scope", "private"),
            metadata=redaction_metadata
        )
        
        # Remove from pending
        os.remove(path)
        self._invalidate_cache()
        
        return memory
    
    def pin(self, item_id: str) -> Optional["Memory"]:
        """Pin a pending memory (approve with high importance)."""
        # Get the original item to add pin metadata
        path = os.path.join(self.review_queue.pending_dir, f"{item_id}.json")
        if not os.path.exists(path):
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        
        # Add pin metadata
        pin_metadata = {
            **item.get("metadata", {}),
            "pinned": True,
            "pinned_at": time.time(),
            "importance": "high",
            "reviewed": True,
            "review_approved_at": time.time()
        }
        
        memory = self._remember_direct(
            content=item["content"],
            memory_type=pin_metadata.get("memory_type", "fact"),
            scope=pin_metadata.get("scope", "private"),
            metadata=pin_metadata
        )
        
        # Remove from pending
        os.remove(path)
        self._invalidate_cache()
        
        return memory
    
    def query_pending(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through pending memories (they're queryable but marked as pending)."""
        pending = self.list_pending()
        results = []
        
        query_lower = query.lower()
        for item in pending:
            content_lower = item.get("content", "").lower()
            if query_lower in content_lower:
                # Add search relevance score
                item_copy = item.copy()
                item_copy["search_score"] = content_lower.count(query_lower)
                item_copy["status"] = "pending"
                results.append(item_copy)
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("search_score", 0), reverse=True)
        return results[:limit]
    
    def count(self) -> Dict[str, int]:
        """Count pending memories and related stats."""
        pending = self.list_pending()
        
        counts = {
            "total_pending": len(pending),
            "auto_approved_today": 0,
            "manually_approved_today": 0,
            "rejected_today": 0
        }
        
        # Count would need tracking in a separate log, but this provides basic info
        return counts
    
    def _invalidate_cache(self):
        """Invalidate the pending items cache."""
        self._pending_cache = None
    
    def get_auto_approve_config(self) -> Dict[str, Any]:
        """Get current auto-approve configuration."""
        return self.auto_approve_rules.config
    
    def update_auto_approve_config(self, config: Dict[str, Any]):
        """Update auto-approve rules configuration."""
        self.auto_approve_rules.config.update(config)