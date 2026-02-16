"""Natural Language Forget - Pattern matching and fuzzy search for plain English forgetting."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse, Memory


class ForgetPatternMatcher:
    """Pattern matcher for natural language forget commands."""
    
    def __init__(self):
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> List[Dict[str, Any]]:
        """Build pattern matching rules for different forget types."""
        return [
            {
                "type": "time_based",
                "patterns": [
                    r"^forget (?:everything|anything|things|memories) older than (\d+) (day|week|month|year)s?$",
                    r"^delete (?:everything|anything|things|memories) older than (\d+) (day|week|month|year)s?$",
                    r"^(?:forget|delete|remove) (?:everything|anything|things|memories) older than (.+)$",
                    r"^remove (?:everything|anything|things|memories) from (?:more than |over )?(\d+) (day|week|month|year)s? ago$",
                    r"^(?:forget|delete|remove) (?:everything|anything|things|memories) from (?:before|prior to) (.+)$",
                ],
                "examples": ["forget everything older than 30 days"]
            },
            {
                "type": "by_memory_type",
                "patterns": [
                    r"^(?:forget|delete) (?:all|my) (preference|fact|event|skill|observation)s?(?: about (.+))?$",
                ],
                "examples": ["forget all preferences about food"]
            },
            {
                "type": "topic_bulk",
                "patterns": [
                    r"^(?:forget|delete|remove) (?:everything|anything|all) (?:about |regarding |related to )(.+)$",
                ],
                "examples": ["forget everything about my old job"]
            },
            {
                "type": "update",
                "patterns": [
                    r"^(?:that changed|that's different now|update that|actually)\b.*$",
                    r"^(?:correction|fix|change)\b.*$",
                    r"^not .+ (?:anymore|any more), (?:now|but) .+$",
                ],
                "examples": ["that changed, I moved to Seattle"]
            },
            {
                "type": "specific_fact",
                "patterns": [
                    r"^(?:forget|delete|remove|don't remember) (.+)$",
                ],
                "examples": ["forget my phone number"]
            },
        ]
    
    def parse_forget_command(self, command: str) -> Dict[str, Any]:
        """Parse a natural language forget command."""
        command = command.strip().lower()
        
        for pattern_group in self.patterns:
            for pattern in pattern_group["patterns"]:
                match = re.match(pattern, command, re.IGNORECASE)
                if match:
                    return {
                        "type": pattern_group["type"],
                        "matches": match.groups(),
                        "raw_command": command,
                        "matched_pattern": pattern
                    }
        
        # Fallback: treat as specific fact search
        return {
            "type": "specific_fact",
            "matches": [command.replace("forget", "").replace("delete", "").replace("remove", "").strip()],
            "raw_command": command,
            "matched_pattern": "fallback"
        }
    
    def extract_time_constraint(self, time_str: str) -> Optional[float]:
        """Extract time constraint from natural language."""
        time_str = time_str.lower().strip()
        
        # Handle relative time (e.g., "30 days", "2 weeks")
        relative_match = re.match(r"(\d+)\s*(day|week|month|year)s?", time_str)
        if relative_match:
            value, unit = relative_match.groups()
            value = int(value)
            
            if unit.startswith("day"):
                delta = timedelta(days=value)
            elif unit.startswith("week"):
                delta = timedelta(weeks=value)
            elif unit.startswith("month"):
                delta = timedelta(days=value * 30)  # Approximate
            elif unit.startswith("year"):
                delta = timedelta(days=value * 365)  # Approximate
            else:
                return None
            
            return time.time() - delta.total_seconds()
        
        # Handle specific dates (basic parsing)
        date_patterns = [
            r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",  # Month YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, time_str, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3 and match.group(1).isdigit():
                        # YYYY-MM-DD or MM/DD/YYYY
                        if len(match.group(1)) == 4:  # YYYY-MM-DD
                            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        else:  # MM/DD/YYYY
                            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        date = datetime(year, month, day)
                        return date.timestamp()
                    elif len(match.groups()) == 2:  # Month YYYY
                        month_name, year = match.groups()
                        month_num = [
                            "january", "february", "march", "april", "may", "june",
                            "july", "august", "september", "october", "november", "december"
                        ].index(month_name.lower()) + 1
                        date = datetime(int(year), month_num, 1)
                        return date.timestamp()
                except (ValueError, IndexError):
                    continue
        
        return None


class NaturalForget:
    """Natural language forget processor using existing Synapse search and deletion."""
    
    def __init__(self, synapse: "Synapse"):
        self.synapse = synapse
        self.matcher = ForgetPatternMatcher()
    
    def process_forget_command(self, command: str, confirm: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """Process a natural language forget command."""
        parsed = self.matcher.parse_forget_command(command)
        
        if parsed["type"] == "specific_fact":
            return self._forget_specific_fact(parsed, confirm, dry_run)
        elif parsed["type"] == "topic_bulk":
            return self._forget_topic_bulk(parsed, confirm, dry_run)
        elif parsed["type"] == "time_based":
            return self._forget_time_based(parsed, confirm, dry_run)
        elif parsed["type"] == "update":
            return self._forget_update(parsed, confirm, dry_run)
        elif parsed["type"] == "by_memory_type":
            return self._forget_by_memory_type(parsed, confirm, dry_run)
        else:
            return {
                "status": "error",
                "message": f"Unknown forget type: {parsed['type']}",
                "parsed": parsed
            }
    
    def _forget_specific_fact(self, parsed: Dict[str, Any], confirm: bool, dry_run: bool) -> Dict[str, Any]:
        """Forget specific facts by searching for them."""
        if not parsed["matches"] or not parsed["matches"][0]:
            return {"status": "error", "message": "No search term specified"}
        
        search_term = parsed["matches"][0].strip()
        if not search_term:
            return {"status": "error", "message": "Empty search term"}
        
        term = search_term.lower()
        matches = []
        recalled = self.synapse.recall(search_term, limit=200)
        for memory in recalled:
            if term in memory.content.lower():
                matches.append((memory.id, memory.content))
        # Also do a direct content scan to catch anything recall might miss
        for memory_id, memory_data in self.synapse.store.memories.items():
            content = str(memory_data.get("content", ""))
            if term in content.lower() and not any(m[0] == memory_id for m in matches):
                matches.append((memory_id, content))

        if not matches:
            return {
                "status": "not_found",
                "message": f"No memories found matching '{search_term}'",
                "search_term": search_term,
            }

        if dry_run:
            return {
                "status": "dry_run",
                "message": f"Would delete {len(matches)} memories",
                "memories": [{"id": mid, "content": text[:100] + "..."} for mid, text in matches],
                "search_term": search_term,
            }

        deleted_ids = []
        for memory_id, _content in matches:
            self.synapse.forget(memory_id)
            deleted_ids.append(memory_id)
        
        return {
            "status": "deleted",
            "message": f"Deleted {len(deleted_ids)} memories matching '{search_term}'",
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "search_term": search_term
        }
    
    def _forget_topic_bulk(self, parsed: Dict[str, Any], confirm: bool, dry_run: bool) -> Dict[str, Any]:
        """Forget everything about a topic using concept-based search."""
        if not parsed["matches"] or not parsed["matches"][0]:
            return {"status": "error", "message": "No topic specified"}
        
        topic = parsed["matches"][0].strip()
        
        # Use recall for both dry_run and actual delete for consistent matching
        memories = self.synapse.recall(topic, limit=100)
        
        if dry_run:
            return {
                "status": "dry_run",
                "message": f"Would delete {len(memories)} memories about '{topic}'",
                "memories": [{"id": m.id, "content": m.content[:100] + "..."} for m in memories],
                "topic": topic
            }
        
        deleted_ids = []
        failures = 0
        for m in memories:
            try:
                self.synapse.forget(m.id)
                deleted_ids.append(m.id)
            except Exception:
                failures += 1
        
        result_info: Dict[str, Any] = {
            "status": "deleted",
            "message": f"Deleted {len(deleted_ids)} memories about '{topic}'",
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "topic": topic,
        }
        if failures:
            result_info["failures"] = failures
        return result_info
    
    def _forget_time_based(self, parsed: Dict[str, Any], confirm: bool, dry_run: bool) -> Dict[str, Any]:
        """Forget memories based on time constraints."""
        matches = parsed["matches"]
        
        if parsed["matched_pattern"].find("older than") != -1:
            # Pattern: "older than X days/weeks/months/years"
            if len(matches) >= 2:
                time_value, time_unit = matches[0], matches[1]
                time_str = f"{time_value} {time_unit}"
            else:
                return {"status": "error", "message": "Could not parse time constraint"}
        elif len(matches) >= 1:
            # Pattern: "from before DATE" or similar
            time_str = matches[0]
        else:
            return {"status": "error", "message": "No time constraint found"}
        
        cutoff_timestamp = self.matcher.extract_time_constraint(time_str)
        if not cutoff_timestamp:
            return {"status": "error", "message": f"Could not parse time: '{time_str}'"}
        
        # Find memories older than cutoff
        old_memories = []
        for memory_id, memory_data in self.synapse.store.memories.items():
            created_at = memory_data.get("created_at", 0)
            if created_at < cutoff_timestamp:
                old_memories.append(memory_id)
        
        if dry_run:
            return {
                "status": "dry_run",
                "message": f"Would delete {len(old_memories)} memories older than {time_str}",
                "memory_count": len(old_memories),
                "cutoff_date": datetime.fromtimestamp(cutoff_timestamp).isoformat(),
                "time_constraint": time_str
            }
        
        # Delete old memories
        deleted_count = 0
        failures = 0
        for memory_id in old_memories:
            try:
                self.synapse.forget(memory_id)
                deleted_count += 1
            except Exception:
                failures += 1
                continue
        
        result: Dict[str, Any] = {
            "status": "deleted",
            "message": f"Deleted {deleted_count} memories older than {time_str}",
            "deleted_count": deleted_count,
            "cutoff_date": datetime.fromtimestamp(cutoff_timestamp).isoformat(),
            "time_constraint": time_str
        }
        if failures:
            result["failures"] = failures
            result["message"] += f" ({failures} failures)"
        return result
    
    def _forget_update(self, parsed: Dict[str, Any], confirm: bool, dry_run: bool) -> Dict[str, Any]:
        """Handle update/correction commands."""
        # This is more complex - would need to identify what to update and with what
        # For now, return guidance
        return {
            "status": "guidance",
            "message": "Update commands need more specific implementation. Try: 'forget [old info]' then remember the new info.",
            "suggestion": "Use separate forget and remember commands for updates",
            "parsed": parsed
        }
    
    def _forget_by_memory_type(self, parsed: Dict[str, Any], confirm: bool, dry_run: bool) -> Dict[str, Any]:
        """Forget memories by type with optional topic filter."""
        matches = parsed["matches"]
        if not matches or not matches[0]:
            return {"status": "error", "message": "No memory type specified"}
        
        memory_type = matches[0].strip()
        topic_filter = matches[1].strip() if len(matches) > 1 and matches[1] else None
        
        # Find memories by type
        matching_memories = []
        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get("memory_type") == memory_type:
                if not topic_filter:
                    matching_memories.append(memory_id)
                else:
                    # Apply topic filter
                    content = memory_data.get("content", "").lower()
                    if topic_filter.lower() in content:
                        matching_memories.append(memory_id)
        
        if dry_run:
            filter_text = f" about '{topic_filter}'" if topic_filter else ""
            return {
                "status": "dry_run", 
                "message": f"Would delete {len(matching_memories)} {memory_type} memories{filter_text}",
                "memory_count": len(matching_memories),
                "memory_type": memory_type,
                "topic_filter": topic_filter
            }
        
        # Delete matching memories
        deleted_count = 0
        failures = 0
        for memory_id in matching_memories:
            try:
                self.synapse.forget(memory_id)
                deleted_count += 1
            except Exception:
                failures += 1
                continue
        
        filter_text = f" about '{topic_filter}'" if topic_filter else ""
        result: Dict[str, Any] = {
            "status": "deleted",
            "message": f"Deleted {deleted_count} {memory_type} memories{filter_text}",
            "deleted_count": deleted_count,
            "memory_type": memory_type,
            "topic_filter": topic_filter
        }
        if failures:
            result["failures"] = failures
            result["message"] += f" ({failures} failures)"
        return result
    
    def suggest_forget_commands(self) -> List[Dict[str, str]]:
        """Suggest example forget commands."""
        return [
            {"command": "forget my phone number", "description": "Delete specific information"},
            {"command": "forget everything about my old job", "description": "Delete all memories about a topic"},
            {"command": "forget memories older than 30 days", "description": "Delete old memories by age"},
            {"command": "forget all preferences about food", "description": "Delete memories by type and topic"},
            {"command": "delete anything related to Sarah", "description": "Topic-based bulk deletion"},
        ]