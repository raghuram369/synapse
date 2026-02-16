"""Storage layer — append-only log with snapshots (Redis AOF/RDB style)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class MemoryMap(dict):
    """Memory map with tolerant lookup for already-resolved memory payloads."""

    def __getitem__(self, key):
        if isinstance(key, dict):
            return key
        return super().__getitem__(key)


class AppendLog:
    """Append-only log for durability (like Redis AOF)."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file_handle: Optional[object] = None
        
        # Create parent directory if needed
        if log_path != ":memory:":
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def open(self):
        """Open the log file for appending."""
        if self.log_path == ":memory:":
            return  # In-memory mode, no file I/O
            
        self._file_handle = open(self.log_path, 'a', encoding='utf-8', buffering=1)
    
    def close(self):
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def append(self, operation: str, data: Dict[str, Any]):
        """Append an operation to the log."""
        if self.log_path == ":memory:":
            return  # In-memory mode, no persistence
            
        if not self._file_handle:
            self.open()
            
        log_entry = {
            'timestamp': time.time(),
            'operation': operation,
            'data': data
        }
        
        json_line = json.dumps(log_entry, ensure_ascii=False)
        self._file_handle.write(json_line + '\n')
    
    def flush(self):
        """Force flush to disk."""
        if self._file_handle:
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())
    
    def read_all(self) -> Iterator[Dict[str, Any]]:
        """Read all entries from the log file."""
        if self.log_path == ":memory:" or not os.path.exists(self.log_path):
            return
            
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue


class Snapshot:
    """Binary snapshot for fast startup (like Redis RDB)."""
    
    def __init__(self, snapshot_path: str):
        self.snapshot_path = snapshot_path
        
        # Create parent directory if needed
        if snapshot_path != ":memory:":
            Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Dict[str, Any]):
        """Save data to snapshot file."""
        if self.snapshot_path == ":memory:":
            return  # In-memory mode, no persistence
            
        # Use JSON for human readability in V2 (can switch to binary later)
        snapshot_data = {
            'timestamp': time.time(),
            'version': '2.0',
            'data': data
        }
        
        # Atomic write using temporary file
        temp_path = self.snapshot_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic move
        os.rename(temp_path, self.snapshot_path)
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load data from snapshot file."""
        if self.snapshot_path == ":memory:" or not os.path.exists(self.snapshot_path):
            return None
            
        try:
            with open(self.snapshot_path, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)
                
            # Validate format
            if 'data' not in snapshot_data or 'version' not in snapshot_data:
                return None
                
            return snapshot_data['data']
        except (json.JSONDecodeError, IOError):
            return None


class MemoryStore:
    """
    Core storage engine with append-only log + snapshots.
    Uses .synapse file format (actually .jsonl for the log).
    """
    
    def __init__(self, path: str):
        self.path = path
        self.in_memory = path == ":memory:"
        if self.in_memory:
            self.base_path = ":memory:"
        else:
            self.base_path = os.path.dirname(os.path.abspath(path))
        
        if not self.in_memory:
            self.log_path = path + ".log"
            self.snapshot_path = path + ".snapshot"
        else:
            self.log_path = ":memory:"
            self.snapshot_path = ":memory:"
            
        self.log = AppendLog(self.log_path)
        self.snapshot = Snapshot(self.snapshot_path)
        
        # Runtime state
        self.next_memory_id = 1
        self.next_episode_id = 1
        self.memories: Dict[int, Dict[str, Any]] = MemoryMap()
        self.edges: Dict[int, Dict[str, Any]] = {}
        self.episodes: Dict[int, Dict[str, Any]] = {}
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.cards: Dict[str, Dict[str, Any]] = {}
        
        self._load_from_storage()
    
    def _load_from_storage(self):
        """Load state from snapshot + replay log."""
        # Try to load from snapshot first
        snapshot_data = self.snapshot.load()
        if snapshot_data:
            self.memories = MemoryMap(snapshot_data.get('memories', {}))
            self.edges = snapshot_data.get('edges', {})
            self.episodes = snapshot_data.get('episodes', {})
            self.concepts = snapshot_data.get('concepts', {})
            self.cards = snapshot_data.get('cards', {})
            self.next_memory_id = snapshot_data.get('next_memory_id', 1)
            self.next_episode_id = snapshot_data.get('next_episode_id', 1)
            
            # Convert string keys back to integers for memory/edge IDs
            self.memories = MemoryMap({int(k): v for k, v in self.memories.items()})
            self.edges = {int(k): v for k, v in self.edges.items()}
            self.episodes = {int(k): v for k, v in self.episodes.items()}
            for memory_data in self.memories.values():
                self._normalize_memory(memory_data)
        
        # Replay log entries on top of snapshot
        for entry in self.log.read_all():
            self._replay_operation(entry)
        
    def _normalize_memory(self, memory_data: Dict[str, Any]):
        """Ensure optional temporal fields always exist for compatibility."""
        for field in ("observed_at", "valid_from", "valid_to"):
            memory_data.setdefault(field, None)
    
    def _replay_operation(self, entry: Dict[str, Any]):
        """Replay a log operation to update in-memory state."""
        op = entry.get('operation')
        data = entry.get('data', {})
        
        if op == 'insert_memory':
            memory_id = data['id']
            self._normalize_memory(data)
            self.memories[memory_id] = data
            self.next_memory_id = max(self.next_memory_id, memory_id + 1)
            
        elif op == 'update_memory':
            memory_id = data['id']
            if memory_id in self.memories:
                # Replace the dict rather than mutating in place so callers
                # holding older references see the prior snapshot (helps tests
                # and avoids surprising aliasing behavior).
                current = self.memories[memory_id]
                updated = dict(current)
                updated.update(data)
                self._normalize_memory(updated)
                self.memories[memory_id] = updated
                
        elif op == 'delete_memory':
            memory_id = data['id']
            if memory_id in self.memories:
                del self.memories[memory_id]
                
        elif op == 'insert_edge':
            edge_id = data['id']
            self.edges[edge_id] = data
            
        elif op == 'delete_edge':
            edge_id = data['id']
            if edge_id in self.edges:
                del self.edges[edge_id]
                
        elif op == 'insert_episode':
            episode_id = data['id']
            self.episodes[episode_id] = data
            self.next_episode_id = max(self.next_episode_id, episode_id + 1)
            
        elif op == 'insert_concept':
            concept_name = data['name']
            self.concepts[concept_name] = data

        elif op == 'insert_card':
            card_id = str(data.get('card_id'))
            if card_id:
                self.cards[card_id] = data

        elif op == 'update_card':
            card_id = str(data.get('card_id', ''))
            if not card_id or card_id not in self.cards:
                return
            current = self.cards.get(card_id, {})
            updated = dict(current)
            updated.update(data)
            self.cards[card_id] = updated

        elif op == 'delete_card':
            card_id = str(data.get('card_id', ''))
            if card_id in self.cards:
                del self.cards[card_id]
    
    def insert_memory(self, memory_data: Dict[str, Any]) -> int:
        """Insert a new memory and return its ID."""
        self._normalize_memory(memory_data)
        memory_id = self.next_memory_id
        self.next_memory_id += 1
        
        memory_data['id'] = memory_id
        self.memories[memory_id] = memory_data
        
        # Log the operation
        self.log.append('insert_memory', memory_data)
        
        return memory_id
    
    def update_memory(self, memory_id: int, updates: Dict[str, Any]):
        """Update an existing memory."""
        if memory_id in self.memories:
            current = self.memories[memory_id]
            updated = dict(current)
            updated.update(updates)
            self._normalize_memory(updated)
            self.memories[memory_id] = updated
            
            # Log the operation  
            update_data = {'id': memory_id, **updates}
            self.log.append('update_memory', update_data)
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory and return success."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.log.append('delete_memory', {'id': memory_id})
            
            # Also clean up related edges
            edges_to_delete = []
            for edge_id, edge in self.edges.items():
                if edge['source_id'] == memory_id or edge['target_id'] == memory_id:
                    edges_to_delete.append(edge_id)
            
            for edge_id in edges_to_delete:
                del self.edges[edge_id]
                self.log.append('delete_edge', {'id': edge_id})
                
            return True
        return False
    
    def insert_edge(self, edge_data: Dict[str, Any]) -> int:
        """Insert a new edge and return its ID."""
        edge_id = len(self.edges) + 1  # Simple ID generation
        edge_data['id'] = edge_id
        self.edges[edge_id] = edge_data
        
        self.log.append('insert_edge', edge_data)
        return edge_id
    
    def insert_episode(self, episode_data: Dict[str, Any]) -> int:
        """Insert a new episode and return its ID."""
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        
        episode_data['id'] = episode_id
        self.episodes[episode_id] = episode_data
        
        self.log.append('insert_episode', episode_data)
        return episode_id
    
    def insert_concept(self, concept_data: Dict[str, Any]):
        """Insert a new concept."""
        concept_name = concept_data['name']
        self.concepts[concept_name] = concept_data

        self.log.append('insert_concept', concept_data)

    def insert_card(self, card_data: Dict[str, Any]):
        """Insert or replace a serialized ContextCard."""
        card_id = str(card_data.get('card_id'))
        if not card_id:
            raise ValueError("card_id is required")
        payload = dict(card_data)
        payload['card_id'] = card_id
        self.cards[card_id] = payload
        self.log.append('insert_card', payload)

    def update_card(self, card_data: Dict[str, Any]):
        """Update an existing card by id."""
        card_id = str(card_data.get('card_id', ''))
        if not card_id:
            raise ValueError("card_id is required")
        current = self.cards.get(card_id)
        if current is None:
            self.insert_card(card_data)
            return
        payload = dict(current)
        payload.update(card_data)
        payload['card_id'] = card_id
        self.cards[card_id] = payload
        self.log.append('update_card', payload)

    def delete_card(self, card_id: str):
        """Delete a stored card by id."""
        card_key = str(card_id)
        if card_key in self.cards:
            del self.cards[card_key]
            self.log.append('delete_card', {'card_id': card_key})
    
    def flush(self):
        """Force flush log to disk."""
        self.log.flush()
    
    def create_snapshot(self):
        """Create a compacted snapshot."""
        snapshot_data = {
            'memories': {str(k): v for k, v in self.memories.items()},  # JSON keys must be strings
            'edges': {str(k): v for k, v in self.edges.items()},
            'episodes': {str(k): v for k, v in self.episodes.items()},
            'concepts': self.concepts,
            'cards': self.cards,
            'next_memory_id': self.next_memory_id,
            'next_episode_id': self.next_episode_id
        }
        
        self.snapshot.save(snapshot_data)
        
        # After successful snapshot, could truncate log (not implemented for safety)
        
    def close(self):
        """Close the storage engine."""
        self.flush()
        self.log.close()
