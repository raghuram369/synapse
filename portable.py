"""
Synapse Portable Memory Format (.synapse)

Binary file format for exporting, importing, and merging Synapse memory databases.
Zero external dependencies. Pure Python.

See FORMAT.md for the binary format specification.
"""

import json
import struct
import time
import zlib
import io
import os
import re
import math
from typing import Optional, List, Dict, Any, Iterator, Tuple, Set
from pathlib import Path
from collections import defaultdict


# ─── Constants ───────────────────────────────────────────────────────────────

MAGIC = b'\x89SYN'
FORMAT_VERSION_MAJOR = 1
FORMAT_VERSION_MINOR = 0
HEADER_SIZE = 32

# Flags
FLAG_PARTIAL = 0x01
FLAG_COMPRESSED = 0x02
FLAG_HAS_PROVENANCE = 0x04

# Section type IDs
SECTION_MEMORIES = 0x01
SECTION_EDGES = 0x02
SECTION_CONCEPTS = 0x03
SECTION_EPISODES = 0x04
SECTION_METADATA = 0x05

# Record types
RECORD_DATA = 0x01
RECORD_END = 0xFF

# Section directory entry size
DIR_ENTRY_SIZE = 20


# ─── Low-level binary helpers ────────────────────────────────────────────────

def _pack_header(flags: int, timestamp: float, num_sections: int, crc: int) -> bytes:
    """Pack the 32-byte file header."""
    ts_usec = int(timestamp * 1_000_000)
    return struct.pack('>4sBBHQIII',
        MAGIC,
        FORMAT_VERSION_MAJOR,
        FORMAT_VERSION_MINOR,
        flags,
        ts_usec,
        num_sections,
        crc,
        0  # reserved 4 bytes (padding to fill 32)
    ) + b'\x00' * 4  # remaining reserved bytes


def _unpack_header(data: bytes) -> Dict[str, Any]:
    """Unpack and validate the 32-byte file header."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)} bytes")
    
    magic = data[0:4]
    if magic != MAGIC:
        # Check for JSON fallback
        if data[0:1] == b'{':
            return {'json_fallback': True}
        raise ValueError(f"Invalid magic bytes: {magic!r} (expected {MAGIC!r})")
    
    major, minor, flags, ts_usec, num_sections, crc, _ = struct.unpack('>BBHQIII', data[4:28])
    
    if major > FORMAT_VERSION_MAJOR:
        raise ValueError(f"Unsupported format version {major}.{minor} (reader supports up to {FORMAT_VERSION_MAJOR}.x)")
    
    return {
        'json_fallback': False,
        'version_major': major,
        'version_minor': minor,
        'flags': flags,
        'timestamp': ts_usec / 1_000_000,
        'num_sections': num_sections,
        'crc': crc,
    }


def _pack_dir_entry(section_type: int, offset: int, length: int) -> bytes:
    """Pack a 20-byte section directory entry."""
    return struct.pack('>HHQQ', section_type, 0, offset, length)


def _unpack_dir_entry(data: bytes) -> Dict[str, int]:
    """Unpack a 20-byte section directory entry."""
    section_type, _, offset, length = struct.unpack('>HHQQ', data)
    return {'type': section_type, 'offset': offset, 'length': length}


def _write_record(f, record_type: int, payload: bytes):
    """Write a TLV record."""
    f.write(struct.pack('>BI', record_type, len(payload)))
    f.write(payload)


def _read_record(f) -> Optional[Tuple[int, bytes]]:
    """Read a TLV record. Returns (type, payload) or None at EOF."""
    header = f.read(5)
    if len(header) < 5:
        return None
    record_type, length = struct.unpack('>BI', header)
    if record_type == RECORD_END:
        return (RECORD_END, b'')
    payload = f.read(length)
    if len(payload) < length:
        raise ValueError(f"Truncated record: expected {length} bytes, got {len(payload)}")
    return (record_type, payload)


def _iter_records(f, section_offset: int, section_length: int) -> Iterator[Dict]:
    """Iterate over data records in a section."""
    f.seek(section_offset)
    end = section_offset + section_length
    while f.tell() < end:
        result = _read_record(f)
        if result is None or result[0] == RECORD_END:
            break
        record_type, payload = result
        if record_type == RECORD_DATA:
            yield json.loads(payload.decode('utf-8'))


# ─── Similarity / Dedup helpers ──────────────────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    """Simple tokenizer for dedup similarity."""
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── MinHash LSH for fast approximate dedup ──────────────────────────────────

_NUM_HASHES = 128
_NUM_BANDS = 16
_ROWS_PER_BAND = _NUM_HASHES // _NUM_BANDS  # 8
_LARGE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1

import random as _random
_rng = _random.Random(42)
_HASH_PARAMS = [(_rng.randint(1, _LARGE_PRIME - 1), _rng.randint(0, _LARGE_PRIME - 1))
                for _ in range(_NUM_HASHES)]


def _shingle(tokens: Set[str], k: int = 2) -> Set[int]:
    """Convert token set to integer shingle hashes for MinHash."""
    if not tokens:
        return set()
    sorted_tokens = sorted(tokens)
    shingles = set()
    if len(sorted_tokens) < k:
        shingles.add(hash(tuple(sorted_tokens)) & _MAX_HASH)
    else:
        for i in range(len(sorted_tokens) - k + 1):
            shingles.add(hash(tuple(sorted_tokens[i:i+k])) & _MAX_HASH)
    # Also add individual tokens as shingles for better recall on short texts
    for t in sorted_tokens:
        shingles.add(hash(t) & _MAX_HASH)
    return shingles


def _minhash_signature(shingles: Set[int]) -> List[int]:
    """Compute MinHash signature (list of _NUM_HASHES min values)."""
    if not shingles:
        return [_MAX_HASH] * _NUM_HASHES
    sig = []
    for a, b in _HASH_PARAMS:
        min_val = _MAX_HASH
        for s in shingles:
            h = ((a * s + b) % _LARGE_PRIME) & _MAX_HASH
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig


class _LSHIndex:
    """Locality-sensitive hashing index for fast approximate nearest neighbor dedup."""

    def __init__(self):
        # band_idx -> {band_hash -> set of item ids}
        self._bands: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(_NUM_BANDS)]
        # item_id -> signature
        self._signatures: Dict[int, List[int]] = {}

    def add(self, item_id: int, signature: List[int]):
        """Add an item's MinHash signature to the index."""
        self._signatures[item_id] = signature
        for band_idx in range(_NUM_BANDS):
            start = band_idx * _ROWS_PER_BAND
            band_slice = tuple(signature[start:start + _ROWS_PER_BAND])
            band_hash = hash(band_slice)
            self._bands[band_idx][band_hash].append(item_id)

    def query_candidates(self, signature: List[int]) -> Set[int]:
        """Find candidate items that might be similar (via LSH band collision)."""
        candidates = set()
        for band_idx in range(_NUM_BANDS):
            start = band_idx * _ROWS_PER_BAND
            band_slice = tuple(signature[start:start + _ROWS_PER_BAND])
            band_hash = hash(band_slice)
            bucket = self._bands[band_idx].get(band_hash)
            if bucket:
                candidates.update(bucket)
        return candidates

    def estimate_jaccard(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard from two MinHash signatures."""
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


# ─── Writer ──────────────────────────────────────────────────────────────────

class SynapseWriter:
    """Streaming writer for .synapse files."""
    
    def __init__(self, path: str, flags: int = 0, source_agent: str = "unknown",
                 filter_info: Optional[Dict] = None):
        self.path = path
        self.flags = flags
        self.source_agent = source_agent
        self.filter_info = filter_info or {"type": "full"}
        self.timestamp = time.time()
        
        # Buffers for each section (list of JSON-serializable dicts)
        self._sections: Dict[int, List[Dict]] = {
            SECTION_MEMORIES: [],
            SECTION_EDGES: [],
            SECTION_CONCEPTS: [],
            SECTION_EPISODES: [],
        }
        self._counts = {
            'memory_count': 0,
            'edge_count': 0,
            'concept_count': 0,
            'episode_count': 0,
        }
    
    def add_memory(self, memory: Dict[str, Any], provenance: Optional[Dict] = None):
        """Add a memory record."""
        record = dict(memory)
        if provenance:
            record['provenance'] = provenance
            self.flags |= FLAG_HAS_PROVENANCE
        self._sections[SECTION_MEMORIES].append(record)
        self._counts['memory_count'] += 1
    
    def add_edge(self, edge: Dict[str, Any]):
        self._sections[SECTION_EDGES].append(edge)
        self._counts['edge_count'] += 1
    
    def add_concept(self, concept: Dict[str, Any]):
        self._sections[SECTION_CONCEPTS].append(concept)
        self._counts['concept_count'] += 1
    
    def add_episode(self, episode: Dict[str, Any]):
        self._sections[SECTION_EPISODES].append(episode)
        self._counts['episode_count'] += 1
    
    def write(self):
        """Write the .synapse file."""
        # Build metadata section
        metadata = {
            'format_version': [FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR],
            'synapse_version': '2.0',
            'exported_at': self.timestamp,
            'source_agent': self.source_agent,
            'filter': self.filter_info,
            **self._counts,
        }
        
        section_order = [
            SECTION_MEMORIES, SECTION_EDGES, SECTION_CONCEPTS,
            SECTION_EPISODES, SECTION_METADATA
        ]
        num_sections = len(section_order)
        
        # Pre-serialize all section data to compute offsets
        section_blobs: Dict[int, bytes] = {}
        for sec_type in section_order[:-1]:  # all except metadata
            buf = io.BytesIO()
            for record in self._sections[sec_type]:
                payload = json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                _write_record(buf, RECORD_DATA, payload)
            _write_record(buf, RECORD_END, b'')
            section_blobs[sec_type] = buf.getvalue()
        
        # Metadata section
        meta_buf = io.BytesIO()
        meta_payload = json.dumps(metadata, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        _write_record(meta_buf, RECORD_DATA, meta_payload)
        _write_record(meta_buf, RECORD_END, b'')
        section_blobs[SECTION_METADATA] = meta_buf.getvalue()
        
        # Compute offsets
        dir_size = num_sections * DIR_ENTRY_SIZE
        data_start = HEADER_SIZE + dir_size
        
        offsets = {}
        current_offset = data_start
        for sec_type in section_order:
            offsets[sec_type] = current_offset
            current_offset += len(section_blobs[sec_type])
        
        # Compute CRC over all section data
        crc = 0
        for sec_type in section_order:
            crc = zlib.crc32(section_blobs[sec_type], crc)
        crc &= 0xFFFFFFFF
        
        # Write file
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'wb') as f:
            # Header
            f.write(_pack_header(self.flags, self.timestamp, num_sections, crc))
            
            # Directory
            for sec_type in section_order:
                f.write(_pack_dir_entry(sec_type, offsets[sec_type], len(section_blobs[sec_type])))
            
            # Section data
            for sec_type in section_order:
                f.write(section_blobs[sec_type])


# ─── Reader ──────────────────────────────────────────────────────────────────

class SynapseReader:
    """Reader for .synapse files. Supports random access to sections."""
    
    def __init__(self, path: str):
        self.path = path
        self._header = None
        self._directory: Dict[int, Dict[str, int]] = {}
        self._metadata: Optional[Dict] = None
        self._json_data: Optional[Dict] = None
        self._parse_header()
    
    def _parse_header(self):
        with open(self.path, 'rb') as f:
            header_bytes = f.read(HEADER_SIZE)
            self._header = _unpack_header(header_bytes)
            
            if self._header.get('json_fallback'):
                # Read entire file as JSON
                f.seek(0)
                self._json_data = json.loads(f.read().decode('utf-8'))
                return
            
            # Read section directory
            num_sections = self._header['num_sections']
            for _ in range(num_sections):
                entry_bytes = f.read(DIR_ENTRY_SIZE)
                entry = _unpack_dir_entry(entry_bytes)
                self._directory[entry['type']] = entry
    
    @property
    def header(self) -> Dict[str, Any]:
        return self._header
    
    @property
    def is_json_fallback(self) -> bool:
        return self._header.get('json_fallback', False)
    
    def verify_crc(self) -> bool:
        """Verify file integrity via CRC32."""
        if self.is_json_fallback:
            return True
        
        crc = 0
        with open(self.path, 'rb') as f:
            for entry in sorted(self._directory.values(), key=lambda e: e['offset']):
                f.seek(entry['offset'])
                data = f.read(entry['length'])
                crc = zlib.crc32(data, crc)
        crc &= 0xFFFFFFFF
        return crc == self._header['crc']
    
    def metadata(self) -> Dict[str, Any]:
        """Read metadata section (cheap — doesn't load memories)."""
        if self._metadata is not None:
            return self._metadata
        
        if self.is_json_fallback:
            self._metadata = self._json_data.get('metadata', {})
            return self._metadata
        
        if SECTION_METADATA not in self._directory:
            return {}
        
        entry = self._directory[SECTION_METADATA]
        with open(self.path, 'rb') as f:
            records = list(_iter_records(f, entry['offset'], entry['length']))
        
        self._metadata = records[0] if records else {}
        return self._metadata
    
    def iter_memories(self) -> Iterator[Dict]:
        """Stream memory records without loading all into RAM."""
        if self.is_json_fallback:
            yield from self._json_data.get('memories', [])
            return
        
        if SECTION_MEMORIES not in self._directory:
            return
        
        entry = self._directory[SECTION_MEMORIES]
        with open(self.path, 'rb') as f:
            yield from _iter_records(f, entry['offset'], entry['length'])
    
    def iter_edges(self) -> Iterator[Dict]:
        if self.is_json_fallback:
            yield from self._json_data.get('edges', [])
            return
        if SECTION_EDGES not in self._directory:
            return
        entry = self._directory[SECTION_EDGES]
        with open(self.path, 'rb') as f:
            yield from _iter_records(f, entry['offset'], entry['length'])
    
    def iter_concepts(self) -> Iterator[Dict]:
        if self.is_json_fallback:
            yield from self._json_data.get('concepts', [])
            return
        if SECTION_CONCEPTS not in self._directory:
            return
        entry = self._directory[SECTION_CONCEPTS]
        with open(self.path, 'rb') as f:
            yield from _iter_records(f, entry['offset'], entry['length'])
    
    def iter_episodes(self) -> Iterator[Dict]:
        if self.is_json_fallback:
            yield from self._json_data.get('episodes', [])
            return
        if SECTION_EPISODES not in self._directory:
            return
        entry = self._directory[SECTION_EPISODES]
        with open(self.path, 'rb') as f:
            yield from _iter_records(f, entry['offset'], entry['length'])
    
    def load_all(self) -> Dict[str, Any]:
        """Load entire file contents."""
        return {
            'metadata': self.metadata(),
            'memories': list(self.iter_memories()),
            'edges': list(self.iter_edges()),
            'concepts': list(self.iter_concepts()),
            'episodes': list(self.iter_episodes()),
        }


# ─── Export from Synapse ─────────────────────────────────────────────────────

def export_synapse(synapse, path: str, *,
                   since: Optional[str] = None,
                   until: Optional[str] = None,
                   concepts: Optional[List[str]] = None,
                   tags: Optional[List[str]] = None,
                   memory_types: Optional[List[str]] = None,
                   source_agent: str = "unknown") -> str:
    """
    Export a Synapse instance to a .synapse file.
    
    Args:
        synapse: Synapse instance
        path: Output file path
        since: ISO date string for start of time range filter
        until: ISO date string for end of time range filter  
        concepts: List of concept names to filter by
        tags: List of metadata tags to filter by
        memory_types: List of memory types to filter by
        source_agent: Agent identifier for provenance
    
    Returns:
        Path to the written file
    """
    from datetime import datetime, timezone
    
    # Parse time filters
    since_ts = None
    until_ts = None
    if since:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        since_ts = dt.timestamp()
    if until:
        dt = datetime.fromisoformat(until)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        until_ts = dt.timestamp()
    
    # Determine which memories to include
    is_partial = any([since, until, concepts, tags, memory_types])
    flags = FLAG_PARTIAL if is_partial else 0
    
    filter_info = {
        'type': 'partial' if is_partial else 'full',
        'since': since,
        'until': until,
        'concepts': concepts,
        'tags': tags,
        'memory_types': memory_types,
    }
    
    writer = SynapseWriter(path, flags=flags, source_agent=source_agent,
                           filter_info=filter_info)
    
    # Build concept -> memory_id mapping for concept filtering
    concept_memory_ids: Optional[Set[int]] = None
    if concepts:
        concept_memory_ids = set()
        for concept_name in concepts:
            concept_name_lower = concept_name.lower()
            for cname, cnode in synapse.concept_graph.concepts.items():
                if cname.lower() == concept_name_lower or concept_name_lower in cname.lower():
                    concept_memory_ids.update(cnode.memory_ids)
    
    # Collect qualifying memory IDs
    included_ids: Set[int] = set()
    provenance = {'source': source_agent, 'exported_at': time.time()}
    
    for memory_id, memory_data in synapse.store.memories.items():
        # Time filter
        if since_ts and memory_data['created_at'] < since_ts:
            continue
        if until_ts and memory_data['created_at'] > until_ts:
            continue
        
        # Concept filter
        if concept_memory_ids is not None and memory_id not in concept_memory_ids:
            continue
        
        # Memory type filter
        if memory_types and memory_data.get('memory_type') not in memory_types:
            continue
        
        # Tag filter
        if tags:
            meta = memory_data.get('metadata', '{}')
            if isinstance(meta, str):
                meta = json.loads(meta)
            memory_tags = meta.get('tags', [])
            if not any(t in memory_tags for t in tags):
                continue
        
        included_ids.add(memory_id)
        writer.add_memory(memory_data, provenance=provenance)
    
    # Export edges (only those connecting included memories)
    for edge_id, edge_data in synapse.store.edges.items():
        if edge_data['source_id'] in included_ids and edge_data['target_id'] in included_ids:
            writer.add_edge(edge_data)
    
    # Export concepts (only those with included memories)
    for concept_name, concept_node in synapse.concept_graph.concepts.items():
        relevant_ids = concept_node.memory_ids & included_ids
        if relevant_ids:
            writer.add_concept({
                'name': concept_name,
                'category': concept_node.category,
                'memory_ids': sorted(relevant_ids),
                'created_at': concept_node.created_at,
            })
    
    # Export episodes (only those with included memories)
    for episode_id, episode_data in synapse.store.episodes.items():
        ep_memory_ids = set(episode_data.get('memory_ids', []))
        relevant_ids = ep_memory_ids & included_ids
        if relevant_ids:
            ep = dict(episode_data)
            ep['memory_ids'] = sorted(relevant_ids)
            writer.add_episode(ep)
    
    writer.write()
    return path


# ─── Import into Synapse ─────────────────────────────────────────────────────

def import_synapse(synapse, path: str, *, deduplicate: bool = True,
                   similarity_threshold: float = 0.85) -> Dict[str, int]:
    """
    Import a .synapse file into a Synapse instance (replaces current data).
    
    Returns:
        Stats dict with counts of imported items.
    """
    reader = SynapseReader(path)
    
    if not reader.verify_crc():
        raise ValueError(f"CRC check failed for {path} — file may be corrupted")
    
    stats = {'memories': 0, 'edges': 0, 'concepts': 0, 'episodes': 0, 'skipped_duplicates': 0}
    
    # ID remapping (file IDs -> new Synapse IDs)
    id_map: Dict[int, int] = {}
    
    # Build LSH index for existing memories for fast dedup
    lsh_index = _LSHIndex() if deduplicate else None
    existing_tokens: Dict[int, Set[str]] = {}
    if deduplicate:
        for mid, mdata in synapse.store.memories.items():
            tokens = _tokenize(mdata['content'])
            existing_tokens[mid] = tokens
            shingles = _shingle(tokens)
            sig = _minhash_signature(shingles)
            lsh_index.add(mid, sig)
    
    # Import memories
    for record in reader.iter_memories():
        old_id = record.get('id')
        content = record.get('content', '')
        
        # Dedup check using LSH
        if deduplicate and lsh_index and lsh_index._signatures:
            new_tokens = _tokenize(content)
            new_shingles = _shingle(new_tokens)
            new_sig = _minhash_signature(new_shingles)
            candidates = lsh_index.query_candidates(new_sig)
            is_dup = False
            for cand_id in candidates:
                if _jaccard(new_tokens, existing_tokens[cand_id]) >= similarity_threshold:
                    id_map[old_id] = cand_id
                    stats['skipped_duplicates'] += 1
                    is_dup = True
                    break
            if is_dup:
                continue
        
        # Insert memory
        memory_data = {
            'content': content,
            'memory_type': record.get('memory_type', 'fact'),
            'strength': record.get('strength', 1.0),
            'access_count': record.get('access_count', 0),
            'created_at': record.get('created_at', time.time()),
            'last_accessed': record.get('last_accessed', time.time()),
            'metadata': record.get('metadata', '{}') if isinstance(record.get('metadata'), str) else json.dumps(record.get('metadata', {})),
            'consolidated': record.get('consolidated', False),
            'summary_of': record.get('summary_of', '[]') if isinstance(record.get('summary_of'), str) else json.dumps(record.get('summary_of', [])),
        }
        
        new_id = synapse.store.insert_memory(memory_data)
        if old_id is not None:
            id_map[old_id] = new_id
        
        # Update indexes
        synapse.inverted_index.add_document(new_id, content)
        synapse.temporal_index.add_memory(new_id, memory_data['created_at'])
        
        if deduplicate:
            new_tokens = _tokenize(content)
            existing_tokens[new_id] = new_tokens
            new_shingles = _shingle(new_tokens)
            new_sig = _minhash_signature(new_shingles)
            lsh_index.add(new_id, new_sig)
        
        stats['memories'] += 1
    
    # Import edges with remapped IDs
    for record in reader.iter_edges():
        src = id_map.get(record.get('source_id'))
        tgt = id_map.get(record.get('target_id'))
        if src is not None and tgt is not None:
            edge_type = record.get('edge_type', 'related')
            weight = record.get('weight', 1.0)
            synapse.edge_graph.add_edge(src, tgt, edge_type, weight)
            synapse.store.insert_edge({
                'source_id': src, 'target_id': tgt,
                'edge_type': edge_type, 'weight': weight,
                'created_at': record.get('created_at', time.time()),
            })
            stats['edges'] += 1
    
    # Import concepts
    for record in reader.iter_concepts():
        name = record.get('name', '')
        category = record.get('category', 'general')
        for old_mid in record.get('memory_ids', []):
            new_mid = id_map.get(old_mid)
            if new_mid is not None:
                synapse.concept_graph.link_memory_concept(new_mid, name, category)
        stats['concepts'] += 1
    
    # Import episodes
    for record in reader.iter_episodes():
        old_eid = record.get('id')
        ep_data = {
            'name': record.get('name', ''),
            'started_at': record.get('started_at', time.time()),
            'ended_at': record.get('ended_at', time.time()),
            'memory_ids': [],
        }
        new_eid = synapse.store.insert_episode(ep_data)
        synapse.episode_index.add_episode(new_eid, ep_data['name'], ep_data['started_at'], ep_data['ended_at'])
        
        for old_mid in record.get('memory_ids', []):
            new_mid = id_map.get(old_mid)
            if new_mid is not None:
                synapse.episode_index.add_memory_to_episode(new_mid, new_eid)
                synapse.store.episodes[new_eid].setdefault('memory_ids', []).append(new_mid)
        
        stats['episodes'] += 1
    
    return stats


# ─── Merge ───────────────────────────────────────────────────────────────────

def merge_synapse(synapse, path: str, *,
                  conflict_resolution: str = "newer_wins",
                  similarity_threshold: float = 0.85) -> Dict[str, int]:
    """
    Merge a .synapse file into an existing Synapse instance without overwriting.
    
    Args:
        synapse: Target Synapse instance
        path: Path to .synapse file to merge from
        conflict_resolution: "newer_wins" | "keep_both"
        similarity_threshold: Jaccard threshold for fuzzy dedup (0.0 to 1.0)
    
    Returns:
        Stats dict.
    """
    reader = SynapseReader(path)
    
    if not reader.verify_crc():
        raise ValueError(f"CRC check failed for {path}")
    
    stats = {'memories_added': 0, 'memories_updated': 0, 'memories_skipped': 0,
             'edges_added': 0, 'concepts_merged': 0, 'episodes_merged': 0}
    
    # Build LSH index for existing memories
    lsh_index = _LSHIndex()
    existing_tokens: Dict[int, Set[str]] = {}
    existing_by_content: Dict[int, Dict] = {}
    for mid, mdata in synapse.store.memories.items():
        tokens = _tokenize(mdata['content'])
        existing_tokens[mid] = tokens
        existing_by_content[mid] = mdata
        shingles = _shingle(tokens)
        sig = _minhash_signature(shingles)
        lsh_index.add(mid, sig)
    
    id_map: Dict[int, int] = {}
    source_meta = reader.metadata()
    source_agent = source_meta.get('source_agent', 'unknown')
    
    for record in reader.iter_memories():
        old_id = record.get('id')
        content = record.get('content', '')
        new_tokens = _tokenize(content)
        new_shingles = _shingle(new_tokens)
        new_sig = _minhash_signature(new_shingles)
        
        # Find best match among LSH candidates (with fallback for low thresholds)
        candidates = lsh_index.query_candidates(new_sig)
        # For low thresholds, also check MinHash estimate against all items
        if similarity_threshold < 0.7 and len(existing_tokens) <= 5000:
            for eid, esig in lsh_index._signatures.items():
                est = lsh_index.estimate_jaccard(new_sig, esig)
                if est >= similarity_threshold * 0.5:
                    candidates.add(eid)
        best_match_id = None
        best_similarity = 0.0
        for cand_id in candidates:
            sim = _jaccard(new_tokens, existing_tokens[cand_id])
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = cand_id
        
        if best_similarity >= similarity_threshold and best_match_id is not None:
            # Duplicate found
            if conflict_resolution == "newer_wins":
                existing = existing_by_content[best_match_id]
                if record.get('created_at', 0) > existing.get('created_at', 0):
                    # Update existing with newer data
                    synapse.store.update_memory(best_match_id, {
                        'content': content,
                        'metadata': record.get('metadata', '{}') if isinstance(record.get('metadata'), str) else json.dumps(record.get('metadata', {})),
                        'last_accessed': max(record.get('last_accessed', 0), existing.get('last_accessed', 0)),
                        'strength': max(record.get('strength', 1.0), existing.get('strength', 1.0)),
                    })
                    synapse.inverted_index.add_document(best_match_id, content)
                    stats['memories_updated'] += 1
                else:
                    stats['memories_skipped'] += 1
                id_map[old_id] = best_match_id
            elif conflict_resolution == "keep_both":
                # Add provenance tag and insert as new
                meta = record.get('metadata', {})
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta['merged_from'] = source_agent
                meta['merge_duplicate_of'] = best_match_id
                
                memory_data = {
                    'content': content,
                    'memory_type': record.get('memory_type', 'fact'),
                    'strength': record.get('strength', 1.0),
                    'access_count': record.get('access_count', 0),
                    'created_at': record.get('created_at', time.time()),
                    'last_accessed': record.get('last_accessed', time.time()),
                    'metadata': json.dumps(meta),
                    'consolidated': record.get('consolidated', False),
                    'summary_of': json.dumps(record.get('summary_of', [])),
                }
                new_id = synapse.store.insert_memory(memory_data)
                id_map[old_id] = new_id
                synapse.inverted_index.add_document(new_id, content)
                synapse.temporal_index.add_memory(new_id, memory_data['created_at'])
                existing_tokens[new_id] = new_tokens
                lsh_index.add(new_id, new_sig)
                stats['memories_added'] += 1
            else:
                stats['memories_skipped'] += 1
                id_map[old_id] = best_match_id
        else:
            # New memory — add it
            meta = record.get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta['merged_from'] = source_agent
            
            memory_data = {
                'content': content,
                'memory_type': record.get('memory_type', 'fact'),
                'strength': record.get('strength', 1.0),
                'access_count': record.get('access_count', 0),
                'created_at': record.get('created_at', time.time()),
                'last_accessed': record.get('last_accessed', time.time()),
                'metadata': json.dumps(meta),
                'consolidated': record.get('consolidated', False),
                'summary_of': json.dumps(record.get('summary_of', [])),
            }
            new_id = synapse.store.insert_memory(memory_data)
            id_map[old_id] = new_id
            synapse.inverted_index.add_document(new_id, content)
            synapse.temporal_index.add_memory(new_id, memory_data['created_at'])
            existing_tokens[new_id] = new_tokens
            lsh_index.add(new_id, new_sig)
            stats['memories_added'] += 1
    
    # Merge edges
    for record in reader.iter_edges():
        src = id_map.get(record.get('source_id'))
        tgt = id_map.get(record.get('target_id'))
        if src is not None and tgt is not None:
            synapse.edge_graph.add_edge(src, tgt, record.get('edge_type', 'related'), record.get('weight', 1.0))
            synapse.store.insert_edge({
                'source_id': src, 'target_id': tgt,
                'edge_type': record.get('edge_type', 'related'),
                'weight': record.get('weight', 1.0),
                'created_at': record.get('created_at', time.time()),
            })
            stats['edges_added'] += 1
    
    # Merge concept graph (union of edges, max of weights)
    for record in reader.iter_concepts():
        name = record.get('name', '')
        category = record.get('category', 'general')
        for old_mid in record.get('memory_ids', []):
            new_mid = id_map.get(old_mid)
            if new_mid is not None:
                synapse.concept_graph.link_memory_concept(new_mid, name, category)
        stats['concepts_merged'] += 1
    
    # Merge episodes
    for record in reader.iter_episodes():
        ep_name = record.get('name', '')
        # Try to find existing episode with same name
        existing_ep = None
        for eid, edata in synapse.store.episodes.items():
            if edata.get('name') == ep_name and ep_name:
                existing_ep = eid
                break
        
        if existing_ep is not None:
            # Merge into existing episode
            for old_mid in record.get('memory_ids', []):
                new_mid = id_map.get(old_mid)
                if new_mid is not None:
                    synapse.episode_index.add_memory_to_episode(new_mid, existing_ep)
                    synapse.store.episodes[existing_ep].setdefault('memory_ids', []).append(new_mid)
        else:
            ep_data = {
                'name': ep_name,
                'started_at': record.get('started_at', time.time()),
                'ended_at': record.get('ended_at', time.time()),
                'memory_ids': [],
            }
            new_eid = synapse.store.insert_episode(ep_data)
            synapse.episode_index.add_episode(new_eid, ep_name, ep_data['started_at'], ep_data['ended_at'])
            for old_mid in record.get('memory_ids', []):
                new_mid = id_map.get(old_mid)
                if new_mid is not None:
                    synapse.episode_index.add_memory_to_episode(new_mid, new_eid)
                    synapse.store.episodes[new_eid].setdefault('memory_ids', []).append(new_mid)
        stats['episodes_merged'] += 1
    
    return stats


# ─── Inspect ─────────────────────────────────────────────────────────────────

def inspect_synapse(path: str) -> Dict[str, Any]:
    """Inspect a .synapse file without loading all data."""
    reader = SynapseReader(path)
    meta = reader.metadata()
    
    file_size = os.path.getsize(path)
    
    info = {
        'path': path,
        'file_size': file_size,
        'file_size_human': _human_size(file_size),
        'format': 'json_fallback' if reader.is_json_fallback else 'binary',
        'crc_valid': reader.verify_crc(),
    }
    
    if not reader.is_json_fallback:
        info['version'] = f"{reader.header['version_major']}.{reader.header['version_minor']}"
        info['flags'] = {
            'partial': bool(reader.header['flags'] & FLAG_PARTIAL),
            'compressed': bool(reader.header['flags'] & FLAG_COMPRESSED),
            'has_provenance': bool(reader.header['flags'] & FLAG_HAS_PROVENANCE),
        }
        info['created_at'] = reader.header['timestamp']
        info['sections'] = {
            sec_type: {'offset': entry['offset'], 'length': entry['length']}
            for sec_type, entry in reader._directory.items()
        }
    
    info.update(meta)
    return info


def _human_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ─── Diff ────────────────────────────────────────────────────────────────────

def diff_synapse(path_a: str, path_b: str, *, similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Compare two .synapse files and return differences.
    """
    reader_a = SynapseReader(path_a)
    reader_b = SynapseReader(path_b)
    
    memories_a = {m['id']: m for m in reader_a.iter_memories()}
    memories_b = {m['id']: m for m in reader_b.iter_memories()}
    
    # Tokenize all
    tokens_a = {mid: _tokenize(m['content']) for mid, m in memories_a.items()}
    tokens_b = {mid: _tokenize(m['content']) for mid, m in memories_b.items()}
    
    # Find matches
    matched_a: Set[int] = set()
    matched_b: Set[int] = set()
    matches: List[Dict] = []
    
    for aid, atoks in tokens_a.items():
        best_bid = None
        best_sim = 0.0
        for bid, btoks in tokens_b.items():
            if bid in matched_b:
                continue
            sim = _jaccard(atoks, btoks)
            if sim > best_sim:
                best_sim = sim
                best_bid = bid
        if best_sim >= similarity_threshold and best_bid is not None:
            matched_a.add(aid)
            matched_b.add(best_bid)
            if best_sim < 1.0:
                matches.append({
                    'id_a': aid, 'id_b': best_bid,
                    'similarity': round(best_sim, 3),
                    'content_a': memories_a[aid]['content'][:100],
                    'content_b': memories_b[best_bid]['content'][:100],
                })
    
    only_a = [{'id': mid, 'content': memories_a[mid]['content'][:100]}
              for mid in memories_a if mid not in matched_a]
    only_b = [{'id': mid, 'content': memories_b[mid]['content'][:100]}
              for mid in memories_b if mid not in matched_b]
    
    meta_a = reader_a.metadata()
    meta_b = reader_b.metadata()
    
    return {
        'file_a': path_a,
        'file_b': path_b,
        'memories_a': len(memories_a),
        'memories_b': len(memories_b),
        'shared': len(matched_a),
        'modified': len(matches),
        'only_in_a': len(only_a),
        'only_in_b': len(only_b),
        'only_in_a_samples': only_a[:10],
        'only_in_b_samples': only_b[:10],
        'modified_samples': matches[:10],
        'source_a': meta_a.get('source_agent', 'unknown'),
        'source_b': meta_b.get('source_agent', 'unknown'),
    }


# ─── JSON Fallback Export ────────────────────────────────────────────────────

def export_json_fallback(synapse, path: str, **kwargs) -> str:
    """Export as plain JSON (human-readable fallback)."""
    # Use the same filtering logic
    data = _collect_export_data(synapse, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _collect_export_data(synapse, *, since=None, until=None, concepts=None,
                         tags=None, memory_types=None, source_agent="unknown") -> Dict:
    """Collect data for export (shared between binary and JSON)."""
    from datetime import datetime, timezone
    
    def _parse_dt(s):
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    
    since_ts = _parse_dt(since) if since else None
    until_ts = _parse_dt(until) if until else None
    
    concept_memory_ids = None
    if concepts:
        concept_memory_ids = set()
        for cn in concepts:
            cn_lower = cn.lower()
            for cname, cnode in synapse.concept_graph.concepts.items():
                if cname.lower() == cn_lower or cn_lower in cname.lower():
                    concept_memory_ids.update(cnode.memory_ids)
    
    memories = []
    included_ids = set()
    
    for mid, mdata in synapse.store.memories.items():
        if since_ts and mdata['created_at'] < since_ts:
            continue
        if until_ts and mdata['created_at'] > until_ts:
            continue
        if concept_memory_ids is not None and mid not in concept_memory_ids:
            continue
        if memory_types and mdata.get('memory_type') not in memory_types:
            continue
        included_ids.add(mid)
        memories.append(mdata)
    
    edges = [e for e in synapse.store.edges.values()
             if e['source_id'] in included_ids and e['target_id'] in included_ids]
    
    concept_records = []
    for cname, cnode in synapse.concept_graph.concepts.items():
        relevant = cnode.memory_ids & included_ids
        if relevant:
            concept_records.append({
                'name': cname, 'category': cnode.category,
                'memory_ids': sorted(relevant), 'created_at': cnode.created_at,
            })
    
    episodes = []
    for eid, edata in synapse.store.episodes.items():
        ep_mids = set(edata.get('memory_ids', []))
        relevant = ep_mids & included_ids
        if relevant:
            ep = dict(edata)
            ep['memory_ids'] = sorted(relevant)
            episodes.append(ep)
    
    return {
        'metadata': {
            'format_version': [FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR],
            'synapse_version': '2.0',
            'exported_at': time.time(),
            'source_agent': source_agent,
            'memory_count': len(memories),
            'edge_count': len(edges),
            'concept_count': len(concept_records),
            'episode_count': len(episodes),
        },
        'memories': memories,
        'edges': edges,
        'concepts': concept_records,
        'episodes': episodes,
    }
