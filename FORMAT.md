# .synapse File Format Specification v1

## Overview

The `.synapse` format is a portable binary container for Synapse memory databases. It stores memories, indexes, and graph structures in a single file that can be transferred between agents and systems.

Design goals: zero external dependencies, streaming-friendly, forward-compatible, inspectable.

## Binary Layout

```
┌─────────────────────────────────────────┐
│  Header (32 bytes fixed)                │
├─────────────────────────────────────────┤
│  Section Directory (variable)           │
├─────────────────────────────────────────┤
│  Section 0: Memories                    │
├─────────────────────────────────────────┤
│  Section 1: Edges                       │
├─────────────────────────────────────────┤
│  Section 2: Concept Graph               │
├─────────────────────────────────────────┤
│  Section 3: Episodes                    │
├─────────────────────────────────────────┤
│  Section 4: Metadata                    │
└─────────────────────────────────────────┘
```

## Header (32 bytes)

| Offset | Size | Type    | Description                          |
|--------|------|---------|--------------------------------------|
| 0      | 4    | bytes   | Magic: `\x89SYN` (0x89 + "SYN")     |
| 4      | 2    | uint16  | Format version (major.minor as 2 bytes) |
| 6      | 2    | uint16  | Flags (bitfield, see below)          |
| 8      | 8    | uint64  | Creation timestamp (seconds, float as fixed-point: microseconds since epoch) |
| 16     | 4    | uint32  | Number of sections                   |
| 20     | 4    | uint32  | CRC32 of all section data            |
| 24     | 8    | bytes   | Reserved (zero-filled for future use)|

### Flags (bit positions)
- Bit 0: `PARTIAL` — file contains a subset of memories (filtered export)
- Bit 1: `COMPRESSED` — section data is zlib-compressed (using stdlib)
- Bit 2: `HAS_PROVENANCE` — memories include provenance metadata
- Bits 3-15: Reserved

## Section Directory

Immediately after the header. Each entry is 20 bytes:

| Offset | Size | Type    | Description                |
|--------|------|---------|----------------------------|
| 0      | 2    | uint16  | Section type ID            |
| 2      | 2    | uint16  | Reserved                   |
| 4      | 8    | uint64  | Offset from file start     |
| 12     | 8    | uint64  | Section length in bytes    |

### Section Type IDs
- `0x01` — Memories
- `0x02` — Edges  
- `0x03` — Concept Graph
- `0x04` — Episodes
- `0x05` — Metadata (export info, provenance, stats)

## Section Data Encoding

Each section contains a sequence of **records**. Records use a simple TLV (Type-Length-Value) encoding:

### Record Format
```
[type: 1 byte][length: 4 bytes (uint32, big-endian)][payload: JSON bytes]
```

- **type**: Record type within the section (0x01 = data record, 0xFF = section end marker)
- **length**: Byte length of the JSON payload
- **payload**: UTF-8 encoded JSON object

This keeps the format inspectable (JSON payloads) while providing structure for streaming reads.

### Memory Record (section 0x01)
```json
{
  "id": 1,
  "content": "The user prefers dark mode",
  "memory_type": "preference",
  "strength": 1.0,
  "access_count": 5,
  "created_at": 1707000000.0,
  "last_accessed": 1707100000.0,
  "metadata": {},
  "consolidated": false,
  "summary_of": [],
  "provenance": {
    "source": "agent-alpha",
    "exported_at": 1707200000.0
  }
}
```

### Edge Record (section 0x02)
```json
{
  "id": 1,
  "source_id": 1,
  "target_id": 2,
  "edge_type": "related",
  "weight": 0.8,
  "created_at": 1707000000.0
}
```

### Concept Graph Record (section 0x03)
```json
{
  "name": "python",
  "category": "programming languages",
  "memory_ids": [1, 3, 7],
  "created_at": 1707000000.0
}
```

### Episode Record (section 0x04)
```json
{
  "id": 1,
  "name": "debugging session",
  "started_at": 1707000000.0,
  "ended_at": 1707003600.0,
  "memory_ids": [1, 2, 3]
}
```

### Metadata Record (section 0x05)
```json
{
  "format_version": [1, 0],
  "synapse_version": "2.0",
  "exported_at": 1707200000.0,
  "source_agent": "agent-alpha",
  "memory_count": 150,
  "edge_count": 45,
  "concept_count": 30,
  "episode_count": 5,
  "filter": {
    "type": "full",
    "since": null,
    "concepts": null,
    "tags": null
  }
}
```

## Streaming

The format supports streaming in both directions:

- **Export**: Write header with placeholder CRC, write sections sequentially, seek back to update CRC.
- **Import**: Read header, use section directory to seek to any section, read records one at a time.
- **Inspect**: Read only the header + metadata section (last section) to get stats without loading memories.

## Versioning

The format version in the header enables forward compatibility:
- **Major version change**: Breaking format change (readers must reject unknown major versions)
- **Minor version change**: Additive changes (new section types, new record fields). Readers should skip unknown sections/fields.

Current version: **1.0**

## JSON Fallback

For debugging or interop, the entire file can alternatively be a plain JSON file (detected by checking if the first byte is `{` instead of `\x89`). The JSON fallback has the same logical structure but without binary framing.

## Integrity

CRC32 covers all bytes from the first section to end of file. This catches corruption but is not a security mechanism.
