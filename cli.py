#!/usr/bin/env python3
"""
Synapse CLI Tool

Unified command-line interface for Synapse memory engine:
  - Core: remember, recall, forget, link, concepts, stats
  - Portable Format: export, import, inspect, merge, diff
  - Federation: serve, push, pull, sync, peers
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

from client import SynapseClient, SynapseConnectionError, SynapseRequestError


# ─── Formatting helpers ──────────────────────────────────────────────────────

def format_memory(memory: Dict[str, Any], show_metadata: bool = False) -> str:
    lines = []
    lines.append(f"🧠 Memory #{memory['id']} ({memory['memory_type']})")
    lines.append(f"   Content: {memory['content']}")
    lines.append(f"   Strength: {memory.get('strength', 'N/A')}")
    if 'effective_strength' in memory:
        lines.append(f"   Effective: {memory['effective_strength']:.3f}")
    if 'created_at' in memory:
        import datetime
        created = datetime.datetime.fromtimestamp(memory['created_at'])
        lines.append(f"   Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
    if show_metadata and memory.get('metadata'):
        lines.append(f"   Metadata: {json.dumps(memory['metadata'], indent=2)}")
    return '\n'.join(lines)


def format_concept(concept: Dict[str, Any]) -> str:
    return f"🏷️  {concept['name']} ({concept['category']}) - {concept['memory_count']} memories"


def format_stats(stats: Dict[str, Any]) -> str:
    lines = ["📊 Server Statistics:"]
    lines.append(f"   Memories: {stats.get('total_memories', 0)}")
    lines.append(f"   Concepts: {stats.get('total_concepts', 0)}")
    lines.append(f"   Edges: {stats.get('total_edges', 0)}")
    lines.append(f"   Clients: {stats.get('client_count', 0)}")
    if 'data_directory' in stats:
        lines.append(f"   Data Dir: {stats['data_directory']}")
    return '\n'.join(lines)


def _human_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ─── Core commands (daemon client) ───────────────────────────────────────────

def cmd_remember(args, client: SynapseClient):
    try:
        memory = client.remember(content=args.content, memory_type=args.type, extract=args.extract)
        print("✅ Memory stored successfully!")
        print(format_memory(memory))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_recall(args, client: SynapseClient):
    try:
        memories = client.recall(context=args.query, limit=args.limit)
        if not memories:
            print("🔍 No memories found")
            return
        print(f"🔍 Found {len(memories)} memories:\n")
        for i, memory in enumerate(memories):
            if i > 0:
                print()
            print(format_memory(memory, show_metadata=args.metadata))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_forget(args, client: SynapseClient):
    try:
        deleted = client.forget(args.id)
        if deleted:
            print(f"✅ Memory #{args.id} deleted successfully")
        else:
            print(f"⚠️  Memory #{args.id} not found")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_link(args, client: SynapseClient):
    try:
        client.link(source_id=args.source, target_id=args.target,
                     edge_type=args.edge_type, weight=args.weight)
        print(f"✅ Link created: #{args.source} --[{args.edge_type}]--> #{args.target}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_concepts(args, client: SynapseClient):
    try:
        concepts = client.concepts()
        if not concepts:
            print("🏷️  No concepts found")
            return
        print(f"🏷️  Found {len(concepts)} concepts:\n")
        for concept in concepts[:args.limit]:
            print(format_concept(concept))
        if len(concepts) > args.limit:
            print(f"... and {len(concepts) - args.limit} more")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_stats(args, client: SynapseClient):
    try:
        stats = client.stats()
        print(format_stats(stats))
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_ping(args, client: SynapseClient):
    try:
        response = client.ping()
        print(f"📡 Server response: {response}")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_shutdown(args, client: SynapseClient):
    try:
        if not args.force:
            response = input("⚠️  Are you sure you want to shut down the server? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Shutdown cancelled")
                return
        client.shutdown()
        print("🛑 Server shutdown initiated")
    except SynapseConnectionError:
        print("✅ Server shut down successfully")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


# ─── Portable Format commands (standalone, no daemon needed) ─────────────────

def cmd_export(args):
    from portable import export_synapse
    from synapse import Synapse
    db_path = args.db or ":memory:"
    s = Synapse(db_path)
    concepts = args.concepts.split(',') if args.concepts else None
    tags = args.tags.split(',') if args.tags else None
    types = args.types.split(',') if args.types else None
    path = export_synapse(s, args.output, since=args.since, until=args.until,
                          concepts=concepts, tags=tags, memory_types=types,
                          source_agent=args.source or "cli")
    size = os.path.getsize(path)
    print(f"✓ Exported to {path} ({_human_size(size)})")
    s.close()


def cmd_import(args):
    from portable import import_synapse
    from synapse import Synapse
    db_path = args.db or ":memory:"
    s = Synapse(db_path)
    stats = import_synapse(s, args.input, deduplicate=not args.no_dedup,
                           similarity_threshold=args.threshold)
    print(f"✓ Imported from {args.input}")
    print(f"  Memories: {stats['memories']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Concepts: {stats['concepts']}")
    print(f"  Episodes: {stats['episodes']}")
    if stats.get('skipped_duplicates'):
        print(f"  Skipped duplicates: {stats['skipped_duplicates']}")
    s.flush()
    s.close()


def cmd_inspect(args):
    from portable import inspect_synapse
    info = inspect_synapse(args.input)
    print(f"═══ {info['path']} ═══")
    print(f"  Format:      {info.get('format', 'unknown')}")
    print(f"  Size:        {info.get('file_size_human', '?')}")
    print(f"  CRC Valid:   {'✓' if info.get('crc_valid') else '✗'}")
    if 'version' in info:
        print(f"  Version:     {info['version']}")
    flags = info.get('flags', {})
    if flags:
        active_flags = [k for k, v in flags.items() if v]
        print(f"  Flags:       {', '.join(active_flags) if active_flags else 'none'}")
    print(f"  Memories:    {info.get('memory_count', '?')}")
    print(f"  Edges:       {info.get('edge_count', '?')}")
    print(f"  Concepts:    {info.get('concept_count', '?')}")
    print(f"  Episodes:    {info.get('episode_count', '?')}")
    print(f"  Source:      {info.get('source_agent', '?')}")
    filt = info.get('filter', {})
    if filt and filt.get('type') != 'full':
        print(f"  Filter:      {json.dumps(filt)}")
    if 'exported_at' in info:
        from datetime import datetime
        dt = datetime.fromtimestamp(info['exported_at'])
        print(f"  Exported:    {dt.isoformat()}")


def cmd_merge(args):
    from portable import merge_synapse, import_synapse, export_synapse
    from synapse import Synapse
    s = Synapse(":memory:")
    files = args.inputs
    if len(files) < 2:
        print("Error: Need at least 2 files to merge")
        sys.exit(1)
    stats = import_synapse(s, files[0], deduplicate=False)
    print(f"✓ Base: {files[0]} ({stats['memories']} memories)")
    for f in files[1:]:
        stats = merge_synapse(s, f, conflict_resolution=args.conflict,
                              similarity_threshold=args.threshold)
        print(f"✓ Merged: {f} (+{stats['memories_added']} new, "
              f"{stats['memories_updated']} updated, {stats['memories_skipped']} skipped)")
    output = args.output or "merged.synapse"
    export_synapse(s, output, source_agent="merge-cli")
    size = os.path.getsize(output)
    print(f"✓ Output: {output} ({_human_size(size)})")
    s.close()


def cmd_diff(args):
    from portable import diff_synapse
    result = diff_synapse(args.file_a, args.file_b, similarity_threshold=args.threshold)
    print(f"═══ Diff: {result['file_a']} vs {result['file_b']} ═══")
    print(f"  Source A:    {result['source_a']}")
    print(f"  Source B:    {result['source_b']}")
    print(f"  Memories A:  {result['memories_a']}")
    print(f"  Memories B:  {result['memories_b']}")
    print(f"  Shared:      {result['shared']}")
    print(f"  Modified:    {result['modified']}")
    print(f"  Only in A:   {result['only_in_a']}")
    print(f"  Only in B:   {result['only_in_b']}")
    if result['only_in_a_samples']:
        print(f"\n  Only in A (samples):")
        for s in result['only_in_a_samples']:
            print(f"    [{s['id']}] {s['content']}")
    if result['only_in_b_samples']:
        print(f"\n  Only in B (samples):")
        for s in result['only_in_b_samples']:
            print(f"    [{s['id']}] {s['content']}")
    if result['modified_samples']:
        print(f"\n  Modified (samples):")
        for m in result['modified_samples']:
            print(f"    [{m['id_a']}↔{m['id_b']}] sim={m['similarity']}")
            print(f"      A: {m['content_a']}")
            print(f"      B: {m['content_b']}")


# ─── Federation commands (standalone) ────────────────────────────────────────

def cmd_serve(args):
    from federation.node import SynapseNode
    node = SynapseNode(node_id=args.node_id, path=args.data or ":memory:",
                       auth_token=args.token)
    for ns in (args.share or []):
        node.share(ns)
    node.listen(port=args.port, host=args.host)
    if args.discover:
        node.start_discovery(port=args.port)
    print(f"Synapse node '{args.node_id}' listening on {args.host}:{args.port}")
    print(f"  Memories: {len(node.store.memories)}")
    print(f"  Shared namespaces: {sorted(node.store.shared_namespaces) or '(all)'}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        node.save()
        node.stop()


def cmd_push_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.push(args.peer, ns)
    print(f"Pushed {result['pushed']} memories to {args.peer}")


def cmd_pull_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.pull(args.peer, ns)
    print(f"Pulled {result['pulled']} memories from {args.peer}")
    if args.data:
        store.save()


def cmd_sync_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore(args.node_id, args.data or ":memory:")
    client = SyncClient(store)
    ns = set(args.namespace) if args.namespace else None
    result = client.sync(args.peer, ns)
    print(f"Synced with {args.peer}: pulled {result['pulled']}, pushed {result['pushed']}")
    if args.data:
        store.save()


def cmd_peers_fed(args):
    from federation.store import FederatedStore
    from federation.client import SyncClient
    store = FederatedStore("cli-temp")
    client = SyncClient(store)
    try:
        status = client.status(args.peer)
        print(f"Node: {status['node_id']}")
        print(f"  Memories: {status['memory_count']}")
        print(f"  Edges: {status['edge_count']}")
        print(f"  Root hash: {status['root_hash'][:16]}...")
        print(f"  Shared: {status.get('shared_namespaces', [])}")
        peers = client.list_peers(args.peer)
        peer_list = peers.get("peers", {})
        if peer_list:
            print(f"  Known peers:")
            for url, info in peer_list.items():
                print(f"    - {url} ({info.get('node_id', '?')})")
        else:
            print("  No known peers")
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='synapse',
        description='Synapse — AI Memory Engine with Portable Format & Federation'
    )
    parser.add_argument('--host', default='127.0.0.1', help='Daemon host')
    parser.add_argument('--port', type=int, default=7654, help='Daemon port')
    parser.add_argument('--timeout', type=float, default=30.0, help='Request timeout')
    parser.add_argument('--node-id', default='default', help='Federation node ID')
    parser.add_argument('--data', '-d', help='Data file path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── Core commands (need daemon) ──
    p = subparsers.add_parser('remember', help='Store a new memory')
    p.add_argument('content', help='Memory content')
    p.add_argument('--type', default='fact')
    p.add_argument('--extract', action='store_true')
    p.add_argument('--no-extract', dest='extract', action='store_false')
    p.set_defaults(extract=None)

    p = subparsers.add_parser('recall', help='Search for memories')
    p.add_argument('query', nargs='?', default='')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--metadata', action='store_true')

    p = subparsers.add_parser('forget', help='Delete a memory')
    p.add_argument('id', type=int)

    p = subparsers.add_parser('link', help='Link two memories')
    p.add_argument('source', type=int)
    p.add_argument('target', type=int)
    p.add_argument('--edge-type', default='related')
    p.add_argument('--weight', type=float, default=1.0)

    p = subparsers.add_parser('concepts', help='List all concepts')
    p.add_argument('--limit', type=int, default=50)

    subparsers.add_parser('stats', help='Show server statistics')
    subparsers.add_parser('ping', help='Ping the server')

    p = subparsers.add_parser('shutdown', help='Shutdown the server')
    p.add_argument('--force', action='store_true')

    # ── Portable Format commands ──
    p = subparsers.add_parser('export', help='Export database to .synapse file')
    p.add_argument('output', help='Output .synapse file path')
    p.add_argument('--db', help='Synapse database path')
    p.add_argument('--since', help='Export since date (ISO)')
    p.add_argument('--until', help='Export until date (ISO)')
    p.add_argument('--concepts', help='Filter by concepts (comma-separated)')
    p.add_argument('--tags', help='Filter by tags (comma-separated)')
    p.add_argument('--types', help='Filter by memory types (comma-separated)')
    p.add_argument('--source', help='Source agent identifier')

    p = subparsers.add_parser('import', help='Import .synapse file')
    p.add_argument('input', help='Input .synapse file path')
    p.add_argument('--db', help='Target database path')
    p.add_argument('--no-dedup', action='store_true')
    p.add_argument('--threshold', type=float, default=0.85)

    p = subparsers.add_parser('inspect', help='Inspect .synapse file')
    p.add_argument('input', help='.synapse file path')

    p = subparsers.add_parser('merge', help='Merge multiple .synapse files')
    p.add_argument('inputs', nargs='+', help='.synapse files to merge')
    p.add_argument('-o', '--output', default='merged.synapse')
    p.add_argument('--conflict', choices=['newer_wins', 'keep_both'], default='newer_wins')
    p.add_argument('--threshold', type=float, default=0.85)

    p = subparsers.add_parser('diff', help='Compare two .synapse files')
    p.add_argument('file_a')
    p.add_argument('file_b')
    p.add_argument('--threshold', type=float, default=0.85)

    # ── Federation commands ──
    p = subparsers.add_parser('serve', help='Start federation server')
    p.add_argument('--fed-port', '-p', type=int, default=9470, dest='port')
    p.add_argument('--fed-host', default='0.0.0.0', dest='host')
    p.add_argument('--share', nargs='*', help='Namespaces to share')
    p.add_argument('--discover', action='store_true')
    p.add_argument('--token', help='Auth token')

    p = subparsers.add_parser('push', help='Push memories to peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('pull', help='Pull memories from peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('sync', help='Bidirectional sync with peer')
    p.add_argument('peer')
    p.add_argument('--namespace', '-n', nargs='*')

    p = subparsers.add_parser('peers', help='Show peer info')
    p.add_argument('peer', help='Peer URL to query')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Standalone commands (no daemon needed)
    standalone = {
        'export': cmd_export,
        'import': cmd_import,
        'inspect': cmd_inspect,
        'merge': cmd_merge,
        'diff': cmd_diff,
        'serve': cmd_serve,
        'push': cmd_push_fed,
        'pull': cmd_pull_fed,
        'sync': cmd_sync_fed,
        'peers': cmd_peers_fed,
    }

    if args.command in standalone:
        standalone[args.command](args)
        return

    # Daemon-dependent commands
    try:
        client = SynapseClient(host=args.host, port=args.port, timeout=args.timeout)
        client.connect()
    except SynapseConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Make sure the Synapse daemon is running on {args.host}:{args.port}")
        sys.exit(1)

    daemon_commands = {
        'remember': cmd_remember,
        'recall': cmd_recall,
        'forget': cmd_forget,
        'link': cmd_link,
        'concepts': cmd_concepts,
        'stats': cmd_stats,
        'ping': cmd_ping,
        'shutdown': cmd_shutdown,
    }

    try:
        if args.command in daemon_commands:
            daemon_commands[args.command](args, client)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
