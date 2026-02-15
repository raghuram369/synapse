#!/usr/bin/env python3
"""
Synapse CLI Tool

Command-line interface for interacting with the Synapse daemon.
Provides easy access to all Synapse operations from the terminal.
"""

import argparse
import json
import sys
from typing import List, Dict, Any

from client import SynapseClient, SynapseConnectionError, SynapseRequestError


def format_memory(memory: Dict[str, Any], show_metadata: bool = False) -> str:
    """Format a memory for display."""
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
    """Format a concept for display."""
    return f"🏷️  {concept['name']} ({concept['category']}) - {concept['memory_count']} memories"


def format_stats(stats: Dict[str, Any]) -> str:
    """Format server stats for display."""
    lines = []
    lines.append("📊 Server Statistics:")
    lines.append(f"   Memories: {stats.get('total_memories', 0)}")
    lines.append(f"   Concepts: {stats.get('total_concepts', 0)}")
    lines.append(f"   Edges: {stats.get('total_edges', 0)}")
    lines.append(f"   Clients: {stats.get('client_count', 0)}")
    
    if 'data_directory' in stats:
        lines.append(f"   Data Dir: {stats['data_directory']}")
        
    return '\n'.join(lines)


def cmd_remember(args, client: SynapseClient):
    """Handle remember command."""
    try:
        memory = client.remember(
            content=args.content,
            memory_type=args.type,
            extract=args.extract
        )
        
        print("✅ Memory stored successfully!")
        print(format_memory(memory))
        
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_recall(args, client: SynapseClient):
    """Handle recall command."""
    try:
        memories = client.recall(
            context=args.query,
            limit=args.limit
        )
        
        if not memories:
            print("🔍 No memories found")
            return
            
        print(f"🔍 Found {len(memories)} memories:")
        print()
        
        for i, memory in enumerate(memories):
            if i > 0:
                print()  # Blank line between memories
            print(format_memory(memory, show_metadata=args.metadata))
            
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_forget(args, client: SynapseClient):
    """Handle forget command."""
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
    """Handle link command."""
    try:
        client.link(
            source_id=args.source,
            target_id=args.target,
            edge_type=args.edge_type,
            weight=args.weight
        )
        
        print(f"✅ Link created: #{args.source} --[{args.edge_type}]--> #{args.target}")
        
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_concepts(args, client: SynapseClient):
    """Handle concepts command."""
    try:
        concepts = client.concepts()
        
        if not concepts:
            print("🏷️  No concepts found")
            return
            
        print(f"🏷️  Found {len(concepts)} concepts:")
        print()
        
        for concept in concepts[:args.limit]:
            print(format_concept(concept))
            
        if len(concepts) > args.limit:
            print(f"... and {len(concepts) - args.limit} more")
            
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_stats(args, client: SynapseClient):
    """Handle stats command."""
    try:
        stats = client.stats()
        print(format_stats(stats))
        
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_ping(args, client: SynapseClient):
    """Handle ping command."""
    try:
        response = client.ping()
        print(f"📡 Server response: {response}")
        
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def cmd_shutdown(args, client: SynapseClient):
    """Handle shutdown command."""
    try:
        if not args.force:
            response = input("⚠️  Are you sure you want to shut down the server? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Shutdown cancelled")
                return
        
        client.shutdown()
        print("🛑 Server shutdown initiated")
        
    except SynapseConnectionError:
        # Expected - server shuts down
        print("✅ Server shut down successfully")
    except SynapseRequestError as e:
        print(f"❌ Request error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Synapse CLI Tool")
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=7654, help='Server port')
    parser.add_argument('--timeout', type=float, default=30.0, help='Request timeout')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Remember command
    remember_parser = subparsers.add_parser('remember', help='Store a new memory')
    remember_parser.add_argument('content', help='Memory content')
    remember_parser.add_argument('--type', default='fact', 
                                help='Memory type (fact, event, preference, skill, observation)')
    remember_parser.add_argument('--extract', action='store_true',
                                help='Enable fact extraction')
    remember_parser.add_argument('--no-extract', dest='extract', action='store_false',
                                help='Disable fact extraction')
    remember_parser.set_defaults(extract=None)
    
    # Recall command
    recall_parser = subparsers.add_parser('recall', help='Search for memories')
    recall_parser.add_argument('query', nargs='?', default='', help='Search query')
    recall_parser.add_argument('--limit', type=int, default=10, help='Max results')
    recall_parser.add_argument('--metadata', action='store_true', 
                              help='Show metadata in results')
    
    # Forget command
    forget_parser = subparsers.add_parser('forget', help='Delete a memory')
    forget_parser.add_argument('id', type=int, help='Memory ID to delete')
    
    # Link command
    link_parser = subparsers.add_parser('link', help='Create a link between memories')
    link_parser.add_argument('source', type=int, help='Source memory ID')
    link_parser.add_argument('target', type=int, help='Target memory ID')
    link_parser.add_argument('--edge-type', default='related', help='Edge type')
    link_parser.add_argument('--weight', type=float, default=1.0, help='Edge weight')
    
    # Concepts command
    concepts_parser = subparsers.add_parser('concepts', help='List all concepts')
    concepts_parser.add_argument('--limit', type=int, default=50, help='Max results')
    
    # Stats command
    subparsers.add_parser('stats', help='Show server statistics')
    
    # Ping command
    subparsers.add_parser('ping', help='Ping the server')
    
    # Shutdown command
    shutdown_parser = subparsers.add_parser('shutdown', help='Shutdown the server')
    shutdown_parser.add_argument('--force', action='store_true',
                                help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Connect to server
    try:
        client = SynapseClient(
            host=args.host,
            port=args.port,
            timeout=args.timeout
        )
        client.connect()
        
    except SynapseConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Make sure the Synapse daemon is running on {args.host}:{args.port}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'remember':
            cmd_remember(args, client)
        elif args.command == 'recall':
            cmd_recall(args, client)
        elif args.command == 'forget':
            cmd_forget(args, client)
        elif args.command == 'link':
            cmd_link(args, client)
        elif args.command == 'concepts':
            cmd_concepts(args, client)
        elif args.command == 'stats':
            cmd_stats(args, client)
        elif args.command == 'ping':
            cmd_ping(args, client)
        elif args.command == 'shutdown':
            cmd_shutdown(args, client)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
            
    finally:
        client.close()


if __name__ == "__main__":
    main()