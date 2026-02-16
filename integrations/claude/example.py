"""
Example: Claude + Synapse AI Memory — Conversations that remember across sessions.

Your data stays local. Claude handles the thinking, Synapse AI Memory handles the memory.
No Anthropic API calls for storage — only for inference.

Requirements:
    pip install anthropic synapse-ai-memory

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from synapse import Synapse
from integrations.claude.memory import SynapseClaudeMemory
from integrations.claude.tool import synapse_tools, handle_synapse_tool, run_with_tools


def demo_auto_memory():
    """Demo: Claude with automatic persistent memory."""
    print("=" * 60)
    print("Claude + Synapse: Automatic Persistent Memory")
    print("=" * 60)

    syn = Synapse(":memory:")

    # Pre-load some memories (simulating prior sessions)
    syn.remember("User's name is Alex", memory_type="fact")
    syn.remember("Alex is a Python developer", memory_type="fact")
    syn.remember("Alex prefers dark mode in all editors", memory_type="preference")
    syn.remember("Alex is building a trading bot with async Python", memory_type="fact")

    claude = SynapseClaudeMemory(synapse=syn, session_id="demo")

    print("\n📝 Pre-loaded 4 memories about Alex")

    # Show what Claude would receive as context
    memories = syn.recall("What are they working on?", limit=5)
    print("\n🔍 Query: 'What are they working on?'")
    print("   Context injected into Claude's system prompt:")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    memories = syn.recall("programming preferences", limit=5)
    print("\n🔍 Query: 'programming preferences'")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    # Full API demo (requires API key)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n🤖 Calling Claude with memory context...")
        response = claude.chat("What do you remember about my projects?")
        print(f"   Response: {response[:200]}...")
    else:
        print("\n⚠️  Set ANTHROPIC_API_KEY for the live Claude demo")

    syn.close()


def demo_tool_use():
    """Demo: Claude autonomously deciding what to remember/recall."""
    print("\n" + "=" * 60)
    print("Claude + Synapse: Tool-Based Memory (Claude Decides)")
    print("=" * 60)

    syn = Synapse(":memory:")

    # Show tool definitions
    tools = synapse_tools()
    print(f"\n🔧 Registered {len(tools)} Synapse tools:")
    for tool in tools:
        print(f"   • {tool['name']}: {tool['description'][:60]}...")

    # Simulate tool calls (without API)
    print("\n📞 Simulated tool calls:")

    result = handle_synapse_tool(syn, "remember", {
        "content": "User's favorite color is blue",
        "memory_type": "preference",
    })
    print(f"   remember → {result}")

    result = handle_synapse_tool(syn, "remember", {
        "content": "User works at a fintech startup",
        "memory_type": "fact",
    })
    print(f"   remember → {result}")

    result = handle_synapse_tool(syn, "recall", {
        "query": "user preferences",
        "limit": 5,
    })
    print(f"   recall → {result}")

    result = handle_synapse_tool(syn, "forget", {
        "query": "favorite color",
    })
    print(f"   forget → {result}")

    # Full demo with API
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n🤖 Full tool-use conversation:")
        response = run_with_tools(
            syn,
            [{"role": "user", "content": "Remember that I'm learning Rust. Then tell me what you know about me."}],
        )
        print(f"   {response[:200]}...")
    else:
        print("\n⚠️  Set ANTHROPIC_API_KEY for the live tool-use demo")

    syn.close()


def demo_multi_session():
    """Demo: Memory persisting across sessions."""
    print("\n" + "=" * 60)
    print("Claude + Synapse: Cross-Session Memory")
    print("=" * 60)

    # Session 1
    syn = Synapse("/tmp/claude_synapse_demo")
    syn.remember("User is training for a marathon", memory_type="fact")
    syn.remember("User's goal race is in April 2026", memory_type="event")
    syn.remember("User follows a plant-based diet", memory_type="preference")
    syn.close()
    print("\n📝 Session 1: Stored 3 memories → closed")

    # Session 2 (new process, same file)
    syn = Synapse("/tmp/claude_synapse_demo")
    memories = syn.recall("fitness goals", limit=5)
    print(f"\n🔄 Session 2: Recalled {len(memories)} memories:")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    # This is the magic — memory survives restarts
    print("\n✨ Memory persisted across sessions — all local, all private!")

    syn.close()

    # Cleanup
    import shutil
    try:
        shutil.rmtree("/tmp/claude_synapse_demo")
    except OSError:
        pass


if __name__ == "__main__":
    demo_auto_memory()
    demo_tool_use()
    demo_multi_session()
