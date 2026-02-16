"""
Example: ChatGPT + Synapse AI Memory — GPT with persistent private memory.

Your data stays local. Only inference calls go to OpenAI.

Requirements:
    pip install openai synapse-ai-memory

Run:
    export OPENAI_API_KEY=sk-...
    python example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from synapse import Synapse
from integrations.openai.memory import SynapseGPTMemory
from integrations.openai.tool import synapse_functions, handle_synapse_function, run_with_functions


def demo_auto_memory():
    """Demo: GPT with automatic persistent memory."""
    print("=" * 60)
    print("ChatGPT + Synapse: Automatic Persistent Memory")
    print("=" * 60)

    syn = Synapse(":memory:")

    # Pre-load memories
    syn.remember("User's name is Jordan", memory_type="fact")
    syn.remember("Jordan is a data scientist", memory_type="fact")
    syn.remember("Jordan prefers Jupyter notebooks over VS Code", memory_type="preference")
    syn.remember("Jordan is working on a customer churn prediction model", memory_type="fact")

    gpt = SynapseGPTMemory(synapse=syn, session_id="demo")

    print("\n📝 Pre-loaded 4 memories about Jordan")

    # Show recall
    memories = syn.recall("What tools do they use?", limit=5)
    print("\n🔍 Query: 'What tools do they use?'")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    memories = syn.recall("current project", limit=5)
    print("\n🔍 Query: 'current project'")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    if os.environ.get("OPENAI_API_KEY"):
        print("\n🤖 Calling GPT with memory...")
        response = gpt.chat("What do you know about my work?")
        print(f"   {response[:200]}...")
    else:
        print("\n⚠️  Set OPENAI_API_KEY for the live demo")

    syn.close()


def demo_function_calling():
    """Demo: GPT with Synapse as function calls."""
    print("\n" + "=" * 60)
    print("ChatGPT + Synapse: Function-Based Memory")
    print("=" * 60)

    syn = Synapse(":memory:")

    # Show function definitions
    functions = synapse_functions()
    print(f"\n🔧 Registered {len(functions)} Synapse functions:")
    for f in functions:
        print(f"   • {f['function']['name']}: {f['function']['description'][:60]}...")

    # Simulate function calls
    print("\n📞 Simulated function calls:")

    result = handle_synapse_function(syn, "remember", {
        "content": "User loves hiking in Colorado",
        "memory_type": "preference",
    })
    print(f"   remember → {result}")

    result = handle_synapse_function(syn, "remember", {
        "content": "User's dog is named Max",
        "memory_type": "fact",
    })
    print(f"   remember → {result}")

    result = handle_synapse_function(syn, "recall", {"query": "outdoor activities"})
    print(f"   recall → {result}")

    if os.environ.get("OPENAI_API_KEY"):
        print("\n🤖 Full function-calling conversation:")
        response = run_with_functions(
            syn,
            [{"role": "user", "content": "Remember that I'm vegetarian. Then tell me what you know about me."}],
        )
        print(f"   {response[:200]}...")
    else:
        print("\n⚠️  Set OPENAI_API_KEY for the live demo")

    syn.close()


def demo_cross_session():
    """Demo: Memory persisting across sessions."""
    print("\n" + "=" * 60)
    print("ChatGPT + Synapse: Cross-Session Memory")
    print("=" * 60)

    # Session 1
    syn = Synapse("/tmp/gpt_synapse_demo")
    syn.remember("User is learning Japanese", memory_type="skill")
    syn.remember("User's goal is JLPT N3 by December", memory_type="fact")
    syn.remember("User practices with Anki flashcards", memory_type="observation")
    syn.close()
    print("\n📝 Session 1: Stored 3 memories → closed")

    # Session 2
    syn = Synapse("/tmp/gpt_synapse_demo")
    memories = syn.recall("language learning progress", limit=5)
    print(f"\n🔄 Session 2: Recalled {len(memories)} memories:")
    for m in memories:
        print(f"   • [{m.memory_type}] {m.content}")

    print("\n✨ GPT now remembers across sessions — all local, all private!")
    syn.close()

    import shutil
    try:
        shutil.rmtree("/tmp/gpt_synapse_demo")
    except OSError:
        pass


if __name__ == "__main__":
    demo_auto_memory()
    demo_function_calling()
    demo_cross_session()
