"""
Example: LangGraph + Synapse — Stateful agents with persistent memory.

Shows a LangGraph agent with:
- Checkpoint persistence (survives restarts)
- Cross-thread shared memory (agents share knowledge)
- All data stays local. No cloud. No API calls for storage.

Requirements:
    pip install langgraph langchain-openai synapse-ai-memory

Run:
    export OPENAI_API_KEY=sk-...
    python example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from synapse import Synapse
from integrations.langgraph.checkpointer import SynapseCheckpointer
from integrations.langgraph.memory_store import SynapseStore


def demo_checkpointer():
    """Demo: Graph state persistence across restarts."""
    print("=" * 60)
    print("LangGraph + Synapse: Persistent Checkpointing")
    print("=" * 60)

    syn = Synapse(":memory:")
    checkpointer = SynapseCheckpointer(synapse=syn)

    # Simulate saving a checkpoint
    config = {"configurable": {"thread_id": "conversation-1"}}
    state = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny and 72°F."},
        ],
        "step": 2,
    }

    result_config = checkpointer.put(config, state, metadata={"step": 2})
    print(f"\n✅ Saved checkpoint: {result_config['configurable']['checkpoint_id']}")

    # Save another checkpoint
    state["messages"].append({"role": "user", "content": "Thanks!"})
    state["step"] = 3
    result_config = checkpointer.put(config, state, metadata={"step": 3})
    print(f"✅ Saved checkpoint: {result_config['configurable']['checkpoint_id']}")

    # Restore latest checkpoint (simulating restart)
    restored = checkpointer.get_tuple(config)
    if restored:
        cp = restored if isinstance(restored, dict) else {"checkpoint": restored.checkpoint}
        checkpoint = cp.get("checkpoint", cp) if isinstance(cp, dict) else cp
        if hasattr(checkpoint, 'checkpoint'):
            checkpoint = checkpoint.checkpoint
        print(f"\n🔄 Restored state: {len(checkpoint.get('messages', []))} messages, step {checkpoint.get('step')}")
    else:
        print("\n⚠️ No checkpoint found")

    syn.close()


def demo_store():
    """Demo: Cross-thread shared memory."""
    print("\n" + "=" * 60)
    print("LangGraph + Synapse: Cross-Thread Shared Memory")
    print("=" * 60)

    syn = Synapse(":memory:")
    store = SynapseStore(synapse=syn)

    # Thread 1: Learn about the user
    print("\n📝 Thread 1 stores user preferences...")
    store.put(("user", "preferences"), "diet", {"value": "vegetarian", "confidence": 0.9})
    store.put(("user", "preferences"), "language", {"value": "Python", "confidence": 0.95})
    store.put(("user", "context"), "project", {"value": "building a trading bot", "status": "active"})

    # Thread 2: Access shared memory
    print("🔍 Thread 2 retrieves user preferences...")
    diet = store.get(("user", "preferences"), "diet")
    if diet:
        val = diet.value if hasattr(diet, 'value') else diet
        print(f"   Diet: {val}")

    project = store.get(("user", "context"), "project")
    if project:
        val = project.value if hasattr(project, 'value') else project
        print(f"   Project: {val}")

    # Semantic search across all user data
    print("\n🔍 Semantic search: 'food preferences'")
    results = store.search(("user",), query="food preferences")
    for r in results:
        val = r.value if hasattr(r, 'value') else r
        print(f"   Found: {val}")

    # List namespaces
    namespaces = store.list_namespaces(prefix=("user",))
    print(f"\n📂 Namespaces under 'user': {namespaces}")

    syn.close()


def demo_full_agent():
    """Demo: Full LangGraph agent with Synapse (requires langchain-openai)."""
    print("\n" + "=" * 60)
    print("Full LangGraph Agent with Synapse Memory")
    print("=" * 60)

    try:
        from langgraph.graph import StateGraph, START, END
        from langchain_openai import ChatOpenAI
        from typing import TypedDict, Annotated, Sequence
        import operator

        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️  Set OPENAI_API_KEY to run the full agent demo")
            return

        class AgentState(TypedDict):
            messages: Annotated[Sequence[dict], operator.add]
            memory_context: str

        syn = Synapse("./demo_langgraph_memory")
        checkpointer = SynapseCheckpointer(synapse=syn)
        store = SynapseStore(synapse=syn)

        llm = ChatOpenAI(model="gpt-4o-mini")

        def recall_memory(state: AgentState) -> AgentState:
            """Recall relevant memories before responding."""
            last_msg = state["messages"][-1]["content"] if state["messages"] else ""
            results = store.search(("agent",), query=last_msg, limit=3)
            context = "\n".join(
                str(r.value if hasattr(r, 'value') else r) for r in results
            )
            return {"memory_context": context, "messages": []}

        def respond(state: AgentState) -> AgentState:
            """Generate response with memory context."""
            messages = state["messages"]
            context = state.get("memory_context", "")

            system = f"You are a helpful assistant. Relevant memories:\n{context}" if context else "You are a helpful assistant."
            response = llm.invoke([
                {"role": "system", "content": system},
                *messages,
            ])

            return {"messages": [{"role": "assistant", "content": response.content}]}

        def save_memory(state: AgentState) -> AgentState:
            """Save important facts to shared memory."""
            for msg in state["messages"]:
                if msg.get("role") == "user":
                    syn.remember(msg["content"], metadata={"source": "langgraph_agent"})
            return {"messages": []}

        # Build graph
        builder = StateGraph(AgentState)
        builder.add_node("recall", recall_memory)
        builder.add_node("respond", respond)
        builder.add_node("save", save_memory)
        builder.add_edge(START, "recall")
        builder.add_edge("recall", "respond")
        builder.add_edge("respond", "save")
        builder.add_edge("save", END)

        graph = builder.compile(checkpointer=checkpointer)

        # Run
        config = {"configurable": {"thread_id": "demo-thread"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "I'm working on a Python trading bot"}], "memory_context": ""},
            config=config,
        )
        print(f"\n🤖 {result['messages'][-1]['content'][:200]}...")
        print("\n✅ State checkpointed. Memory saved. All local.")

        syn.close()

    except ImportError as e:
        print(f"\n💡 Install dependencies for the full demo:")
        print("   pip install langgraph langchain-openai")
        print(f"   (Missing: {e})")


if __name__ == "__main__":
    demo_checkpointer()
    demo_store()
    demo_full_agent()
