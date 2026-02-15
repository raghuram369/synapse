"""
Example: LangChain + Synapse — AI with persistent, private memory.

This shows a conversational chain that remembers across sessions,
with all memory stored locally. No cloud. No API calls for storage.

Requirements:
    pip install langchain langchain-openai synapse-ai-memory

Run:
    export OPENAI_API_KEY=sk-...
    python example.py
"""

import os
import sys

# Add parent path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from synapse import Synapse
from integrations.langchain.memory import SynapseMemory
from integrations.langchain.retriever import SynapseRetriever
from integrations.langchain.chat_history import SynapseChatMessageHistory


def demo_memory_chain():
    """Demo: Conversational chain with persistent memory."""
    print("=" * 60)
    print("LangChain + Synapse: Persistent Private Memory")
    print("=" * 60)

    # 1. Create Synapse instance — memory persists to disk
    syn = Synapse("./demo_langchain_memory")

    # 2. Pre-load some memories (simulating prior conversations)
    syn.remember("User's name is Alex", memory_type="fact")
    syn.remember("Alex prefers Python over JavaScript", memory_type="preference")
    syn.remember("Alex is building a trading bot", memory_type="fact")
    syn.remember("Alex likes vegetarian food", memory_type="preference")
    syn.remember("Last meeting was about API design patterns", memory_type="event")

    # 3. Create LangChain memory wrapper
    memory = SynapseMemory(synapse=syn, memory_key="context", recall_limit=5)

    # 4. Show semantic recall in action
    print("\n📝 Stored 5 memories about Alex")
    print("\n🔍 Query: 'What programming languages?'")
    result = memory.load_memory_variables({"input": "What programming languages?"})
    print(f"   Recalled:\n{result['context']}")

    print("\n🔍 Query: 'What are they working on?'")
    result = memory.load_memory_variables({"input": "What are they working on?"})
    print(f"   Recalled:\n{result['context']}")

    print("\n🔍 Query: 'dietary preferences'")
    result = memory.load_memory_variables({"input": "dietary preferences"})
    print(f"   Recalled:\n{result['context']}")

    # 5. Demo with actual LangChain (if installed)
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️  Set OPENAI_API_KEY to run the full LangChain demo")
            return

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use this context about the user:\n{context}"),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()

        # Query with memory context
        context = memory.load_memory_variables({"input": "Help me with my project"})
        response = chain.invoke({"input": "Help me with my project", **context})
        print(f"\n🤖 Response: {response}")

        # Save the interaction
        memory.save_context(
            {"input": "Help me with my project"},
            {"output": response},
        )
        print("\n✅ Interaction saved to Synapse (persists across restarts)")

    except ImportError:
        print("\n💡 Install langchain-openai for the full demo:")
        print("   pip install langchain-openai")

    # Cleanup
    syn.close()
    print("\n✨ Done! Memory persisted to ./demo_langchain_memory/")


def demo_retriever():
    """Demo: Synapse as a LangChain retriever for RAG."""
    print("\n" + "=" * 60)
    print("Synapse Retriever: Local RAG without a vector DB")
    print("=" * 60)

    syn = Synapse(":memory:")

    # Load knowledge base
    facts = [
        "Synapse uses BM25 keyword search combined with concept graph expansion",
        "Synapse supports temporal decay — old unused memories fade naturally",
        "Federation allows multiple Synapse instances to share memories peer-to-peer",
        "The .synapse portable format lets you export and import memory files",
        "Synapse has zero external dependencies — pure Python",
        "Episode grouping clusters related memories from the same conversation",
        "Memory consolidation merges similar memories to reduce redundancy",
    ]
    for fact in facts:
        syn.remember(fact, memory_type="fact")

    retriever = SynapseRetriever(synapse=syn, k=3)
    print(f"\n📚 Loaded {len(facts)} facts into Synapse")

    # Retrieve
    query = "How does Synapse handle search?"
    docs = retriever.invoke(query)
    print(f"\n🔍 Query: '{query}'")
    print("   Results:")
    for doc in docs:
        print(f"   • {doc.page_content}")

    syn.close()


def demo_chat_history():
    """Demo: Persistent chat history with semantic search."""
    print("\n" + "=" * 60)
    print("Synapse Chat History: Semantic Memory for Conversations")
    print("=" * 60)

    syn = Synapse(":memory:")
    history = SynapseChatMessageHistory(synapse=syn, session_id="demo-session")

    # Simulate a conversation
    history.add_user_message("I'm learning Rust for systems programming")
    history.add_ai_message("Rust is great for systems programming! It offers memory safety without a garbage collector.")
    history.add_user_message("What about async programming in Rust?")
    history.add_ai_message("Rust has async/await with the tokio runtime. It's very performant.")
    history.add_user_message("I also enjoy cooking Italian food on weekends")
    history.add_ai_message("Italian cooking is wonderful! Do you have favorite dishes?")

    print(f"\n💬 Stored {len(history.messages)} messages")

    # Semantic search — finds relevant messages regardless of position
    print("\n🔍 Searching: 'programming language features'")
    results = history.search("programming language features", limit=3)
    for r in results:
        print(f"   • {r.content[:80]}...")

    print("\n🔍 Searching: 'hobbies and food'")
    results = history.search("hobbies and food", limit=3)
    for r in results:
        print(f"   • {r.content[:80]}...")

    syn.close()


if __name__ == "__main__":
    demo_memory_chain()
    demo_retriever()
    demo_chat_history()
