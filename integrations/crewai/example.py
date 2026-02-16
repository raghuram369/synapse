"""
Example: CrewAI + Synapse AI Memory — Multi-agent crews with shared, persistent memory.

Shows agents sharing knowledge via Synapse AI Memory, with federation for
multi-crew collaboration. All memory stays local.

Requirements:
    pip install crewai synapse-ai-memory

Run:
    python example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from synapse import Synapse
from integrations.crewai.memory import SynapseCrewMemory


def demo_shared_memory():
    """Demo: Agents sharing memory within a crew."""
    print("=" * 60)
    print("CrewAI + Synapse: Shared Memory Between Agents")
    print("=" * 60)

    syn = Synapse(":memory:")
    memory = SynapseCrewMemory(synapse=syn, crew_id="research-crew")

    # Researcher agent discovers facts
    print("\n🔬 Researcher agent stores findings...")
    memory.save_long_term(
        "Python 3.13 introduces a JIT compiler for improved performance",
        agent="researcher",
    )
    memory.save_long_term(
        "Rust adoption in enterprise grew 40% in 2025",
        agent="researcher",
    )
    memory.save_entity(
        "Python", "programming_language",
        "General-purpose language, popular for AI/ML, web development",
        agent="researcher",
    )
    memory.save_entity(
        "Rust", "programming_language",
        "Systems language focused on memory safety, growing in enterprise",
        agent="researcher",
    )

    # Writer agent accesses shared memory
    print("✍️  Writer agent searches for content...")
    results = memory.search("programming language trends", agent=None)  # Search all agents
    print(f"   Found {len(results)} relevant memories:")
    for r in results:
        print(f"   • [{r['agent']}] {r['content'][:70]}...")

    # Entity lookup
    print("\n🏷️  Entity search: 'Python'")
    entities = memory.search_entity("Python", limit=3)
    for e in entities:
        print(f"   • {e['content']}")

    syn.close()


def demo_task_context():
    """Demo: Short-term memory scoped to tasks."""
    print("\n" + "=" * 60)
    print("CrewAI + Synapse: Task-Scoped Short-Term Memory")
    print("=" * 60)

    syn = Synapse(":memory:")
    memory = SynapseCrewMemory(synapse=syn, crew_id="content-crew")

    # Task 1: Research phase
    print("\n📋 Task: 'market-research'")
    memory.save_short_term(
        "AI market expected to reach $1.8T by 2030",
        agent="analyst", task="market-research",
    )
    memory.save_short_term(
        "Key players: OpenAI, Anthropic, Google DeepMind",
        agent="analyst", task="market-research",
    )

    # Task 2: Writing phase (can access research findings)
    print("📋 Task: 'write-report'")
    memory.save_short_term(
        "Draft: The AI market is experiencing unprecedented growth...",
        agent="writer", task="write-report",
    )

    # Cross-task search
    print("\n🔍 Writer searches for market data (cross-task)...")
    results = memory.search("AI market size", limit=3)
    for r in results:
        print(f"   • [{r['agent']}] {r['content'][:70]}...")

    syn.close()


def demo_federation():
    """Demo: Multi-crew federation — crews sharing knowledge."""
    print("\n" + "=" * 60)
    print("CrewAI + Synapse: Multi-Crew Federation")
    print("=" * 60)

    # Crew A: Research team
    syn_a = Synapse(":memory:")
    crew_a = SynapseCrewMemory(synapse=syn_a, crew_id="research")

    crew_a.save_long_term("GraphRAG improves retrieval by 30% over naive RAG", agent="researcher")
    crew_a.save_long_term("Fine-tuning small models can match GPT-4 on specific tasks", agent="researcher")

    # Crew B: Engineering team
    syn_b = Synapse(":memory:")
    crew_b = SynapseCrewMemory(synapse=syn_b, crew_id="engineering")

    crew_b.save_long_term("Our API latency is 200ms p99", agent="engineer")
    crew_b.save_long_term("We use Python 3.13 with uvicorn for the backend", agent="engineer")

    # Export/Import for knowledge sharing (no network needed)
    print("\n📤 Research crew exports knowledge...")
    export_path = "/tmp/research_knowledge.synapse"
    crew_a.export_crew_knowledge(export_path)

    print("📥 Engineering crew imports research findings...")
    stats = crew_b.import_crew_knowledge(export_path)
    print(f"   Imported: {stats}")

    # Engineering crew can now search research findings
    print("\n🔍 Engineering crew searches: 'RAG improvements'")
    results = crew_b.search("RAG improvements")
    for r in results:
        print(f"   • {r['content'][:70]}...")

    syn_a.close()
    syn_b.close()

    # Clean up
    import os
    try:
        os.remove(export_path)
    except OSError:
        pass

    print("\n✨ Multi-crew knowledge sharing — no cloud, no API calls!")


def demo_crewai_integration():
    """Demo: Full CrewAI integration (requires crewai package)."""
    print("\n" + "=" * 60)
    print("Full CrewAI Integration")
    print("=" * 60)

    try:
        from crewai import Agent, Task, Crew

        syn = Synapse("./demo_crew_memory")
        memory = SynapseCrewMemory(synapse=syn, crew_id="demo")

        # Pre-load context
        memory.save_long_term("The user is building an AI-powered trading platform")
        memory.save_long_term("Target market: retail traders in the US")

        researcher = Agent(
            role="Market Researcher",
            goal="Research AI trading platforms",
            backstory="Expert in fintech market analysis",
        )

        writer = Agent(
            role="Content Writer",
            goal="Write compelling product descriptions",
            backstory="Technical writer specializing in fintech",
        )

        research_task = Task(
            description="Research the competitive landscape for AI trading platforms",
            expected_output="A summary of key competitors and opportunities",
            agent=researcher,
        )

        writing_task = Task(
            description="Write a product positioning statement",
            expected_output="A 2-paragraph positioning statement",
            agent=writer,
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
        )

        print("\n🚀 Running crew... (this calls your LLM)")
        # result = crew.kickoff()  # Uncomment to actually run
        print("   (Uncomment crew.kickoff() to run with an LLM)")

        syn.close()

    except ImportError:
        print("\n💡 Install crewai for the full integration:")
        print("   pip install crewai")


if __name__ == "__main__":
    demo_shared_memory()
    demo_task_context()
    demo_federation()
    demo_crewai_integration()
