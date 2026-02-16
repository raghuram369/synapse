# Synapse AI Memory × CrewAI

**Shared, persistent memory for multi-agent crews.** Privacy-first. Federation for multi-crew collaboration.

## Install

```bash
pip install synapse-ai-memory crewai
```

## Quick Start

```python
from synapse import Synapse
from synapse.integrations.crewai import SynapseCrewMemory

syn = Synapse("./crew_memory")
memory = SynapseCrewMemory(synapse=syn, crew_id="my-crew")

# Agents share memory automatically
memory.save("Python 3.13 has a JIT compiler", agent="researcher")
results = memory.search("Python performance", agent=None)  # any agent can find it
```

## Memory Layers

| Layer | Method | Use Case |
|-------|--------|----------|
| Short-term | `save_short_term()` / `search_short_term()` | Current task context |
| Long-term | `save_long_term()` / `search_long_term()` | Persistent facts & learnings |
| Entity | `save_entity()` / `search_entity()` | Structured knowledge (people, orgs, concepts) |

## Multi-Crew Federation

```python
# Crew A exports knowledge
crew_a.export_crew_knowledge("research.synapse")

# Crew B imports it
crew_b.import_crew_knowledge("research.synapse")

# Or sync over the network
crew_a.share_with_crew("http://crew-b:9470")
```

## Run the example

```bash
python example.py
```
