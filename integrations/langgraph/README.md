# Synapse AI Memory × LangGraph

**Persistent state and cross-thread memory for LangGraph agents.** All data stays local.

## Install

```bash
pip install synapse-ai-memory langgraph
```

## Components

### `SynapseCheckpointer` — Graph state persistence

```python
from synapse import Synapse
from synapse.integrations.langgraph import SynapseCheckpointer

syn = Synapse("./agent_state")
checkpointer = SynapseCheckpointer(synapse=syn)
graph = builder.compile(checkpointer=checkpointer)

# State survives process restarts
graph.invoke({"input": "hello"}, config={"configurable": {"thread_id": "t1"}})
```

### `SynapseStore` — Cross-thread shared memory

```python
from synapse.integrations.langgraph import SynapseStore

store = SynapseStore(synapse=syn)

# Thread 1: store user preferences
store.put(("user", "prefs"), "diet", {"value": "vegetarian"})

# Thread 2: retrieve them
item = store.get(("user", "prefs"), "diet")

# Semantic search across all stored data
results = store.search(("user",), query="food preferences")
```

## Why Synapse AI Memory for LangGraph?

- **Survives restarts** — checkpoints persist to local files
- **Cross-thread memory** — agents share knowledge without a database
- **Semantic search** — find relevant memories, not just key lookups
- **Federation** — multiple agents sync memory peer-to-peer
- **Zero infrastructure** — no Redis, no Postgres, no cloud

## Run the example

```bash
python example.py
```
