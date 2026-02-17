# LangGraph Starter Kit

## Repo template pointers

- Repo scaffold (starter): _placeholder_ `https://github.com/synapse-ai-memory/langgraph-starter`
- In-repo implementation example: `integrations/langgraph/example.py`
- Adapters: `integrations/langgraph/checkpointer.py`, `integrations/langgraph/memory_store.py`
- README guide source: `integrations/langgraph/README.md`

## Minimum starter layout

- `graph.py` with:
  - `Synapse` + `SynapseCheckpointer`
  - one or more stateful nodes
  - thread-aware `configurable.thread_id`
- `requirements.txt` with `synapse-ai-memory`, `langgraph`
- `README.md` with run steps + thread IDs + persistence path

## Template run

```bash
python -m venv .venv
source .venv/bin/activate
pip install synapse-ai-memory langgraph
python graph.py
```

## Useful starter knobs

- Persist thread state under a stable path (e.g., `./.synapse`).
- Wire recall/store calls behind retrieval utility nodes, not inline model calls.
- Add a test using a second thread ID to prove cross-thread memory continuity.
