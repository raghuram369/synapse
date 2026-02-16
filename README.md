# 🧠 Synapse AI Memory

**Your AI's memory. Private. Portable. Federated.**

![Demo](demo.gif)

```bash
pip install synapse-ai-memory
```

![Version](https://img.shields.io/badge/version-0.6.0-blue) ![Tests](https://img.shields.io/badge/tests-167%20passing-brightgreen) ![Cloud Calls](https://img.shields.io/badge/cloud%20calls-0-green) ![Speed](https://img.shields.io/badge/recall-fast-lightgrey)

---

## The 30-Second Demo

```python
from synapse import Synapse

s = Synapse()

# 1) Remember (with bitemporal validity windows)
s.remember("I'm vegetarian and allergic to shellfish", memory_type="preference")
s.remember("I lived in Austin, TX", valid_from="2024-01-01", valid_to="2024-06-01")
s.remember("I live in Denver, CO", valid_from="2024-06-01")

# 2) Recall (classic or GraphRAG)
hits = s.recall("What should I eat?", retrieval_mode="graph", limit=5)
past = s.recall("Where did I live?", temporal="as_of:2024-03", limit=1)  # -> Austin
latest = s.recall("Where do I live?", temporal="latest", limit=1)        # -> Denver

# 3) Truth maintenance (contradictions + belief versioning)
s.remember("I am not vegetarian", memory_type="preference")  # creates a contradiction
disputed = s.recall("diet rules", show_disputes=True, limit=5)
worldview = s.beliefs()  # current belief versions derived from extracted triples

# 4) Context compiler (LLM-ready ContextPack)
pack = s.compile_context("Recommend a restaurant", budget=1200, policy="balanced")
prompt_injection = pack.to_system_prompt()

# 5) Sleep mode (maintenance cycle)
report = s.sleep(verbose=True)  # consolidate, promote, mine patterns, prune, refresh communities
```

No API keys. No cloud. No setup. Just `pip install` and go.

---

## Why Synapse AI Memory?

- 🧠 **Multi-index retrieval** — BM25 + concept graph + temporal + episodes + activation spreading, fused into one recall
- 🕸️ **Structured knowledge graph** — zero-LLM triple extraction (SPO + polarity/tense/confidence) indexed per memory
- 🔍 **Dual-path GraphRAG** — combine BM25 with multi-hop activation spreading for multi-hop retrieval (`retrieval_mode="graph"`)
- ✅ **Truth maintenance** — contradiction detection + belief versioning with provenance and evidence chains
- ⏳ **Bitemporal memory** — store `observed_at`, `valid_from`, `valid_to` and query with `as_of`, `during`, `latest`
- 💤 **Sleep maintenance** — consolidation, promotion, pattern mining, pruning, graph cleanup, community refresh
- 📦 **Portable `.synapse` files** — export, import, merge, diff — your memory is a file you own. Runtime storage uses `.log` + `.snapshot` files; portable export uses the `.synapse` binary format.
- 🌐 **Federation** — P2P agent memory sync via Merkle trees and vector clocks
- ✂️ **Forgetting + privacy tools** — TTL, topic-forget, redaction, GDPR delete
- 🔒 **Privacy-first** — zero cloud calls, zero telemetry. Your data never leaves your machine (optional local Ollama calls use localhost HTTP).
- ⚡ **Fast local recall** — pure Python, zero dependencies, runs on a Raspberry Pi

---

## Knowledge Graph (Triples + Graph Queries)

Every `remember()` pass extracts structured triples (no LLM required) and indexes them for graph-style queries.

```python
from synapse import Synapse

s = Synapse(":memory:")
m = s.remember("Alice moved to New York. Alice works at Acme Corp.")

# Triples attached to that memory (S, P, O + metadata)
triples = s.triple_index.get_triples_for_memory(m.id)
for t in triples:
    print(t.subject, t.predicate, t.object, t.polarity, t.tense, t.confidence)

# Query by subject/predicate/object (returns triple IDs)
nyc_triple_ids = s.triple_index.query_spo(obj="new york")
```

For retrieval, use `retrieval_mode="graph"` to activate multi-hop neighbors and recover relevant memories that keyword BM25 can miss.

```python
hits = s.recall("Where did Alice relocate?", retrieval_mode="graph", limit=5)
```

---

## Truth Maintenance (Contradictions + Beliefs)

Synapse continuously detects contradictions and can annotate recall with disputes or exclude conflicted memories.

```python
from synapse import Synapse

s = Synapse(":memory:")
s.remember("User is vegetarian", memory_type="preference")
s.remember("User is not vegetarian", memory_type="preference")  # contradiction

conflicts = s.contradictions()  # unresolved contradictions
with_disputes = s.recall("diet", show_disputes=True, limit=5)
clean = s.recall("diet", exclude_conflicted=True, limit=5)
```

Beliefs are versioned facts derived from triples (with provenance back to memory IDs).

```python
worldview = s.beliefs()              # {fact_key -> BeliefVersion}
history = s.belief_history("user")   # versions matching a topic-like filter
```

---

## Sleep & Consolidation

Sleep mode runs a full maintenance cycle to keep memory healthy over time.

```python
from synapse import Synapse

s = Synapse(":memory:")
# ... add memories over time ...
report = s.sleep(verbose=True)
print(report)
```

Sleep includes (high-level): consolidation, promotion (episodic -> semantic), pattern mining, pruning, contradiction scanning, graph cleanup, and community refresh.

---

## Context Compiler (ContextPack)

`compile_context()` compiles recalled memories, a graph slice, summaries, and evidence chains into a compact `ContextPack` for LLM integration.

```python
from synapse import Synapse

s = Synapse(":memory:")
# ... remember a few facts ...
pack = s.compile_context("What should I remember about the user?", budget=1600, policy="balanced")

print(pack.to_compact())
print(pack.to_system_prompt())
payload = pack.to_dict()  # JSON-serializable for tool/agent frameworks
```

Policies: `balanced`, `precise`, `broad`, `temporal`.

---

## Forgetting & Privacy

Forget by topic, redact specific fields, or perform GDPR-style delete.

```python
from synapse import Synapse

s = Synapse(":memory:")
m = s.remember("User SSN is 123-45-6789", metadata={"tags": ["user:42", "pii"]})

s.redact(memory_id=m.id, fields=["content"])        # -> content becomes "[REDACTED]"
s.forget_topic("pii")                               # -> delete topic-related memories
s.gdpr_delete(user_id="42")                         # -> delete memories tagged user:42

# TTL / retention rules (declarative)
s.set_retention_rules([{"tag": "temporary", "ttl_days": 7, "action": "delete"}])
```

---

## Debug & Inspect (CLI)

The debug CLI is designed for answering "why did I recall this?" and "what does the memory graph believe?"

```bash
synapse why 123 --db ~/.synapse/synapse
synapse graph "vegetarian" --db ~/.synapse/synapse
synapse conflicts --db ~/.synapse/synapse
synapse beliefs --db ~/.synapse/synapse
synapse timeline --db ~/.synapse/synapse
synapse stats --db ~/.synapse/synapse
```

## Benchmarks

The `bench/` suite measures retrieval quality (Recall@K, MRR) and reports end-to-end benchmark runtime; it does not claim or measure per-query latency.

```
LOCOMO Benchmark (industry standard)
─────────────────────────────────────
Recall@1    30.4%   (+10.8% vs BM25)
Recall@5    53.5%   (+9.4%  vs BM25)
Recall@10   62.9%   (+9.0%  vs BM25)
MRR         40.6%   (+10.5% vs BM25)

Practical Benchmark
───────────────────
Recall@10   89.1%   on real-world agent conversations
```

Pure Python. No embeddings API. No GPU. These numbers come from indexes alone.

---

## Works With Everything

### Claude / Anthropic

```python
from synapse import Synapse
from integrations.claude import SynapseClaudeMemory

memory = SynapseClaudeMemory(synapse=Synapse("claude_memory"))
context = memory.get_context("Can you recommend a restaurant?")
# → Recalls shellfish allergy from 3 weeks ago, suggests safe options
```

### OpenAI / ChatGPT

```python
from integrations.openai import SynapseGPTMemory

memory = SynapseGPTMemory(synapse=Synapse("gpt_memory"))
context = memory.get_context("What should I have for lunch?")
# → Recalls vegetarian preference, suggests accordingly
```

### LangChain / LangGraph / CrewAI

```python
from integrations.langchain import SynapseMemory, SynapseRetriever
from integrations.langgraph import SynapseStore, SynapseCheckpointer
from integrations.crewai import SynapseCrewMemory
# Drop-in replacements. See integrations/ for full docs.
```

Tool-use mode also supported — let your AI decide what to remember. See [`integrations/`](integrations/) for full examples.

---

## Research-Backed

Our architecture didn't come from vibes. It matches what the research says works:

- 📄 **"The AI Hippocampus"** (Jan 2026) — describes the exact multi-index architecture Synapse AI Memory implements
- 📄 **"Graph-based Agent Memory"** (Feb 2026) — concept graphs are the frontier; we shipped ours in v0.2
- 📄 **"Memory in the Age of AI Agents"** (Dec 2025) — validates temporal + concept approach over pure embeddings

---

## Architecture

```
Remember / Ingest Path
────────────────────────────────────────────────────────────────────
Text
  ├─ Entity normalization (aliases, lemmatization, coref)
  ├─ Concept extraction -> Concept Graph
  ├─ Triple extraction (SPO + polarity/tense/confidence) -> Triple Index
  ├─ Contradiction detection (polarity / exclusion / numeric / temporal)
  └─ Belief versioning (fact chains with provenance)

Recall Path (classic + GraphRAG)
────────────────────────────────────────────────────────────────────
                        ┌─────────────┐
                        │    Query    │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        ┌──────────┐   ┌─────────────┐   ┌───────────┐
        │   BM25   │   │  Concept    │   │ Temporal   │
        │  Index   │   │   Graph     │   │  Filter    │
        └────┬─────┘   └──────┬──────┘   └─────┬─────┘
              │                │                 │
              ▼                ▼                 ▼
        ┌──────────┐   ┌─────────────┐   ┌───────────┐
        │ Keyword  │   │ Activation  │   │  Bitemp.   │
        │  Match   │   │ Spreading   │   │  Windows   │
        └────┬─────┘   └──────┬──────┘   └─────┬─────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                     ┌──────────────────┐
                     │  Score Fusion +  │
                     │  Episode Groups  │
                     └────────┬─────────┘
                              ▼
                ┌──────────────────────────┐
                │ Conflict-Aware Recall +  │
                │ Evidence Chains          │
                └────────┬─────────────────┘
                         ▼
                 ┌──────────────────┐
                 │  ContextCompiler │
                 │  -> ContextPack  │
                 └──────────────────┘
```

Multiple indexes. One fused result. No LLM in the loop.

---

## Quick Links

- 📦 PyPI: `synapse-ai-memory`
- 🧪 Tests: `tests/` (167 core tests)
- 🔌 Integrations: `integrations/`
- 🧰 Examples: `examples/`
- 📈 Benchmarks: `bench/`
- 🧠 Triples + KG: `triples.py`, `graph_retrieval.py`
- ✅ Truth maintenance: `contradictions.py`, `belief.py`, `evidence.py`
- 💤 Sleep mode: `sleep.py`, `communities.py`
- 🔒 Security policy: `SECURITY.md`
- 🔁 Mem0 compatibility layer (migration shim): `synapse/compat/mem0.py`

---

## License

MIT — see [LICENSE](LICENSE).

Built with 🧠 by [@raghuram369](https://github.com/raghuram369)
