# 🧠 Synapse

**Your AI's memory. Private. Portable. Federated.**

![Demo](demo.gif)

```bash
pip install synapse-ai-memory
```

![Version](https://img.shields.io/badge/version-0.3.0-blue) ![Tests](https://img.shields.io/badge/tests-152%20passing-brightgreen) ![API Calls](https://img.shields.io/badge/API%20calls-0-green) ![Speed](https://img.shields.io/badge/recall-%3C1ms-lightgrey)

---

## The 30-Second Demo

```python
from synapse import Synapse

s = Synapse()

# Your AI remembers
s.remember("I'm vegetarian and allergic to shellfish")
s.remember("I live in Austin, TX")
s.remember("I moved to Denver, CO")  # supersedes Austin

# Your AI recalls — even without exact keywords
results = s.recall("What should I eat?")       # finds dietary info via concept graph
results = s.recall("Where do I live?", temporal="2024-01")  # time-travel: "Austin"

# Your AI evolves
s.consolidate()        # distill repeated patterns into stronger memories
print(s.hot_concepts())  # see what's top-of-mind

# Your AI is portable
s.export("my_memory.synapse")  # take it anywhere

# Your AI connects
s.serve(port=9470)  # other agents can sync with you
```

No API keys. No cloud. No setup. Just `pip install` and go.

---

## Why Synapse?

- 🧠 **5 neuroscience-inspired indexes** — BM25 + concept graph + temporal decay + episodes + activation spreading, fused into one recall
- ⏳ **Time-travel queries** — ask "what was true in March 2024?" and get the answer from then
- 🔄 **Memory consolidation** — repeated facts merge into stronger patterns, like sleep does for your brain
- 📦 **Portable `.synapse` files** — export, import, merge, diff — your memory is a file you own
- 🌐 **Federation** — P2P agent memory sync via Merkle trees and vector clocks
- ✂️ **Smart pruning** — forgetting is a feature, not a bug. Weak memories fade naturally
- 🔒 **Privacy-first** — zero API calls, zero cloud, zero telemetry. Your data never leaves your machine
- ⚡ **Sub-millisecond recall** — pure Python, zero dependencies, runs on a Raspberry Pi

---

## Benchmarks

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

- 📄 **"The AI Hippocampus"** (Jan 2026) — describes the exact multi-index architecture Synapse implements
- 📄 **"Graph-based Agent Memory"** (Feb 2026) — concept graphs are the frontier; we shipped ours in v0.2
- 📄 **"Memory in the Age of AI Agents"** (Dec 2025) — validates temporal + concept approach over pure embeddings

---

## Architecture

```
                        ┌─────────────┐
                        │    Query    │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        ┌──────────┐   ┌─────────────┐   ┌───────────┐
        │   BM25   │   │  Concept    │   │ Temporal   │
        │  Index   │   │   Graph     │   │  Index     │
        └────┬─────┘   └──────┬──────┘   └─────┬─────┘
              │                │                 │
              ▼                ▼                 ▼
        ┌──────────┐   ┌─────────────┐   ┌───────────┐
        │ Keyword  │   │ Activation  │   │  Recency   │
        │  Match   │   │ Spreading   │   │   Boost    │
        └────┬─────┘   └──────┬──────┘   └─────┬─────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                     ┌──────────────────┐
                     │  Score Fusion +  │
                     │  Episode Groups  │
                     └────────┬─────────┘
                              ▼
                       ┌────────────┐
                       │  Results   │
                       └────────────┘
```

Five indexes. One fused result. No LLM in the loop.

---

## Quick Links

| | |
|---|---|
| 📦 [PyPI](https://pypi.org/project/synapse-ai-memory/) | 🔌 [Integrations](integrations/) |
| 📖 [Docs](docs/) | 🌐 [Chrome Extension](extension/) |
| 🧪 [Tests](tests/) | 🤝 [Contributing](CONTRIBUTING.md) |

---

## License

MIT — see [LICENSE](LICENSE).

Built with 🧠 by [@raghuram369](https://github.com/raghuram369)
