# Synapse — A neuroscience-inspired memory database for AI agents

**Zero LLM calls. Pure Python. Runs anywhere. Export, share, federate.**

![Version](https://img.shields.io/badge/version-0.2.0-blue) ![Stats](https://img.shields.io/badge/Recall@10-62.9%25-brightgreen) ![Improvement](https://img.shields.io/badge/vs%20BM25-+9%25-blue) ![API Calls](https://img.shields.io/badge/API%20calls-0-green) ![Speed](https://img.shields.io/badge/recall-%3C1ms-lightgrey) ![Tests](https://img.shields.io/badge/tests-177%20passing-brightgreen)

![Demo](demo.gif)

## 🧠 Why Synapse?

Synapse delivers **competitive recall quality at zero cost, zero dependencies**. Pure Python, runs on anything from a Raspberry Pi to a server. It IS the database — like Redis or Postgres, but for AI memory.

**Three pillars, one package:**

| Pillar | What it does |
|--------|-------------|
| **Memory Engine** | BM25 + concept graphs + embeddings, sub-millisecond recall |
| **Portable Format** | Binary `.synapse` files for export, import, merge, diff |
| **Federation** | HTTP peer-to-peer sync with Merkle trees and vector clocks |

## 🚀 Quick Start

```python
from synapse import Synapse

s = Synapse()

# Remember
s.remember("I prefer vegetarian food")
s.remember("Meeting with Sarah at 3pm", memory_type="event")

# Recall
results = s.recall("dietary preferences")

# Export & Import
s.export("my_memories.synapse")

s2 = Synapse()
s2.load("my_memories.synapse")

# Merge from another agent
s2.merge("other_agent.synapse")
```

That's it! No API keys, no setup, no dependencies.

## 📦 Installation

```bash
pip install synapse-ai-memory
```

Or clone and use directly:

```bash
git clone https://github.com/raghuram369/synapse.git
cd synapse
python3 -c "from synapse import Synapse; s = Synapse(); print('Ready!')"
```

## 📁 Portable Format

Every Synapse instance can export and import `.synapse` files — a binary format with CRC integrity, provenance tracking, and streaming reads.

### Python API

```python
s = Synapse("my_data")

# Full export
s.export("backup.synapse")

# Filtered export
s.export("recent.synapse", since="2024-06-01")
s.export("food.synapse", concepts=["food", "cooking"])

# Import into a fresh instance
s2 = Synapse()
s2.load("backup.synapse")  # with automatic deduplication

# Merge without overwriting
s2.merge("other.synapse")                          # newer wins
s2.merge("other.synapse", conflict_resolution="keep_both")
```

### CLI

```bash
synapse export backup.synapse --db ./synapse_data
synapse import backup.synapse --db ./synapse_data
synapse inspect backup.synapse
synapse merge a.synapse b.synapse -o combined.synapse
synapse diff a.synapse b.synapse
```

## 🌐 Federation

Synapse nodes can sync memories peer-to-peer over HTTP. Uses Merkle trees for efficient delta detection and vector clocks for conflict resolution.

### Python API

```python
s = Synapse("my_data")

# Start a federation server
s.serve(port=9470)

# Sync with peers
s.push("http://peer:9470")
s.pull("http://peer:9470")
s.sync("http://peer:9470")  # bidirectional

# Manage peers
s.add_peer("http://peer:9470", token="secret")
s.share("public")  # selective namespace sharing
```

### CLI

```bash
synapse serve --fed-port 9470 --share public
synapse push http://peer:9470
synapse pull http://peer:9470
synapse sync http://peer:9470
synapse peers http://peer:9470
```

### Federation Features

- **Content-addressed storage** — memories identified by SHA-256 hash (like git)
- **Merkle tree sync** — only transfer what's different (256-bucket fanout)
- **Vector clocks** — track causality across distributed nodes
- **Namespace filtering** — share only what you want (`public`, `research`, etc.)
- **Bearer token auth** — secure peer-to-peer communication
- **LAN discovery** — automatic peer discovery via UDP broadcast
- **Binary wire format** — uses the `.synapse` portable format for efficient transfer

## 🏗️ Architecture

Synapse uses a neuroscience-inspired **three-stage memory system**:

```
Query → BM25 (Primary) → Concept Graph → Local Embeddings → Fused Results
         ↓               ↓                ↓
    Keyword match    Conceptual links   Semantic similarity
```

### Five Native Indexes

1. **InvertedIndex**: BM25 keyword search (primary)
2. **ConceptGraph**: 50+ concept categories with weighted edges
3. **EdgeGraph**: Temporal relationships between memories
4. **TemporalIndex**: Time-based decay and recency boosting
5. **EpisodeIndex**: Auto-grouping of related memories

### The Science Behind It

- **Temporal decay**: Recent memories weighted higher
- **Synaptic plasticity**: Frequently accessed connections get stronger
- **Episode grouping**: Related memories cluster together automatically
- **Activation spreading**: Concepts activate related concepts
- **Supersession**: New information replaces outdated facts

## 📊 Benchmark Results

**LOCOMO Benchmark** (industry standard):

| Metric | Synapse V2 | BM25 Baseline | Improvement |
|--------|------------|---------------|-------------|
| **Recall@1** | **30.4%** | 27.4% | **+10.8%** |
| **Recall@5** | **53.5%** | 48.9% | **+9.4%** |
| **Recall@10** | **62.9%** | 57.7% | **+9.0%** |
| **MRR** | **40.6%** | 36.7% | **+10.5%** |

## 📈 Comparison

| Feature | Synapse | Mem0 | Zep |
|---------|---------|------|-----|
| **LLM required** | No (optional) | Yes (every op) | Yes |
| **Dependencies** | 0 (pure Python) | Many | Many |
| **Cost** | $0 | API costs | Platform fee |
| **Self-hosted** | Yes | Partial | No |
| **Runs offline** | Yes | No | No |
| **Portable export** | ✅ `.synapse` files | ❌ | ❌ |
| **Federation** | ✅ P2P sync | ❌ | ❌ |

## ✨ Features

- **🔍 Hybrid Search**: BM25 + concept graphs + optional embeddings
- **⏰ Temporal Decay**: Recent memories weighted higher automatically
- **🧠 Concept Linking**: Auto-extracted concepts with weighted relationships
- **📚 Episode Grouping**: Related memories cluster together
- **🔄 Activation Spreading**: Query expansion through concept networks
- **📝 Supersession**: New facts can replace outdated information
- **💾 Persistent**: Redis-style AOF/RDB with automatic recovery
- **⚡ Fast**: Sub-millisecond recall, pure in-memory indexes
- **🔌 Optional Embeddings**: Ollama integration for semantic similarity
- **📁 Portable Format**: Binary `.synapse` files with CRC, provenance, streaming
- **🌐 Federation**: P2P sync via Merkle trees, vector clocks, namespace filtering
- **🔐 Auth**: Bearer token authentication for federation peers
- **🐍 Zero Dependencies**: Pure Python, runs anywhere
- **🧪 Tested**: 125 unit tests covering all major functionality

## 🔗 Framework Integrations

Drop-in memory for the frameworks you already use. Each integration lives in [`integrations/`](integrations/) with its own README, examples, and tests.

| Framework | Integration | Key Classes |
|-----------|------------|-------------|
| **[LangChain](integrations/langchain/)** | Chat history, retriever, memory backend | `SynapseMemory`, `SynapseChatMessageHistory`, `SynapseRetriever` |
| **[LangGraph](integrations/langgraph/)** | Checkpointer, cross-thread memory store | `SynapseCheckpointer`, `SynapseStore` |
| **[CrewAI](integrations/crewai/)** | Shared memory for multi-agent crews | `SynapseCrewMemory` |
| **[Claude / Anthropic](integrations/claude/)** | Auto-memory wrapper + tool_use schema | `SynapseClaudeMemory`, `synapse_tools` |
| **[OpenAI / ChatGPT](integrations/openai/)** | Auto-memory wrapper + function calling | `SynapseGPTMemory`, `synapse_functions` |

### Claude / Anthropic

Give Claude persistent memory across conversations — your data stays local, never hits Anthropic's servers for storage.

```python
import anthropic
from synapse import Synapse
from integrations.claude import SynapseClaudeMemory

synapse = Synapse("claude_memory")
memory = SynapseClaudeMemory(synapse=synapse)
client = anthropic.Anthropic()

# Conversation 1
messages = [{"role": "user", "content": "I'm allergic to shellfish and I love hiking in Colorado"}]
context = memory.get_context(messages[-1]["content"])  # Recalls relevant memories
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=f"You have memory of past conversations:\n{context}",
    messages=messages
)
memory.save_exchange(messages[-1]["content"], response.content[0].text)

# Conversation 2 — days later, Claude remembers
messages = [{"role": "user", "content": "Can you recommend a restaurant for tonight?"}]
context = memory.get_context(messages[-1]["content"])
# → Recalls: shellfish allergy, suggests safe restaurants
```

**As a Claude tool** — let Claude decide what to remember:

```python
from integrations.claude import synapse_tools, handle_synapse_tool

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=synapse_tools,  # remember, recall, forget tools
    messages=[{"role": "user", "content": "Remember that my daughter's birthday is March 15"}]
)
# Claude calls the remember tool → stored locally in Synapse
```

### OpenAI / ChatGPT

Same privacy-first memory for GPT. Your conversations build persistent context without OpenAI storing anything.

```python
import openai
from synapse import Synapse
from integrations.openai import SynapseGPTMemory

synapse = Synapse("gpt_memory")
memory = SynapseGPTMemory(synapse=synapse)
client = openai.OpenAI()

# Conversation 1
user_msg = "I'm a vegetarian and I work at Google on the Search team"
context = memory.get_context(user_msg)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"You have memory of past conversations:\n{context}"},
        {"role": "user", "content": user_msg}
    ]
)
memory.save_exchange(user_msg, response.choices[0].message.content)

# Conversation 2 — GPT knows your preferences
user_msg = "What should I have for lunch?"
context = memory.get_context(user_msg)
# → Recalls: vegetarian preference, suggests accordingly
```

**As GPT function calls** — GPT manages its own memory:

```python
from integrations.openai import synapse_functions, handle_synapse_function

response = client.chat.completions.create(
    model="gpt-4o",
    functions=synapse_functions,  # remember, recall, forget
    messages=[{"role": "user", "content": "What do you remember about my food preferences?"}]
)
# GPT calls recall → searches local Synapse → responds with context
```

### LangChain / LangGraph / CrewAI

```python
# LangChain — drop-in memory + retriever
from integrations.langchain import SynapseMemory, SynapseRetriever
memory = SynapseMemory(data_dir="./memory", k=5)
retriever = SynapseRetriever(data_dir="./memory")

# LangGraph — checkpointing + cross-thread memory
from integrations.langgraph import SynapseStore, SynapseCheckpointer
store = SynapseStore(data_dir="./agent_memory")
checkpointer = SynapseCheckpointer(data_dir="./checkpoints")

# CrewAI — shared memory across agent crews
from integrations.crewai import SynapseCrewMemory
crew_mem = SynapseCrewMemory(synapse=synapse, crew_id="research-team")
```

## 🖥️ Daemon Mode

```bash
synapsed --port 8080 --data-dir ./synapse_data
synapse remember "Meeting with John at 3pm"
synapse recall "John meeting"
```

## 🏃‍♂️ Running the Tests

```bash
python3 -m unittest test_synapse test_portable test_federation test_entity_graph test_episode_graph -v
```

All 125 tests pass.

## 📝 License

MIT License — see [LICENSE](LICENSE) file for details.

---

**Synapse**: Memory + Portability + Federation. One package, three pillars. 🧠
