# Synapse — A neuroscience-inspired memory database for AI agents

**Zero LLM calls. Pure Python. Runs anywhere.**

![Stats](https://img.shields.io/badge/Recall@10-62.9%25-brightgreen) ![Improvement](https://img.shields.io/badge/vs%20BM25-+9%25-blue) ![API Calls](https://img.shields.io/badge/API%20calls-0-green) ![Speed](https://img.shields.io/badge/recall-%3C1ms-lightgrey)

## 🧠 Why Synapse?

Synapse delivers **competitive recall quality at zero cost, zero dependencies**. Pure Python, runs on anything from a Raspberry Pi to a server. It IS the database, like Redis or Postgres.

**Key benefits:**
- **Zero cost**: No API fees or platform charges
- **Zero dependencies**: Pure Python, no external services required
- **Fast**: Sub-millisecond recall with in-memory indexes
- **Self-hosted**: Complete control over your data
- **Offline capable**: Works without internet connection

## 🚀 Quick Start

```python
from synapse import Synapse
s = Synapse()
s.remember("I prefer vegetarian food")
results = s.recall("What are my dietary preferences?")
```

That's it! No API keys, no setup, no dependencies.

## 🏗️ Architecture

Synapse uses a neuroscience-inspired **three-stage memory system**:

```
Query → BM25 (Primary) → Concept Graph → Local Embeddings → Fused Results
         ↓               ↓                ↓
    Keyword match    Conceptual links   Semantic similarity
```

### The Science Behind It

Just like your brain, Synapse implements:

- **Temporal decay**: Recent memories are weighted higher
- **Synaptic plasticity**: Frequently accessed connections get stronger
- **Episode grouping**: Related memories cluster together automatically  
- **Activation spreading**: Concepts activate related concepts
- **Supersession**: New information can replace outdated facts

### Five Native Indexes

1. **InvertedIndex**: BM25 keyword search (primary)
2. **ConceptGraph**: 50+ concept categories with weighted edges
3. **EdgeGraph**: Temporal relationships between memories
4. **TemporalIndex**: Time-based decay and recency boosting
5. **EpisodeIndex**: Auto-grouping of related memories

## 📊 Benchmark Results

**LOCOMO Benchmark** (industry standard):

BM25 is the industry-standard baseline used by LOCOMO. Synapse beats it by 9% through neuroscience-inspired indexing.

| Metric | Synapse V2 | BM25 Baseline | Improvement |
|--------|------------|---------------|-------------|
| **Recall@1** | **30.4%** | 27.4% | **+10.8%** |
| **Recall@5** | **53.5%** | 48.9% | **+9.4%** |
| **Recall@10** | **62.9%** | 57.7% | **+9.0%** |
| **MRR** | **40.6%** | 36.7% | **+10.5%** |

**Performance by Category** (Recall@10 vs BM25):

| Category | Synapse V2 | BM25 Baseline | Improvement |
|----------|------------|---------------|-------------|
| **Single-hop factual** | **54.3%** | 40.1% | **+35.4%** |
| **Temporal reasoning** | **67.9%** | 62.3% | **+9.0%** |
| **Multi-hop reasoning** | **37.5%** | 30.2% | **+24.1%** |
| **Open-domain** | **64.8%** | 62.1% | **+4.4%** |
| **Adversarial** | **66.8%** | 63.2% | **+5.7%** |

## 📈 Comparison

Focus on architecture and cost, not benchmark scores:

| Feature | Synapse | Mem0 | Zep |
|---------|---------|------|-----|
| **LLM required** | No (optional) | Yes (every op) | Yes |
| **Dependencies** | 0 (pure Python) | Many | Many |
| **Cost** | $0 | API costs | Platform fee |
| **Self-hosted** | Yes | Partial | No |
| **Runs offline** | Yes | No | No |

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
- **🐍 Zero Dependencies**: Pure Python, runs anywhere
- **🧪 Tested**: 45+ unit tests covering all major functionality

## 🛠️ Advanced Usage

### Basic Operations

```python
from synapse import Synapse

db = Synapse()

# Remember facts
db.remember("The capital of France is Paris", tags=["geography"])
db.remember("Paris has 2.2 million people", tags=["demographics"])

# Recall with context
results = db.recall("French capital city", limit=5)
for memory in results:
    print(f"{memory.content} (score: {memory.score:.3f})")

# Link related concepts
db.link("Paris", "Eiffel Tower", weight=0.8)
```

### 🧪 Concept Extraction (Experimental)

**Note**: Requires Ollama for LLM-powered extraction

```python
# Extract concepts automatically (requires Ollama)
concepts = db.concepts("Paris is known for art, cuisine, and fashion")
# → [("Paris", "LOCATION"), ("art", "ABSTRACT"), ("cuisine", "FOOD"), ...]

# Add custom concept patterns
db.entity_graph.add_pattern("CRYPTO", [
    r'\b(bitcoin|ethereum|dogecoin|crypto)\b'
])

db.remember("Bitcoin hit $100k today")
concepts = db.concepts("Bitcoin news")
# → [("Bitcoin", "CRYPTO"), ...]
```

### Temporal Queries

```python
# Find recent memories (last 24 hours)
recent = db.recall("project update", temporal_boost=True, max_age=86400)

# Find memories from specific time range  
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
old_memories = db.recall("meeting notes", before=yesterday)
```

### Episode Linking

```python
# Memories automatically group into episodes
db.remember("Started working on the presentation")
db.remember("Added slides about market research") 
db.remember("Finished the conclusion slide")

# These will be linked as part of the same episode
episodes = db.get_episodes()
```

## 🖥️ Daemon Mode

Run Synapse as a background service with REST API:

### Setup

```bash
# Start daemon
python synapsed.py --port 8080 --data-dir ./synapse_data

# Or use the CLI
python cli.py daemon start --port 8080
```

### HTTP API

```bash
# Remember
curl -X POST http://localhost:8080/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Meeting with John at 3pm", "tags": ["meetings"]}'

# Recall
curl -X POST http://localhost:8080/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "John meeting", "limit": 5}'
```

## 🏃‍♂️ Running the Tests

```bash
python -m unittest test_synapse test_entity_graph test_episode_graph
```

All 45 tests should pass with detailed output of the 5 indexes and concept extraction.

## 📦 Installation

```bash
# For now, clone and import (pip package coming soon)
git clone https://github.com/raghuram369/synapse.git
cd synapse

# No dependencies to install!
python
>>> from synapse import Synapse
>>> s = Synapse()
>>> # Ready to use!
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is the initial release! We'd love contributors to help with:

- 📦 PyPI packaging (`pip install synapse-ai-memory`)
- 🔌 Framework integrations (LangChain, CrewAI, etc.)
- 🧪 More benchmarks and test cases
- 📖 Documentation and tutorials
- ⚡ Performance optimizations
- 🔍 Additional concept categories

## 🚀 What's Next

- [ ] PyPI package (`pip install synapse-ai-memory`)
- [ ] REST API server mode
- [ ] Distributed/clustered deployments  
- [ ] More embedding providers
- [ ] Graph visualization tools
- [ ] Framework integrations

---

**Synapse**: Memory that works like your brain, not like a chatbot. 🧠