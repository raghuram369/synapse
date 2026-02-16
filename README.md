# 🧠 Synapse AI Memory

**Your AI's memory. Private. Portable. Federated.**

```bash
pip install synapse-ai-memory
```

![Version](https://img.shields.io/badge/version-0.9.0-blue) ![Tests](https://img.shields.io/badge/tests-420%20passing-brightgreen) ![Cloud Calls](https://img.shields.io/badge/cloud%20calls-0-green) ![Speed](https://img.shields.io/badge/recall-fast-lightgrey)

---

## Install → Configure → Remember (30 seconds)

![Install Demo](assets/install.gif)

```bash
pip install synapse-ai-memory
synapse install claude          # auto-configures Claude Desktop
```

```python
from synapse import Synapse
s = Synapse()
s.remember("I prefer dark mode")
s.recall("what theme?")         # → "I prefer dark mode" (6ms)
```

That's it. Your AI now has persistent memory. No API keys. No cloud. No config files.

---

## Import Your Existing Data

![Import Demo](assets/import-policy.gif)

```bash
synapse import chat ~/Downloads/chatgpt_export.json    # ChatGPT
synapse import chat ~/Downloads/claude_export.json      # Claude
synapse import notes ~/obsidian-vault/                  # Markdown notes
synapse import clipboard                                # Clipboard
```

Bring your history. Every conversation, note, and snippet — indexed and searchable in seconds.

---

## Privacy Presets (One Command)

```bash
synapse policy apply private    # PII redaction + 90-day TTL + auto-forget sensitive topics
synapse policy apply minimal    # Keep tagged memories, prune everything else
synapse policy apply ephemeral  # Auto-delete after session ends
synapse policy apply work       # Team-friendly retention rules
```

No YAML files. No config pages. Pick a preset and go.

---

## Sleep Digest — Your Memory's Daily Report

![Sleep Digest](assets/sleep-digest.gif)

```bash
synapse sleep --digest
```

Sleep runs a full maintenance cycle and gives you a human-readable report: what got promoted, patterns discovered, contradictions found, hot topics, cleanup stats, and actionable suggestions.

---

## Chat-Native Memory (`/mem`)

![Mem Commands](assets/mem-commands.gif)

Drop memory into any chat runtime with a tiny command surface:

```
/mem remember I'm working on Project Horizon
/mem recall what project?
/mem contradict
/mem timeline
/mem sleep
/mem forget sensitive-topic
/mem stats
```

Works in Telegram, Discord, OpenClaw, NanoClaw, and any chat-native agent shell.

---

## MCP Memory Appliance

![MCP Appliance](assets/mcp-appliance.gif)

```bash
synapse serve --db ~/.synapse/store              # stdio MCP server
synapse serve --http --port 8765                  # HTTP JSON-RPC (localhost)
synapse doctor --db ~/.synapse/store              # health check
synapse inspect --db ~/.synapse/store --json      # tool catalog + stats
```

8 tools that make any agent 2x smarter: `remember`, `compile_context`, `timeline`, `what_changed`, `contradictions`, `fact_history`, `sleep`, `stats`.

---

## Capture Anything

![Capture Demo](assets/capture.gif)

```bash
synapse clip "Meeting notes: Q3 budget approved"     # inline text
echo "important context" | synapse clip --tag project:x  # pipe from anywhere
synapse watch --clipboard                             # auto-capture clipboard
synapse remember --review "sensitive info"            # routes to review queue
```

Everything you clip is indexed, searchable, and tagged. The clipboard watcher runs until Ctrl+C — every new copy goes to memory.

---

## Review Before Saving (Privacy Mode)

![Review Queue](assets/review-queue.gif)

```bash
synapse review list                  # see what's pending
synapse review approve 3             # approve one item
synapse review approve all           # approve everything
synapse review reject 5              # discard an item
synapse review count                 # just the count
```

Route memories through a pending inbox with `--review`. Nothing gets saved until you approve it. The inspector dashboard shows pending items too.

---

## Runs Forever (Autostart)

```bash
synapse service install              # autostart on login (launchd/systemd)
synapse service status               # check if running
synapse service uninstall            # remove autostart
```

Memory becomes a background OS capability. Start once, forget about it.

---

## Undo Everything

```bash
synapse uninstall claude             # restore Claude config from backup
synapse uninstall openclaw           # remove skill folder
synapse uninstall all                # clean slate
```

Every install is reversible. Consumers trust tools they can undo.

---

## Signed Artifacts

```bash
synapse sign memory-export.brain     # SHA-256 signature
synapse verify memory-export.brain   # verify integrity
```

Brain packs and card exports are auto-signed. Share with confidence — recipients can verify nothing was tampered with.

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
print(report.to_digest())       # human-readable summary
```

No API keys. No cloud. No setup. Just `pip install` and go.

---

## Why Synapse AI Memory?

- 🧠 **Multi-index retrieval** — BM25 + concept graph + temporal + episodes + activation spreading, fused into one recall
- 🕸️ **Structured knowledge graph** — zero-LLM triple extraction (SPO + polarity/tense/confidence) indexed per memory
- 🔍 **Dual-path GraphRAG** — combine BM25 with multi-hop activation spreading for multi-hop retrieval
- ✅ **Truth maintenance** — contradiction detection + belief versioning with provenance and evidence chains
- ⏳ **Bitemporal memory** — store `observed_at`, `valid_from`, `valid_to` and query with `as_of`, `during`, `latest`
- 💤 **Sleep maintenance** — consolidation, promotion, pattern mining, pruning, graph cleanup, community refresh
- 📦 **Portable `.synapse` files** — export, import, merge, diff — your memory is a file you own
- 🌐 **Federation** — P2P agent memory sync via Merkle trees and vector clocks
- ✂️ **Forgetting + privacy** — TTL, topic-forget, redaction, GDPR delete, policy presets
- 🔒 **Privacy-first** — zero cloud calls, zero telemetry. Your data never leaves your machine
- 🔧 **One-line install** — `synapse install claude/openclaw` auto-configures everything
- 📥 **Importers** — ChatGPT, Claude, WhatsApp, notes, clipboard, JSONL, CSV
- 🎛️ **Policy presets** — minimal, private, work, ephemeral — no config needed
- 🧰 **MCP appliance** — 8-tool surface with `serve`, `doctor`, `inspect`
- 🧠 **Brain packs + checkpoints** — share `.brain` packs, checkpoint/restore like Git
- 🗂️ **ContextPack cards** — deterministic, replayable context snapshots
- 🤖 **Chat-native `/mem` DSL** — works in Telegram, Discord, OpenClaw, NanoClaw
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

## Brain Packs (`synapse pack`)

Brain packs are the share object for Synapse: "Here is my project brain pack."

```bash
# Build a topic pack for the last 30 days
synapse pack --topic "project-x" --range 30d --db ~/.synapse/synapse_store

# Replay a pack to preview what would be injected
synapse pack --replay project-x_2026-02-16.brain --db ~/.synapse/synapse_store

# Compare two packs
synapse pack --diff sprint1.brain sprint2.brain
```

`.brain` files are portable, diffable, and replayable for handoffs between agents/teammates.

---

## Checkpointable Memory (`synapse checkpoint`)

Git for your agent's brain.

```bash
# Create a named snapshot
synapse checkpoint create "before refactor" --db ~/.synapse/synapse_store

# Compare two checkpoints (facts changed, beliefs evolved, contradictions)
synapse checkpoint diff before-refactor after-refactor --db ~/.synapse/synapse_store

# Restore a checkpoint
synapse checkpoint restore before-refactor --confirm --db ~/.synapse/synapse_store
```

---

## ContextPack Cards

ContextPack cards are deterministic, replayable context snapshots for sharing memory bundles across sessions.

- Deterministic card IDs from memory/evidence signatures
- Replay support to compare "then vs now" context injection
- Markdown rendering for human review
- JSON serialization + compact binary deck format for transport
- Card decks for collections (`synapse card export deck.scdp`)

```bash
# Create a card from a query
synapse card create "What changed in project-x?" --budget 1800 --db ~/.synapse/synapse_store

# Show an existing card
synapse card show card-abc123 --db ~/.synapse/synapse_store
```

---

## Debug & Inspect (CLI)

```bash
synapse why 123 --db ~/.synapse/synapse      # why did I recall this?
synapse graph "vegetarian" --db ~/.synapse     # explore concept graph
synapse conflicts --db ~/.synapse              # list contradictions
synapse beliefs --db ~/.synapse                # current belief state
synapse timeline --db ~/.synapse               # chronological view
synapse stats --db ~/.synapse                  # store health
synapse inspect --web --db ~/.synapse          # local web dashboard
```

---

## Synapse Recall Challenge

![Benchmark](assets/bench.gif)

```bash
synapse bench                                    # run all 3 scenarios
synapse bench --scenario recall                  # just one
synapse bench --output ./my-results --format md  # custom output
```

One command → shareable proof. Outputs `report.md`, `transcript.md`, and `results.json`.

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

### MCP Tool Mode
```python
# 1. pip install synapse-ai-memory
# 2. synapse install claude
# 3. Restart Claude Desktop — done.
```

### Library Mode
```python
from synapse import Synapse
s = Synapse('my-agent')
s.remember('User prefers dark mode')
results = s.recall('what theme?')
context = s.compile_context('user preferences', budget=2000)
```

### Daemon Mode
```bash
synapse up --port 9470
# Any HTTP client:
curl -X POST localhost:9470/tool -d '{"tool":"remember","args":{"content":"User likes jazz"}}'
```

### Chat Mode
```python
from synapse import Synapse
s = Synapse('chat-bot')
if s.command_parser.is_memory_command(user_message):
    response = s.command(user_message)  # /mem remember, /mem recall, etc.
```

### Framework Mode
```python
from synapse import Synapse
from integrations.langchain import SynapseMemory
memory = SynapseMemory(synapse=Synapse('langchain-agent'))
chain.memory = memory  # drop into any LangChain chain
```

See [`integrations/`](integrations/) for LangChain / LangGraph / CrewAI / Claude / OpenAI examples.

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
- 🧪 Tests: `tests/` (420 tests)
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
