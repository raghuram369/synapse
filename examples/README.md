# Synapse AI Memory Examples

- `examples/personal_prefs.py` — Personal Preferences  
  Stores user preference memories (theme, language, dietary restrictions, music taste), runs natural-language recall queries, and prints `explain=True` score breakdowns.

- `examples/time_travel.py` — Time Travel  
  Stores facts that evolve over time (location, job, preferences), uses `fact_history()` to inspect changes, and runs temporal queries (e.g. `as_of` / month filters). Also shows `consolidate()` compressing repeated preference patterns.

- `examples/federation_sync.py` — Federation Sync  
  Creates two Synapse AI Memory agents, syncs through a localhost-only federation server, shows shared memories arriving on the peer, and demonstrates namespace-based selective sharing.

- `examples/knowledge_graph_truth.py` — Knowledge Graph + Truth Maintenance  
  Demonstrates zero-LLM triple extraction + queries, contradiction detection, belief versioning, and GraphRAG retrieval (`retrieval_mode="graph"`).

- `examples/context_pack_compiler.py` — ContextPack Compiler  
  Demonstrates `compile_context(query, budget, policy)` and how to render `ContextPack` for LLM prompt injection.

- `examples/forgetting_privacy.py` — Forgetting + Privacy  
  Demonstrates TTL/retention rules, `forget_topic`, redaction, and GDPR-style deletion.
