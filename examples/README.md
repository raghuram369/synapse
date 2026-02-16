# Synapse AI Memory Examples

- `examples/personal_prefs.py` — Personal Preferences  
  Stores user preference memories (theme, language, dietary restrictions, music taste), runs natural-language recall queries, and prints `explain=True` score breakdowns.

- `examples/time_travel.py` — Time Travel  
  Stores facts that evolve over time (location, job, preferences), uses `fact_history()` to inspect changes, and runs a temporal query for March 2024. Also shows `consolidate()` compressing repeated preference patterns.

- `examples/federation_sync.py` — Federation Sync  
  Creates two Synapse AI Memory agents, syncs through a localhost-only federation server, shows shared memories arriving on the peer, and demonstrates namespace-based selective sharing.
