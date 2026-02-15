# Code Review — Synapse V2
**Reviewer:** Codex 5.3 (gpt-5.3-codex) + Kai  
**Date:** 2026-02-14  

## Summary: SHIP WITH FIXES

The core engine (synapse.py, storage.py, indexes.py) is solid — 45 tests passing, clean architecture, good performance. The integrations need work before they're production-ready. Fix the critical issues below before Tuesday launch.

**Code Quality Score: 7/10**
- Core engine: 8.5/10
- Integrations: 5/10
- Tests: 6/10

---

## 🔴 Critical Issues (must fix before launch)

### 1. Integration tests don't run (0 tests discovered)
`python3 -m unittest discover -s integrations -p "test_*.py"` returns **0 tests**. The test classes in `integrations/test_integrations.py` don't inherit from `unittest.TestCase` — they're plain classes. unittest discovery won't find them.

**Fix:** Change all test classes to inherit from `unittest.TestCase` and use `self.assert*` methods instead of bare `assert`.

### 2. `extractor.py` imports `requests` — breaks "zero dependencies" claim
Line 9: `import requests`. This is a third-party package. The rest of the codebase uses `urllib.request` for HTTP calls (see `embeddings.py`). Anyone who `pip install synapse-ai-memory` and tries `extract=True` will get `ModuleNotFoundError: No module named 'requests'`.

**Fix:** Replace `requests` with `urllib.request` like `embeddings.py` does, OR add `requests` to optional dependencies and wrap the import in try/except.

### 3. `SynapseChatMessageHistory.clear()` destroys ALL data (langchain.py:225-231)
The `clear()` method closes the entire Synapse instance and creates a new one. If multiple sessions share the same persistent `data_dir`, this nukes everything — not just the current session's messages. 

**Fix:** Selectively forget memories for the current `session_id` only.

### 4. `SynapseCheckpointer` doesn't subclass `BaseCheckpointSaver` (langgraph.py)
The README claims "drop-in replacement" but `SynapseCheckpointer` is a standalone class. It won't work with LangGraph's `compile(checkpointer=...)` because LangGraph expects `BaseCheckpointSaver` interface.

**Fix:** Either subclass `BaseCheckpointSaver` (when langgraph is installed) or clearly document this as "LangGraph-compatible" not "drop-in".

### 5. Checkpoint retrieval is unreliable (langgraph.py)
`get()` uses `self.synapse.recall("checkpoint thread_id:{thread_id}")` — this relies on BM25 keyword matching of a JSON blob. The tokenizer will split on colons/braces, making matches unreliable. A checkpoint stored as `{"thread_id": "abc", "state": {...}}` won't reliably match query `"checkpoint thread_id:abc"`.

**Fix:** Store checkpoints with a dedicated tag/metadata system, or use exact ID lookup instead of recall.

---

## 🟡 Warnings (should fix soon)

### 6. `SynapseVectorStore.similarity_search` returns `effective_strength` as score
This isn't a similarity score — it's a temporal decay value (0.0-1.0). Users expect cosine similarity or relevance scores. Misleading.

### 7. `SynapseMemory` (langchain.py) uses Pydantic v1 style but LangChain v0.2+ uses Pydantic v2
`BaseMemory` in recent LangChain uses Pydantic v2 model config. Fields like `memory_key: str = "history"` may need `model_config` instead of `class Config`.

### 8. `demo_script.py` references `pip install synapse-memory` (old name)
Line 19 still says the old package name. Should be `synapse-ai-memory`.

### 9. `debug.py` is shipped in the PyPI package
`pyproject.toml` includes `debug` in `py-modules`. This is a dev-only script with hardcoded test data — shouldn't be in the distributed package.

### 10. No `__init__.py` at package root
Synapse is distributed as flat modules (synapse.py, storage.py, etc.) not as a proper package. This works but `from synapse_ai_memory import Synapse` won't work — users must use `from synapse import Synapse`. Could confuse people expecting the PyPI name to match the import.

---

## 💡 Suggestions (nice to have)

### 11. Add `py.typed` marker for type checking support

### 12. Integration `__init__.py` has complex import fallback logic
The try/except chain in `integrations/__init__.py` is fragile. Consider lazy imports instead.

### 13. `SynapseMemory.clear()` creates a fresh Synapse without preserving `data_dir`
Same issue as #3 but for the main memory class.

### 14. Daemon (`synapsed.py`) has no authentication
Anyone on the network can connect and read/write/delete all memories. Fine for local dev, but should be documented as a security limitation.

### 15. Consider adding a `CHANGELOG.md` for the v0.1.0 release

---

## Test Results

| Test Suite | Result |
|---|---|
| `test_synapse` | ✅ 33 tests passed |
| `test_entity_graph` | ✅ 5 tests passed |
| `test_episode_graph` | ✅ 7 tests passed |
| `test_extractor` | ⚠️ Requires `requests` package |
| `test_daemon` | ⚠️ May hang on port binding |
| `integrations/test_integrations` | ❌ 0 tests discovered (not unittest.TestCase) |

**Core: 45/45 passing. Integrations: 0 running.**

---

## Verdict

**Ship the core engine — it's solid.** The integrations need the critical fixes above before they're credible as "drop-in replacements." Priority order:

1. Fix integration tests (make them actually run)
2. Fix `extractor.py` requests dependency
3. Fix `clear()` methods to not nuke shared data
4. Fix or remove `SynapseCheckpointer` (it won't work as-is)
5. Fix `demo_script.py` package name
6. Remove `debug.py` from PyPI package
