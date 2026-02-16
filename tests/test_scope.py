import json

import pytest

from synapse import Synapse, SynapseValidationError


@pytest.fixture
def synapse_instance():
    syn = Synapse(":memory:")
    try:
        yield syn
    finally:
        syn.close()


def _remember(syn: Synapse, content: str, scope: str | None = None):
    kwargs = {"deduplicate": False}
    if scope is not None:
        kwargs["scope"] = scope
    return syn.remember(content, **kwargs)


def _memory_scope(memory) -> str | None:
    if hasattr(memory, "scope"):
        return getattr(memory, "scope")
    if isinstance(getattr(memory, "metadata", None), dict):
        return memory.metadata.get("scope")
    return None


def _recall_scope(syn: Synapse, scope: str | None):
    kwargs = {"context": "", "limit": 50}
    if scope is not None:
        kwargs["scope"] = scope
    return syn.recall(**kwargs)


def _compile_context_scope(syn: Synapse, scope: str | None):
    kwargs = {"query": "", "budget": 5000, "policy": "broad"}
    if scope is not None:
        kwargs["scope"] = scope
    return syn.compile_context(**kwargs)


def _bulk_set_scope(syn: Synapse, memory_ids: list[int], scope: str):
    if hasattr(syn, "bulk_set_scope"):
        method = syn.bulk_set_scope
        for kwargs in (
            {"memory_ids": memory_ids, "scope": scope},
            {"ids": memory_ids, "scope": scope},
        ):
            try:
                return method(**kwargs)
            except TypeError:
                continue
    if hasattr(syn, "set_scope"):
        method = syn.set_scope
        for kwargs in (
            {"memory_ids": memory_ids, "scope": scope},
            {"ids": memory_ids, "scope": scope},
            {"scope": scope, "memory_ids": memory_ids},
        ):
            try:
                return method(**kwargs)
            except TypeError:
                continue
    raise AssertionError("Synapse is missing a bulk scope update API")


def test_default_scope_is_private(synapse_instance):
    memory = _remember(synapse_instance, "default-scope-memory")
    assert _memory_scope(memory) == "private"


def test_remember_with_scope(synapse_instance):
    public = _remember(synapse_instance, "public-memory", scope="public")
    shared = _remember(synapse_instance, "shared-memory", scope="shared")
    private = _remember(synapse_instance, "private-memory", scope="private")

    assert _memory_scope(public) == "public"
    assert _memory_scope(shared) == "shared"
    assert _memory_scope(private) == "private"


def test_invalid_scope_raises(synapse_instance):
    with pytest.raises(SynapseValidationError):
        _remember(synapse_instance, "bad-scope-memory", scope="team-only")


def test_recall_scope_public(synapse_instance):
    public = _remember(synapse_instance, "public-only", scope="public")
    _remember(synapse_instance, "shared-hidden", scope="shared")
    _remember(synapse_instance, "private-hidden", scope="private")

    recalled_ids = {m.id for m in _recall_scope(synapse_instance, "public")}
    assert recalled_ids == {public.id}


def test_recall_scope_shared(synapse_instance):
    public = _remember(synapse_instance, "shared-sees-public", scope="public")
    shared = _remember(synapse_instance, "shared-sees-shared", scope="shared")
    private = _remember(synapse_instance, "shared-hides-private", scope="private")

    recalled_ids = {m.id for m in _recall_scope(synapse_instance, "shared")}
    assert public.id in recalled_ids
    assert shared.id in recalled_ids
    assert private.id not in recalled_ids


def test_recall_scope_private(synapse_instance):
    public = _remember(synapse_instance, "private-sees-public", scope="public")
    shared = _remember(synapse_instance, "private-sees-shared", scope="shared")
    private = _remember(synapse_instance, "private-sees-private", scope="private")

    recalled_ids = {m.id for m in _recall_scope(synapse_instance, "private")}
    assert recalled_ids == {public.id, shared.id, private.id}


def test_recall_no_scope_returns_all(synapse_instance):
    public = _remember(synapse_instance, "no-scope-public", scope="public")
    shared = _remember(synapse_instance, "no-scope-shared", scope="shared")
    private = _remember(synapse_instance, "no-scope-private", scope="private")

    recalled_ids = {m.id for m in _recall_scope(synapse_instance, None)}
    assert recalled_ids == {public.id, shared.id, private.id}


def test_compile_context_respects_scope(synapse_instance):
    public = _remember(synapse_instance, "context-public", scope="public")
    shared = _remember(synapse_instance, "context-shared", scope="shared")
    private = _remember(synapse_instance, "context-private", scope="private")

    pack = _compile_context_scope(synapse_instance, "shared")
    ids = {item["id"] for item in pack.memories}

    assert public.id in ids
    assert shared.id in ids
    assert private.id not in ids


def test_scope_persisted_across_reload(tmp_path):
    db_path = tmp_path / "scope-persist.synapse"
    syn = Synapse(str(db_path))
    try:
        memory = _remember(syn, "persisted-shared", scope="shared")
        memory_id = memory.id
        syn.flush()
    finally:
        syn.close()

    reloaded = Synapse(str(db_path))
    try:
        recalled = {m.id: m for m in _recall_scope(reloaded, "private")}
        assert memory_id in recalled
        assert _memory_scope(recalled[memory_id]) == "shared"
    finally:
        reloaded.close()


def test_scope_in_memory_metadata(synapse_instance):
    memory = _remember(synapse_instance, "metadata-scope", scope="public")
    assert _memory_scope(memory) == "public"


def test_bulk_set_scope(synapse_instance):
    m1 = _remember(synapse_instance, "bulk-a", scope="private")
    m2 = _remember(synapse_instance, "bulk-b", scope="private")
    _remember(synapse_instance, "bulk-c", scope="public")

    _bulk_set_scope(synapse_instance, [m1.id, m2.id], "shared")

    recalled = {m.id: m for m in _recall_scope(synapse_instance, "private")}
    assert _memory_scope(recalled[m1.id]) == "shared"
    assert _memory_scope(recalled[m2.id]) == "shared"


def test_mixed_scope_recall(synapse_instance):
    public_a = _remember(synapse_instance, "mixed-public-a", scope="public")
    public_b = _remember(synapse_instance, "mixed-public-b", scope="public")
    shared_a = _remember(synapse_instance, "mixed-shared-a", scope="shared")
    shared_b = _remember(synapse_instance, "mixed-shared-b", scope="shared")
    private_a = _remember(synapse_instance, "mixed-private-a", scope="private")
    private_b = _remember(synapse_instance, "mixed-private-b", scope="private")

    public_ids = {m.id for m in _recall_scope(synapse_instance, "public")}
    shared_ids = {m.id for m in _recall_scope(synapse_instance, "shared")}
    private_ids = {m.id for m in _recall_scope(synapse_instance, "private")}

    assert public_ids == {public_a.id, public_b.id}
    assert shared_ids == {public_a.id, public_b.id, shared_a.id, shared_b.id}
    assert private_ids == {
        public_a.id,
        public_b.id,
        shared_a.id,
        shared_b.id,
        private_a.id,
        private_b.id,
    }


def test_recall_temporal_all_respects_scope_visibility(synapse_instance):
    older = _remember(synapse_instance, "roadmap phase is alpha", scope="public")
    newer = _remember(synapse_instance, "roadmap phase is beta", scope="private")

    older_meta = dict(older.metadata or {})
    older_meta["superseded_by"] = newer.id
    newer_meta = dict(newer.metadata or {})
    newer_meta["supersedes"] = older.id
    synapse_instance.store.update_memory(older.id, {"metadata": json.dumps(older_meta)})
    synapse_instance.store.update_memory(newer.id, {"metadata": json.dumps(newer_meta)})

    recalled = synapse_instance.recall(
        context="roadmap phase",
        temporal="all",
        scope="public",
        limit=10,
    )
    recalled_ids = {memory.id for memory in recalled}
    assert older.id in recalled_ids
    assert newer.id not in recalled_ids


def test_recall_disputes_do_not_leak_hidden_scope_content(synapse_instance):
    _remember(synapse_instance, "the launch window is open", scope="public")
    _remember(synapse_instance, "the launch window is not open", scope="private")

    recalled = synapse_instance.recall(
        context="launch window",
        limit=5,
        scope="public",
        show_disputes=True,
    )

    for memory in recalled:
        assert all(dispute.get("text", "") != "the launch window is not open" for dispute in memory.disputes)
