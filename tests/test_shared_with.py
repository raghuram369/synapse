import json

import pytest

from synapse import Synapse


@pytest.fixture
def synapse_instance():
    syn = Synapse(":memory:")
    try:
        yield syn
    finally:
        syn.close()


def _recall_ids(syn: Synapse, scope: str, caller_groups=None) -> set[int]:
    return {
        memory.id
        for memory in syn.recall(
            context="",
            limit=100,
            scope=scope,
            caller_groups=caller_groups,
        )
    }


def test_remember_with_shared_with(synapse_instance):
    memory = synapse_instance.remember(
        "shared-with-groups",
        scope="shared",
        shared_with=["family", "team:ops", "user:alice"],
        deduplicate=False,
    )

    assert memory.shared_with == ["family", "team:ops", "user:alice"]
    stored = synapse_instance.store.memories[memory.id].get("shared_with")
    assert json.loads(stored) == ["family", "team:ops", "user:alice"]


def test_shared_with_none_visible_to_all(synapse_instance):
    memory = synapse_instance.remember(
        "shared-with-everyone",
        scope="shared",
        deduplicate=False,
    )

    alice_ids = _recall_ids(synapse_instance, scope="shared", caller_groups=["user:alice"])
    ops_ids = _recall_ids(synapse_instance, scope="shared", caller_groups=["team:ops"])

    assert memory.id in alice_ids
    assert memory.id in ops_ids


def test_shared_with_groups_filter(synapse_instance):
    memory = synapse_instance.remember(
        "shared-team-ops",
        scope="shared",
        shared_with=["team:ops", "family"],
        deduplicate=False,
    )

    ids = _recall_ids(synapse_instance, scope="shared", caller_groups=["team:ops"])
    assert memory.id in ids


def test_shared_with_no_match_hidden(synapse_instance):
    memory = synapse_instance.remember(
        "shared-no-match",
        scope="shared",
        shared_with=["team:ops"],
        deduplicate=False,
    )

    ids = _recall_ids(synapse_instance, scope="shared", caller_groups=["family"])
    assert memory.id not in ids


def test_shared_with_private_ignores_groups(synapse_instance):
    memory = synapse_instance.remember(
        "private-ignores-groups",
        scope="private",
        shared_with=["team:ops"],
        deduplicate=False,
    )

    ids = _recall_ids(synapse_instance, scope="private", caller_groups=["family"])
    assert memory.id in ids


def test_shared_with_persisted_across_reload(tmp_path):
    db_path = tmp_path / "shared-with-persist.synapse"
    syn = Synapse(str(db_path))
    try:
        memory = syn.remember(
            "persisted-shared-with",
            scope="shared",
            shared_with=["team:ops", "user:alice"],
            deduplicate=False,
        )
        memory_id = memory.id
        syn.flush()
    finally:
        syn.close()

    reloaded = Synapse(str(db_path))
    try:
        recalled = {m.id: m for m in reloaded.recall(context="", scope="private", limit=100)}
        assert memory_id in recalled
        assert recalled[memory_id].shared_with == ["team:ops", "user:alice"]
    finally:
        reloaded.close()


def test_compile_context_with_caller_groups(synapse_instance):
    public = synapse_instance.remember("ctx-public", scope="public", deduplicate=False)
    family = synapse_instance.remember(
        "ctx-family",
        scope="shared",
        shared_with=["family"],
        deduplicate=False,
    )
    synapse_instance.remember(
        "ctx-ops",
        scope="shared",
        shared_with=["team:ops"],
        deduplicate=False,
    )

    pack = synapse_instance.compile_context(
        query="",
        budget=5000,
        policy="broad",
        scope="shared",
        caller_groups=["family"],
    )

    ids = {item["id"] for item in pack.memories}
    assert public.id in ids
    assert family.id in ids
    assert all("ctx-ops" not in item.get("content", "") for item in pack.memories)
