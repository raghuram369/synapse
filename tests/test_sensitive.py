import pytest

from egress_guard import SensitiveContentDetector
from synapse import Synapse


@pytest.fixture
def synapse_instance():
    syn = Synapse(":memory:")
    try:
        yield syn
    finally:
        syn.close()


def test_sensitive_default_false(synapse_instance):
    memory = synapse_instance.remember("General note", deduplicate=False)
    assert memory.sensitive is False



def test_remember_with_sensitive_flag(synapse_instance):
    memory = synapse_instance.remember(
        "My child goes to Lincoln Elementary",
        scope="shared",
        sensitive=True,
        deduplicate=False,
    )
    assert memory.sensitive is True



def test_sensitive_hidden_from_shared_scope(synapse_instance):
    visible = synapse_instance.remember("Project update", scope="shared", deduplicate=False)
    hidden = synapse_instance.remember(
        "My child goes to Lincoln Elementary",
        scope="shared",
        sensitive=True,
        deduplicate=False,
    )

    recalled_ids = {m.id for m in synapse_instance.recall(context="", limit=20, scope="shared")}
    assert visible.id in recalled_ids
    assert hidden.id not in recalled_ids



def test_sensitive_hidden_from_public_scope(synapse_instance):
    visible = synapse_instance.remember("Public roadmap note", scope="public", deduplicate=False)
    hidden = synapse_instance.remember(
        "Our home address is 123 Main Street",
        scope="public",
        sensitive=True,
        deduplicate=False,
    )

    recalled_ids = {m.id for m in synapse_instance.recall(context="", limit=20, scope="public")}
    assert visible.id in recalled_ids
    assert hidden.id not in recalled_ids



def test_sensitive_visible_in_private_scope(synapse_instance):
    hidden = synapse_instance.remember(
        "Private health detail",
        scope="shared",
        sensitive=True,
        deduplicate=False,
    )

    recalled_ids = {m.id for m in synapse_instance.recall(context="", limit=20, scope="private")}
    assert hidden.id in recalled_ids



def test_policy_auto_sensitive_tags(synapse_instance):
    synapse_instance.policy("private")
    memory = synapse_instance.remember(
        "Need to gather legal paperwork",
        metadata={"tags": ["legal"]},
        scope="shared",
        deduplicate=False,
    )

    assert memory.sensitive is True



def test_sensitive_persisted_across_reload(tmp_path):
    db_path = tmp_path / "sensitive-persist.synapse"
    syn = Synapse(str(db_path))
    try:
        memory = syn.remember(
            "Medical diagnosis follow-up",
            scope="shared",
            sensitive=True,
            deduplicate=False,
        )
        memory_id = memory.id
        syn.flush()
    finally:
        syn.close()

    reloaded = Synapse(str(db_path))
    try:
        private_visible = {m.id: m for m in reloaded.recall(context="", limit=20, scope="private")}
        shared_visible = {m.id: m for m in reloaded.recall(context="", limit=20, scope="shared")}

        assert memory_id in private_visible
        assert private_visible[memory_id].sensitive is True
        assert memory_id not in shared_visible
    finally:
        reloaded.close()



def test_sensitive_content_detector_health():
    assert SensitiveContentDetector.detect("I was diagnosed with diabetes last year") is True



def test_sensitive_content_detector_children():
    assert SensitiveContentDetector.detect("My kid goes to Lincoln Elementary") is True



def test_sensitive_content_detector_normal_text():
    assert SensitiveContentDetector.detect("We should refactor the API handler tomorrow") is False



def test_mark_sensitive_bulk(synapse_instance):
    shared_memory = synapse_instance.remember("Shared plan", scope="shared", deduplicate=False)
    public_memory = synapse_instance.remember("Public status", scope="public", deduplicate=False)

    updated = synapse_instance.bulk_set_sensitive([shared_memory.id, public_memory.id], sensitive=True)
    assert updated == 2

    shared_recall_ids = {m.id for m in synapse_instance.recall(context="", limit=20, scope="shared")}
    private_recall_ids = {m.id for m in synapse_instance.recall(context="", limit=20, scope="private")}

    assert shared_memory.id not in shared_recall_ids
    assert public_memory.id not in shared_recall_ids
    assert shared_memory.id in private_recall_ids
    assert public_memory.id in private_recall_ids
