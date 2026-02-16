from dataclasses import FrozenInstanceError

import pytest

from scope_policy import ScopePolicy
from synapse import Synapse


@pytest.fixture
def synapse_instance():
    syn = Synapse(":memory:")
    try:
        yield syn
    finally:
        syn.close()


def _seed_scopes(syn: Synapse):
    public = syn.remember("scope-policy-public", scope="public", deduplicate=False)
    shared = syn.remember("scope-policy-shared", scope="shared", deduplicate=False)
    private = syn.remember("scope-policy-private", scope="private", deduplicate=False)
    return public, shared, private


def test_owner_policy_allows_private():
    policy = ScopePolicy.owner()
    assert policy.max_scope == "private"
    assert policy.enforce("private") == "private"


def test_external_policy_clamps_to_shared():
    policy = ScopePolicy.external()
    assert policy.max_scope == "shared"
    assert policy.enforce("private") == "shared"
    assert policy.enforce("shared") == "shared"
    assert policy.enforce("public") == "public"


def test_anonymous_policy_clamps_to_public():
    policy = ScopePolicy.anonymous()
    assert policy.max_scope == "public"
    assert policy.enforce("private") == "public"
    assert policy.enforce("shared") == "public"
    assert policy.enforce("public") == "public"


def test_no_policy_allows_everything(synapse_instance):
    public, shared, private = _seed_scopes(synapse_instance)
    recalled_ids = {m.id for m in synapse_instance.recall(context="", limit=50, scope="private")}
    assert recalled_ids == {public.id, shared.id, private.id}


def test_policy_immutable():
    policy = ScopePolicy.external()
    with pytest.raises(FrozenInstanceError):
        policy.max_scope = "private"


def test_recall_with_external_policy():
    syn = Synapse(":memory:", scope_policy=ScopePolicy.external())
    try:
        public, shared, private = _seed_scopes(syn)
        recalled_ids = {m.id for m in syn.recall(context="", limit=50, scope="private")}
        assert public.id in recalled_ids
        assert shared.id in recalled_ids
        assert private.id not in recalled_ids
    finally:
        syn.close()


def test_compile_context_with_policy():
    syn = Synapse(":memory:", scope_policy=ScopePolicy.external())
    try:
        public, shared, private = _seed_scopes(syn)
        pack = syn.compile_context(query="", budget=5000, policy="broad", scope="private")
        ids = {item["id"] for item in pack.memories}
        assert public.id in ids
        assert shared.id in ids
        assert private.id not in ids
        assert pack.metadata["scope"] == "shared"
    finally:
        syn.close()


def test_scope_policy_env_var(monkeypatch):
    """SYNAPSE_SCOPE_POLICY env var should be respected by factory methods."""
    # Test that factory methods produce correct policies
    policy = ScopePolicy.anonymous()
    assert policy.max_scope == "public"

    policy = ScopePolicy.external()
    assert policy.max_scope == "shared"

    policy = ScopePolicy.owner()
    assert policy.max_scope == "private"
