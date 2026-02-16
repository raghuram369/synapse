"""Synapse AI Memory example: federation sync between two agents (localhost-only server)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse import Synapse
from federation.memory_object import FederatedMemory
import time


def namespace_alice_memories(alice):
    node = alice._federation_node
    alice.share("public")
    alice.share("private")
    for fm in list(node.store.memories.values()):
        ns = "public" if "roadmap" in fm.content.lower() else "private"
        if fm.namespaces == [ns]:
            continue
        node.store.remove_memory(fm.hash)
        node.store.add_memory(
            FederatedMemory(
                content=fm.content, memory_type=fm.memory_type,
                metadata=fm.metadata, created_at=fm.created_at,
                origin_node=fm.origin_node, namespaces=[ns],
                vclock=fm.vclock, supersedes=fm.supersedes
            )
        )


def sync_and_show(name, peer_url, namespaces=None):
    peer = Synapse(":memory:")
    stats = peer.sync(peer_url, namespaces=namespaces)
    print(f"{name} sync: {stats}")
    print([m.content for m in peer.recall("alice", limit=5)])


def main():
    alice = Synapse(":memory:")
    bob = Synapse(":memory:")

    alice.remember("Alice publishes public roadmap for Q3.")
    alice.remember("Alice keeps private runbook credentials.")
    bob.remember("Bob manages deployment checks.")

    server = alice.serve(port=9640, host="127.0.0.1")
    time.sleep(0.1)
    try:
        peer_url = "http://127.0.0.1:9640"
        sync_and_show("Bob full", peer_url)
        namespace_alice_memories(alice)
        sync_and_show("Bob public", peer_url, namespaces={"public"})
        sync_and_show("Bob private", peer_url, namespaces={"private"})
    finally:
        server.stop()


if __name__ == "__main__":
    main()
