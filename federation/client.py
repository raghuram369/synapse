"""HTTP client for federated sync — pure stdlib (urllib)."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Set
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from federation.store import FederatedStore
from federation.sync import SyncEngine
from federation.memory_object import FederatedMemory


class SyncClient:
    """Client for syncing with a remote Synapse node over HTTP."""

    def __init__(self, store: FederatedStore, timeout: float = 30.0):
        self.store = store
        self.engine = SyncEngine(store)
        self.timeout = timeout
        self._peer_tokens: Dict[str, str] = {}  # peer_url -> token

    def set_peer_token(self, peer_url: str, token: str):
        """Configure auth token for a specific peer."""
        self._peer_tokens[peer_url] = token

    def _get_token_for_url(self, url: str) -> Optional[str]:
        """Find the configured token for a URL by matching its base."""
        for peer_url, token in self._peer_tokens.items():
            if url.startswith(peer_url):
                return token
        return None

    def _request(self, url: str, data: Any = None, method: str = "GET") -> Dict[str, Any]:
        """Make an HTTP request and return parsed JSON."""
        headers = {"Content-Type": "application/json"}
        token = self._get_token_for_url(url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body = None
        if data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            method = "POST"

        req = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise ConnectionError(f"HTTP {e.code}: {error_body}") from e
        except URLError as e:
            raise ConnectionError(f"Connection failed: {e.reason}") from e

    # ── High-level operations ─────────────────────────────────

    def status(self, peer_url: str) -> Dict[str, Any]:
        """Get status of a remote peer."""
        return self._request(f"{peer_url}/v1/status")

    def pull(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """
        Pull memories from a remote peer.
        Returns {"pulled": N}.
        """
        ns_list = sorted(namespaces) if namespaces else []

        # Phase 1: Get remote tree state
        remote_tree = self._request(f"{peer_url}/v1/tree")

        # Phase 2: Compute what we need
        delta = self.engine.compute_missing(remote_tree, namespaces)
        if delta.get("status") == "in_sync":
            return {"pulled": 0}

        # Phase 2b: Get hashes in differing buckets
        diff_buckets = delta["differing_buckets"]
        resp = self._request(f"{peer_url}/v1/sync/hashes", {
            "bucket_ids": diff_buckets,
            "namespaces": ns_list,
        })
        remote_hashes = set(resp["hashes"])

        # Phase 2c: Which are we missing?
        missing = remote_hashes - self.store.all_hashes()
        if not missing:
            return {"pulled": 0}

        # Phase 3: Fetch full objects
        resp = self._request(f"{peer_url}/v1/sync/pull", {
            "hashes": sorted(missing),
        })
        pulled = self.engine.import_memories(resp.get("memories", []))
        self.engine.import_edges(resp.get("edges", []))

        return {"pulled": pulled}

    def push(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """
        Push memories to a remote peer.
        Returns {"pushed": N}.
        """
        ns_list = sorted(namespaces) if namespaces else []

        # Phase 1: Get our tree state
        local_tree = self.engine.get_tree_state(namespaces)

        # Send to remote, get delta info
        delta = self._request(f"{peer_url}/v1/sync/tree", {
            "tree_state": local_tree,
            "namespaces": ns_list,
        })

        if delta.get("status") == "in_sync":
            return {"pushed": 0}

        # Phase 2: Get our hashes in differing buckets
        diff_buckets = delta["differing_buckets"]
        local_hashes = self.engine.get_items_in_buckets(diff_buckets, namespaces)

        # Phase 3: Push full objects
        memories = self.engine.export_memories(local_hashes)
        edges = self.engine.export_edges(self.store.all_hashes())

        resp = self._request(f"{peer_url}/v1/sync/push", {
            "memories": memories,
            "edges": edges,
        })

        return {"pushed": resp.get("imported_memories", 0)}

    def sync(self, peer_url: str, namespaces: Set[str] | None = None) -> Dict[str, int]:
        """Bidirectional sync. Returns {"pushed": N, "pulled": M}."""
        pulled = self.pull(peer_url, namespaces)
        pushed = self.push(peer_url, namespaces)
        return {"pulled": pulled["pulled"], "pushed": pushed["pushed"]}

    def announce(self, peer_url: str):
        """Announce ourselves to a peer."""
        self._request(f"{peer_url}/v1/peers/announce", {
            "url": f"http://localhost",  # will be overridden by caller
            "info": {
                "node_id": self.store.node_id,
                "memory_count": len(self.store.memories),
            },
        })

    def list_peers(self, peer_url: str) -> Dict[str, Any]:
        """Get peer list from a remote node."""
        return self._request(f"{peer_url}/v1/peers")
