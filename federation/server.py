"""HTTP server for federated sync — pure stdlib (http.server + json)."""

from __future__ import annotations
import hmac
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse, parse_qs

from federation.store import FederatedStore
from federation.sync import SyncEngine


MAX_PAYLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


class SynapseHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the sync protocol."""

    # Silence request logging
    def log_message(self, format, *args):
        pass

    @property
    def engine(self) -> SyncEngine:
        return self.server._sync_engine  # type: ignore

    @property
    def store(self) -> FederatedStore:
        return self.server._store  # type: ignore

    @property
    def auth_token(self) -> Optional[str]:
        return getattr(self.server, '_auth_token', None)

    def _check_auth(self) -> bool:
        """Check bearer token auth. Returns True if authorized."""
        token = self.auth_token
        if token is None:
            return True  # open mode
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if hmac.compare_digest(auth_header, expected):
            return True
        self._send_json({"error": "unauthorized"}, 401)
        return False

    def _read_json(self) -> Dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        if length > MAX_PAYLOAD_SIZE:
            self._send_json({"error": f"payload too large ({length} bytes, max {MAX_PAYLOAD_SIZE})"}, 413)
            return None
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json({"error": f"invalid JSON: {e}"}, 400)
            return None

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/x-synapse")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_push_synapse(self):
        """Handle binary .synapse blob push."""
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_PAYLOAD_SIZE:
            self._send_json({"error": "payload too large"}, 413)
            return
        data = self.rfile.read(length)
        try:
            result = self.engine.import_synapse_binary(data)
            self._send_json({
                "imported_memories": result["memories"],
                "imported_edges": result["edges"],
            })
        except (ValueError, KeyError) as e:
            self._send_json({"error": f"invalid synapse data: {e}"}, 400)

    # ── Routes ────────────────────────────────────────────────

    def do_GET(self):
        if not self._check_auth():
            return
        path = urlparse(self.path).path

        if path == "/v1/status":
            self._send_json({
                "node_id": self.store.node_id,
                "memory_count": len(self.store.memories),
                "edge_count": len(self.store.edges),
                "root_hash": self.store.memory_tree.root,
                "shared_namespaces": sorted(self.store.shared_namespaces),
            })

        elif path == "/v1/tree":
            params = parse_qs(urlparse(self.path).query)
            ns = set(params.get("namespace", []))
            self._send_json(self.engine.get_tree_state(ns or None))

        elif path == "/v1/peers":
            self._send_json({"peers": self.store.peers})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._check_auth():
            return
        path = urlparse(self.path).path

        # Special case: binary push reads raw body, not JSON
        if path == "/v1/sync/push-synapse":
            return self._handle_push_synapse()

        body = self._read_json()
        if body is None:
            return  # error already sent

        if path == "/v1/sync/tree":
            # Phase 1: Receive remote tree state, return delta info
            ns = set(body.get("namespaces", []))
            remote_state = body.get("tree_state", {})
            delta = self.engine.compute_missing(remote_state, ns or None)
            self._send_json(delta)

        elif path == "/v1/sync/hashes":
            # Phase 2: Return hashes in specified buckets
            bucket_ids = body.get("bucket_ids", [])
            ns = set(body.get("namespaces", []))
            hashes = self.engine.get_items_in_buckets(bucket_ids, ns or None)
            self._send_json({"hashes": sorted(hashes)})

        elif path == "/v1/sync/pull":
            # Phase 3: Return full memory objects for requested hashes
            requested = set(body.get("hashes", []))
            # Try binary .synapse format first
            synapse_data = self.engine.export_synapse_binary(requested)
            if synapse_data is not None and body.get("accept_synapse", False):
                self._send_binary(synapse_data)
            else:
                memories = self.engine.export_memories(requested)
                edges = self.engine.export_edges(
                    self.store.all_hashes()
                )
                self._send_json({
                    "memories": memories,
                    "edges": edges,
                })

        elif path == "/v1/sync/push":
            # Receive memories and edges from remote
            try:
                mem_count = self.engine.import_memories(body.get("memories", []))
                edge_count = self.engine.import_edges(body.get("edges", []))
                self._send_json({
                    "imported_memories": mem_count,
                    "imported_edges": edge_count,
                })
            except (ValueError, KeyError) as e:
                self._send_json({"error": f"invalid data: {e}"}, 400)

        elif path == "/v1/sync/push-synapse":
            self._handle_push_synapse()

        elif path == "/v1/peers/announce":
            # Peer announcement
            peer_url = body.get("url", "")
            peer_info = body.get("info", {})
            if peer_url:
                self.store.peers[peer_url] = peer_info
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": "missing url"}, 400)

        else:
            self._send_json({"error": "not found"}, 404)


class SynapseServer:
    """Threaded HTTP server for a Synapse federation node."""

    def __init__(self, store: FederatedStore, host: str = "0.0.0.0", port: int = 9470,
                 auth_token: Optional[str] = None):
        self.store = store
        self.engine = SyncEngine(store)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the server in a background thread."""
        self._server = HTTPServer((self.host, self.port), SynapseHandler)
        self._server._store = self.store  # type: ignore
        self._server._sync_engine = self.engine  # type: ignore
        self._server._auth_token = self.auth_token  # type: ignore
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
