"""LAN peer discovery via UDP broadcast (mDNS-lite). Pure stdlib."""

from __future__ import annotations
import json
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional, Set


BROADCAST_PORT = 9471
MAGIC = b"SYNAPSE\x00"
ANNOUNCE_INTERVAL = 30.0  # seconds


class PeerDiscovery:
    """
    Simple UDP broadcast discovery for Synapse nodes on LAN.
    
    Each node periodically broadcasts:
      SYNAPSE\x00 + JSON payload
    
    Payload: {"node_id": "...", "port": 9470, "namespaces": [...]}
    """

    def __init__(
        self,
        node_id: str,
        port: int = 9470,
        broadcast_port: int = BROADCAST_PORT,
        on_peer_found: Callable[[str, Dict], None] | None = None,
    ):
        self.node_id = node_id
        self.port = port
        self.broadcast_port = broadcast_port
        self.on_peer_found = on_peer_found

        # Known peers: url -> {node_id, last_seen, ...}
        self.peers: Dict[str, Dict] = {}
        self._running = False
        self._threads: list = []

    def start(self):
        """Start broadcast sender and listener."""
        self._running = True

        # Listener thread
        t1 = threading.Thread(target=self._listen_loop, daemon=True)
        t1.start()
        self._threads.append(t1)

        # Announcer thread
        t2 = threading.Thread(target=self._announce_loop, daemon=True)
        t2.start()
        self._threads.append(t2)

    def stop(self):
        self._running = False

    def _announce_loop(self):
        """Periodically broadcast our presence."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)

        payload = json.dumps({
            "node_id": self.node_id,
            "port": self.port,
        }).encode("utf-8")
        message = MAGIC + payload

        while self._running:
            try:
                sock.sendto(message, ("<broadcast>", self.broadcast_port))
            except OSError:
                pass
            # Sleep in small increments so stop() is responsive
            for _ in range(int(ANNOUNCE_INTERVAL)):
                if not self._running:
                    break
                time.sleep(1.0)

        sock.close()

    def _listen_loop(self):
        """Listen for peer broadcasts."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        sock.settimeout(2.0)

        try:
            sock.bind(("", self.broadcast_port))
        except OSError:
            return  # Can't bind, skip discovery

        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data.startswith(MAGIC):
                continue

            try:
                payload = json.loads(data[len(MAGIC):])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            peer_node_id = payload.get("node_id", "")
            peer_port = payload.get("port", 9470)

            # Don't discover ourselves
            if peer_node_id == self.node_id:
                continue

            peer_url = f"http://{addr[0]}:{peer_port}"
            peer_info = {
                "node_id": peer_node_id,
                "url": peer_url,
                "last_seen": time.time(),
                "ip": addr[0],
                "port": peer_port,
            }

            is_new = peer_url not in self.peers
            self.peers[peer_url] = peer_info

            if is_new and self.on_peer_found:
                self.on_peer_found(peer_url, peer_info)

        sock.close()
