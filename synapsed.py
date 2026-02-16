#!/usr/bin/env python3
"""Multi-threaded TCP daemon that serves Synapse over JSON-over-TCP."""

from __future__ import annotations

import argparse
import os
import json
import logging
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from synapse import Synapse


class SynapseServer:
    """Multi-threaded TCP server for Synapse operations."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7654, 
                 data_dir: str = "./synapse_data", extract_default: bool = False):
        self.host = host
        self.port = port  
        self.data_dir = Path(data_dir)
        self.extract_default = extract_default
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Synapse instance (shared across all connections)
        synapse_db_path = str(self.data_dir / "synapse.db")
        self.synapse = Synapse(synapse_db_path)
        
        # Thread safety
        self.synapse_lock = threading.Lock()
        
        # Server state
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_count = 0
        
    def start(self):
        """Start the daemon server."""
        print(f"🧠 Synapse Daemon v2.0")
        print(f"📍 Listening on {self.host}:{self.port}")
        print(f"💾 Data directory: {self.data_dir.absolute()}")
        print(f"🔬 Default extraction: {'ON' if self.extract_default else 'OFF'}")
        print("-" * 50)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)  # Allow up to 10 queued connections
            self.running = True
            
            print(f"✅ Server started successfully")
            
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    self.client_count += 1
                    client_id = self.client_count
                    
                    logger.info("Client #%d connected from %s", client_id, client_addr)
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_id),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:  # Only log if we're still supposed to be running
                        logger.error("Socket error: %s", e)
                        
        except Exception as e:
            logger.error("Server error: %s", e)
            
        finally:
            self._cleanup()
    
    def _handle_client(self, client_socket: socket.socket, client_id: int):
        """Handle a single client connection."""
        try:
            # Setup buffered I/O
            client_file = client_socket.makefile('rw', encoding='utf-8')
            
            while self.running:
                try:
                    # Read JSON request (newline-delimited)
                    line = client_file.readline()
                    if not line:  # Client disconnected
                        break
                        
                    request = json.loads(line.strip())
                    
                    # Process request
                    response = self._process_request(request)
                    
                    # Send response
                    client_file.write(json.dumps(response) + '\n')
                    client_file.flush()
                    
                except json.JSONDecodeError as e:
                    error_response = {"ok": False, "error": f"Invalid JSON: {e}"}
                    client_file.write(json.dumps(error_response) + '\n')
                    client_file.flush()
                    
                except Exception as e:
                    error_response = {"ok": False, "error": f"Request processing error: {e}"}
                    client_file.write(json.dumps(error_response) + '\n')
                    client_file.flush()
                    logger.warning("Client #%d error: %s", client_id, e)
                    
        except Exception as e:
            logger.error("Client #%d connection error: %s", client_id, e)
            
        finally:
            try:
                client_socket.close()
                logger.info("Client #%d disconnected", client_id)
            except OSError:
                pass
    
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single client request."""
        cmd = request.get('cmd')
        
        if not cmd:
            return {"ok": False, "error": "Missing 'cmd' field"}
        
        try:
            if cmd == "ping":
                return {"ok": True, "data": "pong"}
                
            elif cmd == "shutdown":
                # Initiate graceful shutdown
                threading.Thread(target=self._delayed_shutdown, daemon=True).start()
                return {"ok": True, "data": "Shutdown initiated"}
                
            elif cmd == "remember":
                return self._cmd_remember(request)
                
            elif cmd == "recall":
                return self._cmd_recall(request)
                
            elif cmd == "forget":
                return self._cmd_forget(request)
                
            elif cmd == "link":
                return self._cmd_link(request)
                
            elif cmd == "concepts":
                return self._cmd_concepts(request)

            elif cmd == "hot_concepts":
                return self._cmd_hot_concepts(request)

            elif cmd == "prune":
                return self._cmd_prune(request)
                
            elif cmd == "stats":
                return self._cmd_stats(request)

            elif cmd == "policy":
                return self._cmd_policy(request)
                
            else:
                return {"ok": False, "error": f"Unknown command: {cmd}"}
                
        except Exception as e:
            print(f"⚠️  Command '{cmd}' error: {e}")
            traceback.print_exc()
            return {"ok": False, "error": str(e)}
    
    def _cmd_remember(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle remember command."""
        content = request.get('content', '')
        memory_type = request.get('memory_type', 'fact')
        extract = request.get('extract', self.extract_default)
        
        if not content:
            return {"ok": False, "error": "Missing 'content' field"}
        
        with self.synapse_lock:
            memory = self.synapse.remember(content, memory_type=memory_type, extract=extract)
            
        return {
            "ok": True, 
            "data": {
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "strength": memory.strength,
                "created_at": memory.created_at
            }
        }
    
    def _cmd_recall(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recall command."""
        context = request.get('context', '')
        limit = request.get('limit', 10)
        explain = bool(request.get('explain', False))

        with self.synapse_lock:
            memories = self.synapse.recall(context, limit=limit, explain=explain)

        data = []
        for m in memories:
            item = {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "strength": m.strength,
                "effective_strength": m.effective_strength,
                "created_at": m.created_at,
                "metadata": m.metadata,
            }
            if explain and m.score_breakdown is not None:
                bd = m.score_breakdown
                item["score_breakdown"] = {
                    "bm25_score": bd.bm25_score,
                    "concept_score": bd.concept_score,
                    "temporal_score": bd.temporal_score,
                    "episode_score": bd.episode_score,
                    "concept_activation_score": bd.concept_activation_score,
                    "embedding_score": bd.embedding_score,
                    "match_sources": bd.match_sources,
                }
            data.append(item)

        return {"ok": True, "data": data}
    
    def _cmd_forget(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forget command."""
        memory_id = request.get('id')
        
        if memory_id is None:
            return {"ok": False, "error": "Missing 'id' field"}
            
        with self.synapse_lock:
            success = self.synapse.forget(memory_id)
            
        return {"ok": True, "data": {"deleted": success}}
    
    def _cmd_link(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle link command."""
        source = request.get('source')
        target = request.get('target')
        edge_type = request.get('edge_type', 'related')
        weight = request.get('weight', 1.0)
        
        if source is None or target is None:
            return {"ok": False, "error": "Missing 'source' or 'target' field"}
        
        try:
            with self.synapse_lock:
                self.synapse.link(source, target, edge_type, weight)
            return {"ok": True, "data": "Link created"}
            
        except ValueError as e:
            return {"ok": False, "error": str(e)}
    
    def _cmd_concepts(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle concepts command."""
        with self.synapse_lock:
            concepts = self.synapse.concepts()
        return {"ok": True, "data": concepts}

    def _cmd_hot_concepts(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hot_concepts command."""
        k = int(request.get('k', 10))
        with self.synapse_lock:
            hot = self.synapse.hot_concepts(k=k)
        return {"ok": True, "data": [[name, strength] for name, strength in hot]}

    def _cmd_prune(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prune command."""
        min_strength = float(request.get('min_strength', 0.1))
        min_access = int(request.get('min_access', 0))
        max_age_days = float(request.get('max_age_days', 90))
        dry_run = bool(request.get('dry_run', True))
        with self.synapse_lock:
            pruned = self.synapse.prune(
                min_strength=min_strength,
                min_access=min_access,
                max_age_days=max_age_days,
                dry_run=dry_run,
            )
        return {"ok": True, "data": pruned}

    def _cmd_stats(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stats command."""
        with self.synapse_lock:
            total_memories = len([m for m in self.synapse.store.memories.values() 
                                 if not m.get('consolidated', False)])
            total_concepts = len(self.synapse.concept_graph.concepts)
            total_edges = len(self.synapse.store.edges)
            
        return {
            "ok": True,
            "data": {
                "total_memories": total_memories,
                "total_concepts": total_concepts,
                "total_edges": total_edges,
                "client_count": self.client_count,
                "data_directory": str(self.data_dir.absolute())
            }
        }

    def _cmd_policy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle policy command."""
        action = str(request.get("action", "show")).strip().lower()
        if action in {"show", ""}:
            return {"ok": True, "data": self.synapse.policy()}
        if action == "list":
            return {"ok": True, "data": self.synapse.list_presets()}
        if action == "set":
            preset = self._coerce_text(request.get("preset"))
            if not preset:
                return {"ok": False, "error": "Missing 'preset' field"}
            return {"ok": True, "data": self.synapse.policy(preset)}
        return {"ok": False, "error": f"Unsupported policy action: {action}"}

    @staticmethod
    def _coerce_text(value: Any) -> str:
        return str(value).strip() if value is not None else ""
    
    def _delayed_shutdown(self):
        """Delayed shutdown to allow response to be sent."""
        time.sleep(0.1)  # Give response time to be sent
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")

        if self.synapse:
            self.synapse.close()

        if self.server_socket:
            self.server_socket.close()

        print("✅ Shutdown complete")


def _resolve_data_dir(args_db: Optional[str], args_data_dir: Optional[str]) -> str:
    source = args_db if args_db is not None else args_data_dir or "./synapse_data"
    expanded = os.path.expanduser(source)
    path = Path(expanded)
    if path.suffix:
        return str(path.parent if str(path.parent) else Path("."))
    return str(path)


def _write_pid_file(path: Optional[str], pid: int) -> None:
    if not path:
        return
    expanded = os.path.expanduser(path)
    os.makedirs(os.path.dirname(expanded), exist_ok=True)
    with open(expanded, "w", encoding="utf-8") as fp:
        fp.write(str(int(pid)))


def _daemonize(path: Optional[str]) -> bool:
    """Daemonize process when running on POSIX. Returns True for parent process."""
    if os.name != "posix" or not hasattr(os, "fork"):
        return False

    first_pid = os.fork()
    if first_pid > 0:
        return True

    os.setsid()
    second_pid = os.fork()
    if second_pid > 0:
        os._exit(0)

    os.chdir("/")
    os.umask(0)
    with open(os.devnull, "rb", buffering=0) as null_in:
        os.dup2(null_in.fileno(), 0)
    with open(os.devnull, "ab", buffering=0) as null_out:
        os.dup2(null_out.fileno(), 1)
        os.dup2(null_out.fileno(), 2)

    _write_pid_file(path, os.getpid())
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Synapse Daemon Server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7654, help='Port to bind to')
    parser.add_argument('--data-dir', default=None, help='Data directory')
    parser.add_argument('--db', default=None, help='Database path (or data directory)')
    parser.add_argument('--mode', choices=['appliance', 'full'], default='full',
                        help='Synapse startup mode')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon process')
    parser.add_argument('--pid-file', default=None, help='Write PID to this file')
    parser.add_argument('--extract', action='store_true', 
                       help='Enable fact extraction by default')
    
    args = parser.parse_args()

    if args.daemon and _daemonize(args.pid_file):
        return

    data_dir = _resolve_data_dir(args.db, args.data_dir)
    if args.mode == 'appliance':
        logger.info("Starting Synapse daemon in appliance mode")
    else:
        logger.info("Starting Synapse daemon in full mode")
    
    server = SynapseServer(
        host=args.host,
        port=args.port, 
        data_dir=data_dir,
        extract_default=args.extract
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
