#!/usr/bin/env python3
"""TCP client for connecting to the Synapse daemon (*synapsed*)."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional

from exceptions import SynapseConnectionError, SynapseError


class SynapseRequestError(SynapseError):
    """Raised when the daemon returns an error response."""


class SynapseClient:
    """Client for connecting to Synapse daemon."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7654, 
                 auto_reconnect: bool = True, timeout: float = 30.0):
        """Initialize Synapse client.
        
        Args:
            host: Server hostname
            port: Server port
            auto_reconnect: Whether to automatically reconnect on connection loss
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect
        self.timeout = timeout
        
        self._socket: Optional[socket.socket] = None
        self._file: Optional[Any] = None
        self._connected = False
    
    def connect(self):
        """Connect to the Synapse daemon."""
        if self._connected:
            return
            
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self._file = self._socket.makefile('rw', encoding='utf-8')
            self._connected = True
            
        except socket.error as e:
            self._cleanup()
            raise SynapseConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
    
    def disconnect(self):
        """Disconnect from the server."""
        self._cleanup()
    
    def _cleanup(self):
        """Clean up connection resources."""
        self._connected = False
        
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and return the response."""
        max_retries = 2 if self.auto_reconnect else 1
        
        for attempt in range(max_retries):
            try:
                if not self._connected:
                    self.connect()
                
                # Send request
                request_line = json.dumps(request) + '\n'
                self._file.write(request_line)
                self._file.flush()
                
                # Read response
                response_line = self._file.readline()
                if not response_line:
                    raise SynapseConnectionError("Server closed connection")
                
                response = json.loads(response_line.strip())
                
                # Handle response
                if not response.get('ok', False):
                    error_msg = response.get('error', 'Unknown error')
                    raise SynapseRequestError(error_msg)
                
                return response
                
            except (socket.error, OSError, json.JSONDecodeError) as e:
                self._cleanup()
                
                if attempt == max_retries - 1:  # Last attempt
                    raise SynapseConnectionError(f"Request failed: {e}")
                
                # Wait before retry
                time.sleep(0.1)
        
        raise SynapseConnectionError("Max retries exceeded")
    
    def remember(self, content: str, memory_type: str = "fact", 
                 extract: Optional[bool] = None) -> Dict[str, Any]:
        """Store a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory (fact, event, preference, skill, observation)
            extract: Whether to use fact extraction (None uses server default)
            
        Returns:
            Dictionary with memory information
        """
        request = {
            "cmd": "remember",
            "content": content,
            "memory_type": memory_type
        }
        
        if extract is not None:
            request["extract"] = extract
            
        response = self._send_request(request)
        return response["data"]
    
    def recall(self, context: str = "", limit: int = 10,
               explain: bool = False) -> List[Dict[str, Any]]:
        """Retrieve memories based on context.
        
        Args:
            context: Search context/query
            limit: Maximum number of memories to return
            explain: Include score breakdown in results
            
        Returns:
            List of memory dictionaries
        """
        request = {
            "cmd": "recall", 
            "context": context,
            "limit": limit,
            "explain": explain,
        }
        
        response = self._send_request(request)
        return response["data"]

    def list(self, limit: int = 50, offset: int = 0,
             sort: str = "recent") -> List[Dict[str, Any]]:
        """List memories without a query."""
        request = {"cmd": "list", "limit": limit, "offset": offset, "sort": sort}
        response = self._send_request(request)
        return response["data"]

    def count(self) -> int:
        """Return total memory count."""
        request = {"cmd": "count"}
        response = self._send_request(request)
        return response["data"]["count"]

    def browse(self, concept: str, limit: int = 50,
               offset: int = 0) -> List[Dict[str, Any]]:
        """Browse memories by concept."""
        request = {"cmd": "browse", "concept": concept, "limit": limit, "offset": offset}
        response = self._send_request(request)
        return response["data"]

    def forget(self, memory_id: int) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if memory was deleted, False if it didn't exist
        """
        request = {
            "cmd": "forget",
            "id": memory_id
        }
        
        response = self._send_request(request)
        return response["data"]["deleted"]
    
    def link(self, source_id: int, target_id: int, edge_type: str = "related", 
             weight: float = 1.0):
        """Create a link between two memories.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID  
            edge_type: Type of relationship
            weight: Link weight/strength
        """
        request = {
            "cmd": "link",
            "source": source_id,
            "target": target_id,
            "edge_type": edge_type,
            "weight": weight
        }
        
        self._send_request(request)
    
    def concepts(self) -> List[Dict[str, Any]]:
        """Get all concepts with their memory counts."""
        request = {"cmd": "concepts"}
        response = self._send_request(request)
        return response["data"]

    def hot_concepts(self, k: int = 10) -> List[tuple[str, float]]:
        """Get the top-k most active concepts."""
        request = {"cmd": "hot_concepts", "k": k}
        response = self._send_request(request)
        return [tuple(x) for x in response["data"]]

    def prune(self, *, min_strength: float = 0.1, min_access: int = 0,
              max_age_days: float = 90, dry_run: bool = True) -> List[int]:
        """Auto-prune weak, old memories."""
        request = {
            "cmd": "prune",
            "min_strength": min_strength,
            "min_access": min_access,
            "max_age_days": max_age_days,
            "dry_run": dry_run,
        }
        response = self._send_request(request)
        return list(response["data"])
    
    def stats(self) -> Dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Dictionary with server stats
        """
        request = {"cmd": "stats"}
        response = self._send_request(request)
        return response["data"]
    
    def ping(self) -> str:
        """Ping the server.
        
        Returns:
            "pong" if server is responsive
        """
        request = {"cmd": "ping"}
        response = self._send_request(request)
        return response["data"]
    
    def shutdown(self):
        """Shutdown the server."""
        request = {"cmd": "shutdown"}
        try:
            self._send_request(request)
        except SynapseConnectionError:
            # Expected - server shuts down after responding
            pass
    
    def close(self):
        """Close the client connection."""
        self.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Simple test/demo of the client."""
    try:
        with SynapseClient() as client:
            print("🔗 Connected to Synapse daemon")
            
            # Test ping
            pong = client.ping()
            print(f"📡 Ping: {pong}")
            
            # Test stats
            stats = client.stats()
            print(f"📊 Stats: {stats}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()