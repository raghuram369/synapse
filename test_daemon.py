#!/usr/bin/env python3
"""Tests for the Synapse daemon and client."""

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from client import SynapseClient, SynapseConnectionError, SynapseRequestError
from synapsed import SynapseServer


class TestSynapseClient(unittest.TestCase):
    """Test the client library in isolation."""

    def test_init(self):
        """Test client initialization."""
        client = SynapseClient(host="test.com", port=9999, timeout=60.0)
        
        self.assertEqual(client.host, "test.com")
        self.assertEqual(client.port, 9999)
        self.assertEqual(client.timeout, 60.0)
        self.assertFalse(client._connected)

    def test_connect_failure(self):
        """Test connection failure."""
        client = SynapseClient(host="nonexistent.host", port=9999, timeout=1.0)
        
        with self.assertRaises(SynapseConnectionError):
            client.connect()

    def test_context_manager(self):
        """Test client as context manager."""
        # This will fail to connect, but we're testing the context manager behavior
        client = SynapseClient(host="nonexistent.host", port=9999, timeout=0.1)
        
        with self.assertRaises(SynapseConnectionError):
            with client:
                pass

    def test_request_error_handling(self):
        """Test request error handling."""
        client = SynapseClient()
        
        # Mock a connected state and file
        client._connected = True
        
        class MockFile:
            def write(self, data):
                pass
            def flush(self):
                pass
            def readline(self):
                return '{"ok": false, "error": "Test error"}\n'
        
        client._file = MockFile()
        
        with self.assertRaises(SynapseRequestError) as cm:
            client._send_request({"cmd": "test"})
        
        self.assertIn("Test error", str(cm.exception))


class TestSynapseIntegration(unittest.TestCase):
    """Integration tests with actual daemon and client."""

    def setUp(self):
        """Setup test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_synapse"
        
        # Start daemon on a random available port
        self.server = SynapseServer(
            host="127.0.0.1",
            port=0,  # Let OS choose available port
            data_dir=str(self.data_dir),
            extract_default=False
        )
        
        # Start server in background thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start and get the actual port
        time.sleep(0.1)
        self.server_port = self._get_server_port()
        
        # Create client
        self.client = SynapseClient(host="127.0.0.1", port=self.server_port, timeout=5.0)

    def tearDown(self):
        """Clean up test environment."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass
            
        try:
            if hasattr(self, 'server'):
                self.server.running = False
                if self.server.server_socket:
                    self.server.server_socket.close()
        except:
            pass
            
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def _run_server(self):
        """Run server (called in background thread)."""
        try:
            # Create server socket to get available port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            
            # Get the port and store it
            self.server_port = sock.getsockname()[1]
            
            # Update server config and use this socket
            self.server.port = self.server_port
            self.server.server_socket = sock
            
            # Start server (modified to use existing socket)
            sock.listen(10)
            self.server.running = True
            
            while self.server.running:
                try:
                    client_socket, client_addr = sock.accept()
                    self.server.client_count += 1
                    client_id = self.server.client_count
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.server._handle_client,
                        args=(client_socket, client_id),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error:
                    if self.server.running:
                        break
                        
        except Exception as e:
            print(f"Server error: {e}")

    def _get_server_port(self):
        """Get the server port (wait for it to be set)."""
        for _ in range(50):  # Wait up to 5 seconds
            if hasattr(self, 'server_port'):
                return self.server_port
            time.sleep(0.1)
        raise RuntimeError("Server failed to start")

    def test_ping(self):
        """Test ping command."""
        response = self.client.ping()
        self.assertEqual(response, "pong")

    def test_remember_basic(self):
        """Test basic remember operation."""
        memory = self.client.remember("The sky is blue", memory_type="fact")
        
        self.assertIsInstance(memory, dict)
        self.assertEqual(memory['content'], "The sky is blue")
        self.assertEqual(memory['memory_type'], "fact")
        self.assertIn('id', memory)
        self.assertIn('strength', memory)

    @patch('synapse.extract_facts')
    def test_remember_with_extraction(self, mock_extract):
        """Test remember with fact extraction."""
        # Mock extraction to return multiple facts
        mock_extract.return_value = [
            "Caroline is researching adoption agencies",
            "Adoption has been on Caroline's mind"
        ]
        
        memory = self.client.remember(
            "Caroline: Yeah I've been looking into adoption agencies recently, it's been on my mind a lot",
            extract=True
        )
        
        self.assertIsInstance(memory, dict)
        self.assertIn('id', memory)
        
        # Should have called the extractor
        mock_extract.assert_called_once()

    def test_remember_and_recall(self):
        """Test remember followed by recall."""
        # Remember a few memories
        mem1 = self.client.remember("Python is a programming language", memory_type="fact")
        mem2 = self.client.remember("I love coding in Python", memory_type="preference")
        mem3 = self.client.remember("JavaScript is also useful", memory_type="fact")
        
        # Recall with context
        memories = self.client.recall("Python programming", limit=5)
        
        self.assertIsInstance(memories, list)
        self.assertTrue(len(memories) >= 2)  # Should find at least the Python-related memories
        
        # Check that we got relevant results
        contents = [m['content'] for m in memories]
        self.assertTrue(any("Python" in content for content in contents))

    def test_forget(self):
        """Test forget operation."""
        # Remember something
        memory = self.client.remember("This will be forgotten", memory_type="fact")
        memory_id = memory['id']
        
        # Verify it exists
        memories = self.client.recall("forgotten")
        self.assertTrue(any(m['id'] == memory_id for m in memories))
        
        # Forget it
        deleted = self.client.forget(memory_id)
        self.assertTrue(deleted)
        
        # Verify it's gone
        memories = self.client.recall("forgotten")
        self.assertFalse(any(m['id'] == memory_id for m in memories))

    def test_link_memories(self):
        """Test linking memories."""
        # Remember two related memories
        mem1 = self.client.remember("Cause event happened", memory_type="event")
        mem2 = self.client.remember("Effect occurred", memory_type="event")
        
        # Link them
        self.client.link(mem2['id'], mem1['id'], edge_type="caused_by", weight=0.8)
        
        # This should succeed without error
        # (We can't easily verify the link exists without additional query methods)

    def test_concepts(self):
        """Test concepts listing."""
        # Remember some memories to create concepts
        self.client.remember("I love Python programming", memory_type="preference")
        self.client.remember("Python is great for data science", memory_type="fact")
        
        concepts = self.client.concepts()
        
        self.assertIsInstance(concepts, list)
        # Should have at least some concepts from the memories
        self.assertTrue(len(concepts) > 0)
        
        # Check concept structure
        if concepts:
            concept = concepts[0]
            self.assertIn('name', concept)
            self.assertIn('memory_count', concept)

    def test_stats(self):
        """Test stats retrieval."""
        # Remember a few memories
        self.client.remember("Memory 1", memory_type="fact")
        self.client.remember("Memory 2", memory_type="event")
        
        stats = self.client.stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_memories', stats)
        self.assertIn('total_concepts', stats)
        self.assertIn('total_edges', stats)
        self.assertTrue(stats['total_memories'] >= 2)

    def test_error_handling(self):
        """Test error handling for invalid requests."""
        # Test invalid memory type
        with self.assertRaises(SynapseRequestError):
            self.client.remember("Test", memory_type="invalid_type")
        
        # Test forgetting non-existent memory
        deleted = self.client.forget(99999)
        self.assertFalse(deleted)  # Should return False, not raise error

    def test_multiple_clients(self):
        """Test multiple concurrent clients."""
        # Create a second client
        client2 = SynapseClient(host="127.0.0.1", port=self.server_port)
        
        try:
            # Both clients should be able to operate
            mem1 = self.client.remember("Memory from client 1", memory_type="fact")
            mem2 = client2.remember("Memory from client 2", memory_type="fact")
            
            # Both should see each other's memories
            memories1 = self.client.recall("Memory from", limit=10)
            memories2 = client2.recall("Memory from", limit=10)
            
            self.assertTrue(len(memories1) >= 2)
            self.assertTrue(len(memories2) >= 2)
            
        finally:
            client2.close()

    def test_auto_reconnect(self):
        """Test auto-reconnect functionality."""
        # This is harder to test without actually killing the connection
        # For now, just verify the client has auto_reconnect enabled by default
        client = SynapseClient()
        self.assertTrue(client.auto_reconnect)


if __name__ == "__main__":
    unittest.main()