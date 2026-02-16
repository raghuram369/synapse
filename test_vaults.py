"""Test cases for per-user vaults feature."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from synapse import Synapse, create_vault_manager, create_synapse_with_vaults
from vault_manager import VaultManager, VaultRegistry


class TestVaultRegistry(unittest.TestCase):
    """Test vault registry functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = VaultRegistry(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_register_vault(self):
        """Test vault registration."""
        result = self.registry.register_vault("test_vault", "user123")
        self.assertTrue(result)
        
        # Can't register same vault twice
        result = self.registry.register_vault("test_vault", "user123")
        self.assertFalse(result)
    
    def test_get_vault_info(self):
        """Test getting vault information."""
        self.registry.register_vault("test_vault", "user123", {"custom": "data"})
        
        info = self.registry.get_vault_info("test_vault")
        self.assertIsNotNone(info)
        self.assertEqual(info["vault_id"], "test_vault")
        self.assertEqual(info["user_id"], "user123")
        self.assertEqual(info["metadata"]["custom"], "data")
        
        # Non-existent vault
        info = self.registry.get_vault_info("nonexistent")
        self.assertIsNone(info)
    
    def test_find_user_vault(self):
        """Test finding vault by user ID."""
        self.registry.register_vault("vault1", "user123")
        self.registry.register_vault("vault2", "user456")
        
        vault_id = self.registry.find_user_vault("user123")
        self.assertEqual(vault_id, "vault1")
        
        vault_id = self.registry.find_user_vault("user456")
        self.assertEqual(vault_id, "vault2")
        
        vault_id = self.registry.find_user_vault("nonexistent")
        self.assertIsNone(vault_id)
    
    def test_list_vaults(self):
        """Test listing all vaults."""
        self.registry.register_vault("vault1", "user123")
        self.registry.register_vault("vault2", "user456")
        
        vaults = self.registry.list_vaults()
        self.assertEqual(len(vaults), 2)
        vault_ids = [v["vault_id"] for v in vaults]
        self.assertIn("vault1", vault_ids)
        self.assertIn("vault2", vault_ids)


class TestVaultManager(unittest.TestCase):
    """Test vault manager functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = VaultManager(self.temp_dir)
    
    def tearDown(self):
        self.manager.close_all_vaults()
        shutil.rmtree(self.temp_dir)
    
    def test_get_vault_path(self):
        """Test vault path generation."""
        path = self.manager.get_vault_path("test_vault")
        expected = os.path.join(self.temp_dir, "vault_test_vault", "synapse_store")
        self.assertEqual(path, expected)
    
    def test_get_or_create_vault(self):
        """Test getting or creating vaults."""
        vault1 = self.manager.get_or_create_vault("test_vault", "user123")
        self.assertIsInstance(vault1, Synapse)
        self.assertEqual(vault1._vault_id, "test_vault")
        self.assertEqual(vault1._user_id, "user123")
        
        # Should return same instance for same vault
        vault2 = self.manager.get_or_create_vault("test_vault", "user123")
        self.assertIs(vault1, vault2)
    
    def test_get_user_vault(self):
        """Test getting vault by user ID."""
        # Create vault for user
        vault = self.manager.get_user_vault("user123", auto_create=True)
        self.assertIsInstance(vault, Synapse)
        self.assertEqual(vault._user_id, "user123")
        
        # Should return same vault for same user
        vault2 = self.manager.get_user_vault("user123", auto_create=True)
        self.assertIs(vault, vault2)
        
        # No auto-create
        vault3 = self.manager.get_user_vault("newuser", auto_create=False)
        self.assertIsNone(vault3)
    
    def test_vault_isolation(self):
        """Test that vaults are isolated from each other."""
        vault1 = self.manager.get_user_vault("user1")
        vault2 = self.manager.get_user_vault("user2")
        
        # Store different memories in each vault
        vault1.remember("User 1 likes pizza")
        vault2.remember("User 2 likes sushi")
        
        # Each vault should only see its own memories
        memories1 = vault1.recall("likes")
        memories2 = vault2.recall("likes")
        
        self.assertEqual(len(memories1), 1)
        self.assertEqual(len(memories2), 1)
        self.assertIn("pizza", memories1[0].content)
        self.assertIn("sushi", memories2[0].content)
    
    def test_memory_mode_vault_manager(self):
        """Test vault manager with in-memory storage."""
        manager = VaultManager(":memory:")
        
        vault1 = manager.get_or_create_vault("vault1", "user1")
        vault2 = manager.get_or_create_vault("vault2", "user2")
        
        vault1.remember("Memory in vault 1")
        vault2.remember("Memory in vault 2")
        
        memories1 = vault1.recall("Memory")
        memories2 = vault2.recall("Memory")
        
        self.assertEqual(len(memories1), 1)
        self.assertEqual(len(memories2), 1)
        self.assertIn("vault 1", memories1[0].content)
        self.assertIn("vault 2", memories2[0].content)


class TestSynapseVaultIntegration(unittest.TestCase):
    """Test Synapse integration with vaults."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vault_manager = VaultManager(self.temp_dir)
        self.synapse = Synapse(path=":memory:", vault_manager=self.vault_manager)
    
    def tearDown(self):
        self.vault_manager.close_all_vaults()
        shutil.rmtree(self.temp_dir)
    
    def test_user_routing_remember(self):
        """Test user routing in remember method."""
        # Remember with user_id should route to user vault
        memory = self.synapse.remember("Test memory", user_id="user123")
        self.assertIsNotNone(memory)
        
        # Check that memory was stored in user vault, not main instance
        main_memories = self.synapse.recall("Test memory")
        self.assertEqual(len(main_memories), 0)
        
        # Check user vault has the memory
        user_vault = self.vault_manager.get_user_vault("user123")
        user_memories = user_vault.recall("Test memory")
        self.assertEqual(len(user_memories), 1)
    
    def test_user_routing_recall(self):
        """Test user routing in recall method."""
        # Store memory in user vault
        user_vault = self.vault_manager.get_user_vault("user123")
        user_vault.remember("User specific memory")
        
        # Recall with user_id should route to user vault
        memories = self.synapse.recall("User specific", user_id="user123")
        self.assertEqual(len(memories), 1)
        
        # Recall without user_id should not find it
        memories = self.synapse.recall("User specific")
        self.assertEqual(len(memories), 0)
    
    def test_no_vault_manager(self):
        """Test behavior when no vault manager is configured."""
        synapse = Synapse(path=":memory:")  # No vault manager
        
        # Should store normally even with user_id
        memory = synapse.remember("Test memory", user_id="user123")
        self.assertIsNotNone(memory)
        
        # Should find the memory
        memories = synapse.recall("Test memory")
        self.assertEqual(len(memories), 1)


class TestSynapseFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating Synapse instances with vaults."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._to_close = []
    
    def tearDown(self):
        for obj in self._to_close:
            try:
                obj.close()
            except Exception:
                pass
        shutil.rmtree(self.temp_dir)
    
    def test_create_vault_manager(self):
        """Test create_vault_manager factory function."""
        manager = create_vault_manager(self.temp_dir)
        self._to_close.append(manager)
        self.assertIsInstance(manager, VaultManager)
        self.assertEqual(manager.base_path, self.temp_dir)
    
    def test_create_synapse_with_vaults(self):
        """Test create_synapse_with_vaults factory function."""
        synapse = create_synapse_with_vaults(self.temp_dir)
        self._to_close.append(synapse)
        self.assertIsInstance(synapse, Synapse)
        self.assertIsNotNone(synapse._vault_manager)
        
        # Should be able to use user routing
        memory = synapse.remember("Test memory", user_id="user123")
        memories = synapse.recall("Test memory", user_id="user123")
        self.assertEqual(len(memories), 1)


if __name__ == '__main__':
    unittest.main()