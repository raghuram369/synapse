"""Test cases for memory inbox feature."""

import os
import shutil
import tempfile
import time
import unittest

from synapse import Synapse
from memory_inbox import MemoryInbox, AutoApproveRules


class TestAutoApproveRules(unittest.TestCase):
    """Test auto-approve rules functionality."""
    
    def setUp(self):
        self.rules = AutoApproveRules()
    
    def test_default_config(self):
        """Test default auto-approve configuration."""
        config = self.rules._default_config()
        self.assertTrue(config["enabled"])
        self.assertTrue(config["preferences"]["enabled"])
        self.assertTrue(config["goals"]["enabled"])
        self.assertTrue(config["facts"]["enabled"])
        self.assertFalse(config["personal_info"]["enabled"])
    
    def test_preference_auto_approve(self):
        """Test auto-approval of preferences."""
        # Should auto-approve preference content
        self.assertTrue(self.rules.should_auto_approve("I prefer coffee", "preference"))
        self.assertTrue(self.rules.should_auto_approve("I like pizza", "fact"))
        self.assertTrue(self.rules.should_auto_approve("I hate mornings", "fact"))
        
        # Should not auto-approve non-preference content
        self.assertFalse(self.rules.should_auto_approve("The weather is nice", "fact"))
    
    def test_goal_auto_approve(self):
        """Test auto-approval of goals."""
        # Should auto-approve goal content
        self.assertTrue(self.rules.should_auto_approve("I want to learn Python", "fact"))
        self.assertTrue(self.rules.should_auto_approve("My goal is to run a marathon", "fact"))
        self.assertTrue(self.rules.should_auto_approve("I plan to visit Japan", "fact"))
        
        # Should not auto-approve non-goal content
        self.assertFalse(self.rules.should_auto_approve("I went to the store", "fact"))
    
    def test_uncertain_fact_rejection(self):
        """Test rejection of uncertain facts."""
        # Should not auto-approve uncertain facts
        self.assertFalse(self.rules.should_auto_approve("Maybe it will rain", "fact"))
        self.assertFalse(self.rules.should_auto_approve("I might go shopping", "fact"))
        self.assertFalse(self.rules.should_auto_approve("I'm not sure about that", "fact"))
        
        # Certain facts are still conservative by default (manual review)
        self.assertFalse(self.rules.should_auto_approve("It is raining", "fact"))
    
    def test_personal_info_security(self):
        """Test personal info security (should not auto-approve by default)."""
        # Should not auto-approve potential PII by default
        self.assertFalse(self.rules.should_auto_approve("My SSN is 123-45-6789", "fact"))
        self.assertFalse(self.rules.should_auto_approve("Call me at 555-123-4567", "fact"))
    
    def test_disabled_rules(self):
        """Test behavior when auto-approve is disabled."""
        config = {"enabled": False}
        rules = AutoApproveRules(config)
        
        # Should not auto-approve anything when disabled
        self.assertFalse(rules.should_auto_approve("I prefer coffee", "preference"))
        self.assertFalse(rules.should_auto_approve("I want to learn Python", "fact"))


class TestMemoryInbox(unittest.TestCase):
    """Test memory inbox functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.synapse = Synapse(path=":memory:")
        self.inbox = MemoryInbox(self.synapse, self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_auto_approve_submission(self):
        """Test submission with auto-approval."""
        result = self.inbox.submit("I prefer tea over coffee", "preference")
        
        # Should be auto-approved
        self.assertEqual(result["status"], "auto_approved")
        self.assertIsNotNone(result["memory_id"])
        
        # Should be stored in synapse
        memories = self.synapse.recall("tea")
        self.assertEqual(len(memories), 1)
        self.assertTrue(memories[0].metadata.get("auto_approved", False))
    
    def test_manual_review_submission(self):
        """Test submission requiring manual review."""
        result = self.inbox.submit("The weather might be nice tomorrow", "fact")
        
        # Should require review (due to 'might' uncertainty)
        self.assertEqual(result["status"], "pending")
        self.assertIsNotNone(result["item_id"])
        self.assertIsNone(result["memory_id"])
        
        # Should not be in synapse yet
        memories = self.synapse.recall("weather")
        self.assertEqual(len(memories), 0)
    
    def test_list_pending(self):
        """Test listing pending memories."""
        # Submit something for manual review
        self.inbox.submit("Maybe I should exercise", "fact")
        self.inbox.submit("I might visit the store", "fact")
        
        pending = self.inbox.list_pending()
        self.assertEqual(len(pending), 2)
        
        # Check structure
        self.assertIn("id", pending[0])
        self.assertIn("content", pending[0])
        self.assertIn("submitted_at", pending[0])
    
    def test_approve_memory(self):
        """Test approving a pending memory."""
        # Submit for review
        result = self.inbox.submit("I'm not sure about this", "fact")
        item_id = result["item_id"]
        
        # Approve it
        memory = self.inbox.approve(item_id)
        self.assertIsNotNone(memory)
        
        # Should be in synapse now
        memories = self.synapse.recall("not sure")
        self.assertEqual(len(memories), 1)
        
        # Should not be pending anymore
        pending = self.inbox.list_pending()
        self.assertEqual(len(pending), 0)
    
    def test_reject_memory(self):
        """Test rejecting a pending memory."""
        # Submit for review
        result = self.inbox.submit("I might do something", "fact")
        item_id = result["item_id"]
        
        # Reject it
        success = self.inbox.reject(item_id)
        self.assertTrue(success)
        
        # Should not be in synapse
        memories = self.synapse.recall("might do")
        self.assertEqual(len(memories), 0)
        
        # Should not be pending
        pending = self.inbox.list_pending()
        self.assertEqual(len(pending), 0)
    
    def test_redact_memory(self):
        """Test redacting a pending memory."""
        # Submit for review
        result = self.inbox.submit("My phone number is 555-1234", "fact")
        item_id = result["item_id"]
        
        # Redact it
        memory = self.inbox.redact(item_id, "My phone number is [REDACTED]")
        self.assertIsNotNone(memory)
        
        # Should store redacted version
        memories = self.synapse.recall("phone number")
        self.assertEqual(len(memories), 1)
        self.assertIn("[REDACTED]", memories[0].content)
        self.assertNotIn("555-1234", memories[0].content)
        
        # Should have redaction metadata but NOT original content
        self.assertTrue(memories[0].metadata.get("redacted", False))
        self.assertNotIn("original_content", memories[0].metadata)
    
    def test_pin_memory(self):
        """Test pinning a pending memory."""
        # Submit for review
        result = self.inbox.submit("This is important but uncertain", "fact")
        item_id = result["item_id"]
        
        # Pin it
        memory = self.inbox.pin(item_id)
        self.assertIsNotNone(memory)
        
        # Should be stored with pin metadata
        memories = self.synapse.recall("important")
        self.assertEqual(len(memories), 1)
        self.assertTrue(memories[0].metadata.get("pinned", False))
        self.assertEqual(memories[0].metadata.get("importance"), "high")
    
    def test_query_pending(self):
        """Test querying pending memories."""
        # Submit multiple items
        self.inbox.submit("I might go shopping", "fact")
        self.inbox.submit("Maybe I'll cook dinner", "fact")
        self.inbox.submit("Perhaps I should exercise", "fact")
        
        # Query for specific content
        results = self.inbox.query_pending("shopping")
        self.assertEqual(len(results), 1)
        self.assertIn("shopping", results[0]["content"])
        
        # Query for common word
        results = self.inbox.query_pending("I")
        self.assertEqual(len(results), 3)  # All contain "I"
    
    def test_force_review(self):
        """Test forcing manual review even for auto-approve content."""
        result = self.inbox.submit("I prefer coffee", "preference", force_review=True)
        
        # Should be pending despite being preference
        self.assertEqual(result["status"], "pending")
        self.assertIsNotNone(result["item_id"])
    
    def test_count_stats(self):
        """Test inbox statistics."""
        # Submit some items
        self.inbox.submit("Maybe this", "fact")
        self.inbox.submit("Perhaps that", "fact")
        
        stats = self.inbox.count()
        self.assertEqual(stats["total_pending"], 2)


class TestSynapseInboxIntegration(unittest.TestCase):
    """Test Synapse integration with inbox mode."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.synapse = Synapse(path=":memory:", inbox_mode=True)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_inbox_mode_remember(self):
        """Test remember method in inbox mode."""
        # Auto-approved content
        memory = self.synapse.remember("I like pizza", "preference")
        self.assertIsNotNone(memory.id)  # Should have real ID
        
        # Content requiring review
        memory = self.synapse.remember("Maybe I should exercise", "fact")
        self.assertIsNone(memory.id)  # Should have no ID (pending)
        self.assertTrue(memory.metadata.get("pending", False))
        self.assertIn("item_id", memory.metadata)
    
    def test_inbox_methods(self):
        """Test inbox-related methods on Synapse."""
        # Enable inbox mode
        synapse = Synapse(path=":memory:", inbox_mode=True)
        
        # Submit something for review
        synapse.remember("I might do something", "fact")
        
        # Test inbox methods
        pending = synapse.list_pending()
        self.assertEqual(len(pending), 1)
        
        item_id = pending[0]["id"]
        
        # Test approve
        memory = synapse.approve_memory(item_id)
        self.assertIsNotNone(memory)
        
        # Should be stored now
        memories = synapse.recall("might do")
        self.assertEqual(len(memories), 1)
    
    def test_no_inbox_mode(self):
        """Test Synapse without inbox mode enabled."""
        synapse = Synapse(path=":memory:", inbox_mode=False)
        
        # Should return empty list for inbox methods
        self.assertEqual(len(synapse.list_pending()), 0)
        self.assertIsNone(synapse.approve_memory("fake_id"))
        self.assertFalse(synapse.reject_memory("fake_id"))
    
    def test_custom_inbox_config(self):
        """Test Synapse with custom inbox configuration."""
        config = {
            "preferences": {"enabled": False},  # Disable preference auto-approve
            "facts": {"enabled": False}         # Disable fact auto-approve
        }
        
        synapse = Synapse(path=":memory:", inbox_mode=True, inbox_config=config)
        
        # Even preferences should require review now
        memory = synapse.remember("I prefer tea", "preference")
        self.assertIsNone(memory.id)  # Should be pending
        self.assertTrue(memory.metadata.get("pending", False))


if __name__ == '__main__':
    unittest.main()