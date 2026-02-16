"""Test cases for natural language forget feature."""

import time
import unittest
from datetime import datetime, timedelta

from synapse import Synapse
from natural_forget import ForgetPatternMatcher, NaturalForget


class TestForgetPatternMatcher(unittest.TestCase):
    """Test pattern matching for natural language forget commands."""
    
    def setUp(self):
        self.matcher = ForgetPatternMatcher()
    
    def test_specific_fact_patterns(self):
        """Test specific fact deletion patterns."""
        # Basic forget patterns
        result = self.matcher.parse_forget_command("forget my phone number")
        self.assertEqual(result["type"], "specific_fact")
        self.assertEqual(result["matches"][0], "my phone number")
        
        result = self.matcher.parse_forget_command("delete about my old job")
        self.assertEqual(result["type"], "specific_fact")
        self.assertIn("old job", result["matches"][0])
        
        result = self.matcher.parse_forget_command("remove my address")
        self.assertEqual(result["type"], "specific_fact")
        self.assertEqual(result["matches"][0], "my address")
    
    def test_topic_bulk_patterns(self):
        """Test topic-based bulk deletion patterns."""
        result = self.matcher.parse_forget_command("forget everything about my ex-coworker")
        self.assertEqual(result["type"], "topic_bulk")
        self.assertEqual(result["matches"][0], "my ex-coworker")
        
        result = self.matcher.parse_forget_command("delete anything related to Sarah")
        self.assertEqual(result["type"], "topic_bulk")
        self.assertEqual(result["matches"][0], "sarah")
        
        result = self.matcher.parse_forget_command("remove all regarding my old company")
        self.assertEqual(result["type"], "topic_bulk")
        self.assertEqual(result["matches"][0], "my old company")
    
    def test_time_based_patterns(self):
        """Test time-based cleanup patterns."""
        result = self.matcher.parse_forget_command("forget everything older than 30 days")
        self.assertEqual(result["type"], "time_based")
        self.assertEqual(result["matches"][0], "30")
        self.assertEqual(result["matches"][1], "day")
        
        result = self.matcher.parse_forget_command("delete memories older than 2 weeks")
        self.assertEqual(result["type"], "time_based")
        self.assertEqual(result["matches"][0], "2")
        self.assertEqual(result["matches"][1], "week")
        
        result = self.matcher.parse_forget_command("remove anything from before January 2024")
        self.assertEqual(result["type"], "time_based")
        self.assertEqual(result["matches"][0], "january 2024")
    
    def test_memory_type_patterns(self):
        """Test memory type specific patterns."""
        result = self.matcher.parse_forget_command("forget all preferences about food")
        self.assertEqual(result["type"], "by_memory_type")
        self.assertEqual(result["matches"][0], "preference")
        self.assertEqual(result["matches"][1], "food")
        
        result = self.matcher.parse_forget_command("delete my observations")
        self.assertEqual(result["type"], "by_memory_type")
        self.assertEqual(result["matches"][0], "observation")
    
    def test_update_patterns(self):
        """Test update/correction patterns."""
        result = self.matcher.parse_forget_command("that changed, I moved to Seattle")
        self.assertEqual(result["type"], "update")
        
        result = self.matcher.parse_forget_command("correction: I'm 30 years old")
        self.assertEqual(result["type"], "update")
    
    def test_fallback_pattern(self):
        """Test fallback to specific fact pattern."""
        result = self.matcher.parse_forget_command("some random text")
        self.assertEqual(result["type"], "specific_fact")
        self.assertEqual(result["matches"][0], "some random text")
        self.assertEqual(result["matched_pattern"], "fallback")
    
    def test_extract_time_constraint(self):
        """Test time constraint extraction."""
        # Test relative time
        now = time.time()
        
        # 30 days ago
        timestamp = self.matcher.extract_time_constraint("30 days")
        expected = now - (30 * 24 * 3600)
        self.assertAlmostEqual(timestamp, expected, delta=5)  # 5 second tolerance
        
        # 2 weeks ago
        timestamp = self.matcher.extract_time_constraint("2 weeks")
        expected = now - (2 * 7 * 24 * 3600)
        self.assertAlmostEqual(timestamp, expected, delta=5)
        
        # Test specific dates
        timestamp = self.matcher.extract_time_constraint("2024-01-01")
        expected = datetime(2024, 1, 1).timestamp()
        self.assertEqual(timestamp, expected)
        
        # Test month/year
        timestamp = self.matcher.extract_time_constraint("January 2024")
        expected = datetime(2024, 1, 1).timestamp()
        self.assertEqual(timestamp, expected)
        
        # Invalid time string
        timestamp = self.matcher.extract_time_constraint("invalid")
        self.assertIsNone(timestamp)


class TestNaturalForget(unittest.TestCase):
    """Test natural language forget processing."""
    
    def setUp(self):
        self.synapse = Synapse(path=":memory:")
        self.forget_processor = NaturalForget(self.synapse)
        
        # Add some test memories
        self.synapse.remember("My phone number is 555-1234")
        self.synapse.remember("I work at Acme Corp")
        self.synapse.remember("Sarah is my coworker")
        self.synapse.remember("I like pizza")
        self.synapse.remember("Meeting with John tomorrow", memory_type="event")
        
        # Add an old memory for time-based tests
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        old_memory = self.synapse.remember("Old memory to delete")
        # Manually set the created_at timestamp
        if old_memory.id:
            self.synapse.store.memories[old_memory.id]["created_at"] = old_time
    
    def test_forget_specific_fact(self):
        """Test forgetting specific facts."""
        result = self.forget_processor.process_forget_command("forget my phone number", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("phone number" in c for c in contents))
        self.assertTrue(any("pizza" in c for c in contents))
    
    def test_forget_topic_bulk(self):
        """Test forgetting everything about a topic."""
        result = self.forget_processor.process_forget_command("forget everything about Sarah", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("sarah" in c for c in contents))
        self.assertTrue(any("pizza" in c for c in contents))
    
    def test_forget_time_based(self):
        """Test forgetting based on time constraints."""
        result = self.forget_processor.process_forget_command("forget everything older than 30 days", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("old memory to delete" in c for c in contents))
        self.assertTrue(any("pizza" in c for c in contents))
    
    def test_forget_by_memory_type(self):
        """Test forgetting by memory type."""
        result = self.forget_processor.process_forget_command("forget all events", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("meeting with john" in c for c in contents))
        self.assertTrue(any("pizza" in c for c in contents))
    
    def test_dry_run_mode(self):
        """Test dry run mode (preview without deleting)."""
        result = self.forget_processor.process_forget_command("forget my phone number", confirm=False, dry_run=True)
        
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("memories", result)
        self.assertGreater(len(result["memories"]), 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertTrue(any("phone number" in c for c in contents))
    
    def test_not_found_case(self):
        """Test when no memories match the forget command."""
        result = self.forget_processor.process_forget_command("forget about unicorns", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "not_found")
        self.assertIn("No memories found", result["message"])
    
    def test_empty_search_term(self):
        """Test error handling for empty search terms."""
        result = self.forget_processor.process_forget_command("forget", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("search term", result["message"].lower())
    
    def test_update_command_guidance(self):
        """Test update/correction commands provide guidance."""
        result = self.forget_processor.process_forget_command("that changed, I moved to Seattle", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "guidance")
        self.assertIn("Update commands", result["message"])
    
    def test_time_parsing_errors(self):
        """Test error handling for invalid time constraints."""
        result = self.forget_processor.process_forget_command("forget everything older than invalid_time", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Could not parse time", result["message"])
    
    def test_multiple_word_topics(self):
        """Test forgetting topics with multiple words."""
        self.synapse.remember("My old company was Acme Corp")
        self.synapse.remember("Acme Corp had great benefits")
        
        result = self.forget_processor.process_forget_command("forget everything about Acme Corp", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("acme corp" in c for c in contents))
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        result = self.forget_processor.process_forget_command("FORGET MY PHONE NUMBER", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
    
    def test_partial_word_matching(self):
        """Test that partial word matching works appropriately."""
        self.synapse.remember("I love programming")
        self.synapse.remember("Programming languages are interesting")
        
        result = self.forget_processor.process_forget_command("forget programming", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        self.assertGreater(result["deleted_count"], 0)
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("programming" in c for c in contents))


class TestSynapseNaturalForgetIntegration(unittest.TestCase):
    """Test Synapse integration with natural forget."""
    
    def setUp(self):
        self.synapse = Synapse(path=":memory:")
        self.synapse.remember("I like chocolate")
        self.synapse.remember("My favorite color is blue")
        self.synapse.remember("I work at TechCorp")
    
    def test_natural_forget_method(self):
        """Test natural_forget method on Synapse instance."""
        result = self.synapse.natural_forget("forget my favorite color", confirm=False, dry_run=False)
        
        self.assertEqual(result["status"], "deleted")
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertFalse(any("favorite color" in c for c in contents))
        self.assertTrue(any("chocolate" in c for c in contents))
    
    def test_natural_forget_dry_run(self):
        """Test natural forget with dry run."""
        result = self.synapse.natural_forget("forget TechCorp", dry_run=True)
        
        self.assertEqual(result["status"], "dry_run")
        
        contents = [m["content"].lower() for m in self.synapse.store.memories.values()]
        self.assertTrue(any("techcorp" in c for c in contents))
    
    def test_suggest_forget_commands(self):
        """Test forget command suggestions."""
        from natural_forget import NaturalForget
        forget_processor = NaturalForget(self.synapse)
        
        suggestions = forget_processor.suggest_forget_commands()
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Each suggestion should have command and description
        for suggestion in suggestions:
            self.assertIn("command", suggestion)
            self.assertIn("description", suggestion)


if __name__ == '__main__':
    unittest.main()