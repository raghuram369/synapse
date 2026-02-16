"""Tests for the Memory Router capture pipeline."""

import pytest
from unittest.mock import Mock, patch

from capture import (
    PreferenceDetector, FactDetector, DecisionDetector, PlanDetector,
    CorrectionDetector, OutcomeDetector, SecretFilter, FluffFilter,
    ingest, IngestResult, ClassificationSignal
)


class TestPreferenceDetector:
    """Test preference detection patterns."""
    
    def test_basic_preferences(self):
        """Test basic preference statements."""
        # Positive preferences
        result = PreferenceDetector.classify("I like chocolate")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "preference"
        
        result = PreferenceDetector.classify("We prefer React over Vue")
        assert result.should_store is True
        assert result.confidence > 0.0
        
        # Negative preferences
        result = PreferenceDetector.classify("I hate mondays")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_strong_preferences(self):
        """Test that strong preference words get higher confidence."""
        strong_result = PreferenceDetector.classify("I love Python")
        weak_result = PreferenceDetector.classify("I use Python")
        
        assert strong_result.confidence > weak_result.confidence
        
    def test_no_preference(self):
        """Test text without preferences."""
        result = PreferenceDetector.classify("The weather is nice today")
        assert result.should_store is False
        assert result.confidence == 0.0
        
    def test_complex_preferences(self):
        """Test more complex preference patterns."""
        result = PreferenceDetector.classify("My favorite editor is VS Code")
        assert result.should_store is True
        assert result.confidence > 0.0
        
        result = PreferenceDetector.classify("I wish we could use TypeScript")
        assert result.should_store is True
        assert result.confidence > 0.0


class TestFactDetector:
    """Test factual information detection."""
    
    def test_person_names(self):
        """Test person name detection."""
        result = FactDetector.classify("John Smith is the new manager")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "person"
        
    def test_locations(self):
        """Test location detection.""" 
        result = FactDetector.classify("I live in San Francisco")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "location"
        
        result = FactDetector.classify("The office is at 123 Main Street")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_jobs(self):
        """Test job/role detection."""
        result = FactDetector.classify("I work as a software engineer")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "job"
        
        result = FactDetector.classify("She is a doctor at the hospital")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_tools(self):
        """Test tool/technology detection."""
        result = FactDetector.classify("We use React for the frontend")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "tool"
        
        result = FactDetector.classify("I prefer VS Code for coding")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_no_facts(self):
        """Test text without factual information."""
        result = FactDetector.classify("That's interesting")
        assert result.confidence == 0.0


class TestDecisionDetector:
    """Test decision statement detection."""
    
    def test_basic_decisions(self):
        """Test basic decision patterns."""
        result = DecisionDetector.classify("Let's go with React")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "decision"
        
        result = DecisionDetector.classify("We decided to ship on Friday")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_strong_decisions(self):
        """Test that definitive decisions get higher confidence."""
        strong_result = DecisionDetector.classify("Final decision: we use TypeScript")
        weak_result = DecisionDetector.classify("Let's try TypeScript")
        
        assert strong_result.confidence > weak_result.confidence
        
    def test_action_items(self):
        """Test action item detection."""
        result = DecisionDetector.classify("Action item: update the docs")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_no_decisions(self):
        """Test text without decisions."""
        result = DecisionDetector.classify("The code looks good")
        assert result.confidence == 0.0


class TestPlanDetector:
    """Test planning statement detection."""
    
    def test_date_patterns(self):
        """Test date-based planning."""
        result = PlanDetector.classify("Meeting tomorrow at 2pm")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "plan"
        
        result = PlanDetector.classify("Deploy by Friday")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_specific_dates(self):
        """Test specific date formats."""
        result = PlanDetector.classify("Release on March 15th")
        assert result.should_store is True
        assert result.confidence > 0.0
        
        result = PlanDetector.classify("Deadline is 2024-03-15")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_planning_words(self):
        """Test planning indicator words."""
        result = PlanDetector.classify("I plan to finish this next week")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_no_plans(self):
        """Test text without planning information."""
        result = PlanDetector.classify("The code is working well")
        assert result.confidence == 0.0


class TestCorrectionDetector:
    """Test correction statement detection."""
    
    def test_explicit_corrections(self):
        """Test explicit correction patterns."""
        result = CorrectionDetector.classify("Actually, that's wrong")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "correction"
        
        result = CorrectionDetector.classify("Correction: it should be 5, not 3")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_strong_corrections(self):
        """Test that explicit corrections get higher confidence."""
        strong_result = CorrectionDetector.classify("That's completely wrong")
        weak_result = CorrectionDetector.classify("Actually, I think")
        
        assert strong_result.confidence > weak_result.confidence
        
    def test_mind_changes(self):
        """Test change of mind patterns."""
        result = CorrectionDetector.classify("On second thought, let's use React")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_no_corrections(self):
        """Test text without corrections."""
        result = CorrectionDetector.classify("That sounds good to me")
        assert result.confidence == 0.0


class TestOutcomeDetector:
    """Test outcome statement detection."""
    
    def test_success_outcomes(self):
        """Test successful outcome detection."""
        result = OutcomeDetector.classify("That worked perfectly")
        assert result.should_store is True
        assert result.confidence > 0.0
        assert result.category == "outcome"
        
        result = OutcomeDetector.classify("Successfully deployed to production")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_failure_outcomes(self):
        """Test failure outcome detection."""
        result = OutcomeDetector.classify("The build failed")
        assert result.should_store is True
        assert result.confidence > 0.0
        
        result = OutcomeDetector.classify("It crashed in production")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_completion_outcomes(self):
        """Test completion-based outcomes."""
        result = OutcomeDetector.classify("Feature is complete")
        assert result.should_store is True
        assert result.confidence > 0.0
        
    def test_no_outcomes(self):
        """Test text without outcomes."""
        result = OutcomeDetector.classify("We should check the logs")
        assert result.confidence == 0.0


class TestSecretFilter:
    """Test sensitive information filtering."""
    
    def test_api_keys(self):
        """Test API key detection."""
        assert SecretFilter.detect("sk-1234567890abcdef1234567890abcdef") is True
        assert SecretFilter.detect("ghp_abcdefghijklmnopqrstuvwxyz123456789abc") is True
        assert SecretFilter.detect("AKIA1234567890123456") is True
        
    def test_passwords(self):
        """Test password detection."""
        assert SecretFilter.detect("password=secret123") is True
        assert SecretFilter.detect("pwd: mypassword") is True
        
    def test_ssn(self):
        """Test SSN detection."""
        assert SecretFilter.detect("123-45-6789") is True
        assert SecretFilter.detect("123 45 6789") is True
        
    def test_credit_cards(self):
        """Test credit card detection."""
        assert SecretFilter.detect("4532-1234-5678-9012") is True
        assert SecretFilter.detect("4532 1234 5678 9012") is True
        
    def test_private_keys(self):
        """Test private key detection."""
        assert SecretFilter.detect("-----BEGIN PRIVATE KEY-----") is True
        assert SecretFilter.detect("-----BEGIN RSA PRIVATE KEY-----") is True
        
    def test_safe_content(self):
        """Test that normal content is not flagged."""
        assert SecretFilter.detect("This is normal text") is False
        assert SecretFilter.detect("The API returned 200") is False
        assert SecretFilter.detect("Password field is empty") is False


class TestFluffFilter:
    """Test conversational fluff filtering."""
    
    def test_simple_acknowledgments(self):
        """Test basic acknowledgment detection."""
        assert FluffFilter.detect("ok") is True
        assert FluffFilter.detect("OK") is True
        assert FluffFilter.detect("yeah") is True
        assert FluffFilter.detect("yep") is True
        assert FluffFilter.detect("sure") is True
        
    def test_thanks(self):
        """Test thank you detection."""
        assert FluffFilter.detect("thanks") is True
        assert FluffFilter.detect("thank you") is True
        assert FluffFilter.detect("thx") is True
        
    def test_reactions(self):
        """Test reaction detection."""
        assert FluffFilter.detect("cool") is True
        assert FluffFilter.detect("nice") is True
        assert FluffFilter.detect("awesome") is True
        assert FluffFilter.detect("lol") is True
        assert FluffFilter.detect("haha") is True
        
    def test_emojis(self):
        """Test emoji detection."""
        assert FluffFilter.detect("😂") is True
        assert FluffFilter.detect("👍") is True
        assert FluffFilter.detect("🙂") is True
        
    def test_agreement(self):
        """Test agreement phrases."""
        assert FluffFilter.detect("sounds good") is True
        assert FluffFilter.detect("makes sense") is True
        assert FluffFilter.detect("i agree") is True
        
    def test_greetings(self):
        """Test greeting detection."""
        assert FluffFilter.detect("hello") is True
        assert FluffFilter.detect("hi") is True
        assert FluffFilter.detect("bye") is True
        
    def test_substantive_content(self):
        """Test that substantive content is not flagged."""
        assert FluffFilter.detect("This is a detailed explanation") is False
        assert FluffFilter.detect("I need to implement the API") is False
        assert FluffFilter.detect("The bug is in line 42") is False


class TestMemoryRouter:
    """Test the main memory router ingest function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_synapse = Mock()
        self.mock_review_queue = Mock()
        
    def test_secret_rejection(self):
        """Test that secrets are rejected."""
        result = ingest("password=secret123", self.mock_synapse, self.mock_review_queue)
        assert result == IngestResult.REJECTED_SECRET
        assert not self.mock_synapse.remember.called
        assert not self.mock_review_queue.submit.called
        
    def test_fluff_ignoring(self):
        """Test that fluff is ignored."""
        result = ingest("ok", self.mock_synapse, self.mock_review_queue)
        assert result == IngestResult.IGNORED_FLUFF
        assert not self.mock_synapse.remember.called
        assert not self.mock_review_queue.submit.called
        
    def test_policy_off(self):
        """Test that policy=off ignores everything."""
        result = ingest("This is important information", self.mock_synapse, self.mock_review_queue, policy="off")
        assert result == IngestResult.IGNORED_POLICY
        assert not self.mock_synapse.remember.called
        assert not self.mock_review_queue.submit.called
        
    def test_auto_store_high_confidence(self):
        """Test auto-store for high confidence signals."""
        # Use a clear preference statement that should get high confidence
        result = ingest("I love using Python for data science", self.mock_synapse, self.mock_review_queue, policy="auto")
        assert result == IngestResult.STORED
        assert self.mock_synapse.remember.called
        
    def test_minimal_policy(self):
        """Test minimal policy only stores very high confidence."""
        # Medium confidence statement
        result = ingest("Python is useful", self.mock_synapse, self.mock_review_queue, policy="minimal")
        assert result == IngestResult.IGNORED_POLICY
        assert not self.mock_synapse.remember.called
        
    def test_review_policy(self):
        """Test that review policy sends everything to queue."""
        result = ingest("Some interesting information", self.mock_synapse, self.mock_review_queue, policy="review")
        assert result == IngestResult.QUEUED_FOR_REVIEW
        assert self.mock_review_queue.submit.called
        assert not self.mock_synapse.remember.called
        
    def test_auto_review_medium_confidence(self):
        """Test auto policy sends medium confidence to review."""
        # Create a statement that should trigger medium confidence
        with patch('capture.DETECTORS') as mock_detectors:
            # Mock a detector that returns medium confidence
            mock_detector = Mock()
            mock_detector.classify.return_value = ClassificationSignal(
                should_store=True,
                confidence=0.5,  # Medium confidence
                category="test",
                extracted="test content"
            )
            mock_detectors.__iter__.return_value = [mock_detector]
            
            result = ingest("test content", self.mock_synapse, self.mock_review_queue, policy="auto")
            assert result == IngestResult.QUEUED_FOR_REVIEW
            assert self.mock_review_queue.submit.called
            
    def test_metadata_attachment(self):
        """Test that metadata is properly attached."""
        metadata = {"test_key": "test_value"}
        ingest("I prefer TypeScript", self.mock_synapse, self.mock_review_queue, meta=metadata)
        
        # Check that remember was called with metadata including our custom data
        call_args = self.mock_synapse.remember.call_args
        called_metadata = call_args.kwargs.get('metadata', {})
        assert called_metadata["test_key"] == "test_value"
        assert "router_category" in called_metadata
        assert "router_confidence" in called_metadata
        
    def test_no_valid_signals(self):
        """Test handling when no classifiers find storable content."""
        # Use text that shouldn't match any patterns
        result = ingest("The sky.", self.mock_synapse, self.mock_review_queue)
        assert result in (IngestResult.IGNORED_FLUFF, IngestResult.QUEUED_FOR_REVIEW)
        assert not self.mock_synapse.remember.called


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_preference_to_storage_flow(self):
        """Test complete flow from preference detection to storage."""
        mock_synapse = Mock()
        
        result = ingest("I really love using React for frontend development", mock_synapse, None, policy="auto")
        
        assert result == IngestResult.STORED
        assert mock_synapse.remember.called
        
        # Check the call arguments
        call_args = mock_synapse.remember.call_args
        assert call_args.args[0] == "I really love using React for frontend development"
        assert call_args.kwargs["memory_type"] == "preference"
        
    def test_fact_to_storage_flow(self):
        """Test complete flow from fact detection to storage."""
        mock_synapse = Mock()
        
        result = ingest("Alice Johnson is the new project manager", mock_synapse, None, policy="auto")
        
        assert result == IngestResult.STORED
        assert mock_synapse.remember.called
        
        call_args = mock_synapse.remember.call_args
        assert call_args.kwargs["memory_type"] == "fact"
        
    def test_decision_to_review_flow(self):
        """Test flow from decision detection to review queue."""
        mock_synapse = Mock()
        mock_review_queue = Mock()
        
        result = ingest("Let's maybe try Vue for the next project", mock_synapse, mock_review_queue, policy="auto")
        
        # This might go to review depending on confidence
        if result == IngestResult.QUEUED_FOR_REVIEW:
            assert mock_review_queue.submit.called
        elif result == IngestResult.STORED:
            assert mock_synapse.remember.called
            
    def test_multiple_signal_types(self):
        """Test text that triggers multiple classifiers."""
        mock_synapse = Mock()
        
        # Text that could match both decision and preference patterns
        result = ingest("I decided I love using Python because it's so readable", mock_synapse, None, policy="auto")
        
        assert result == IngestResult.STORED
        assert mock_synapse.remember.called
        
        # The router should pick the highest confidence signal
        call_args = mock_synapse.remember.call_args
        metadata = call_args.kwargs.get('metadata', {})
        assert 'router_category' in metadata
        assert 'router_confidence' in metadata
        assert metadata['router_confidence'] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])