"""Memory Router — intelligent capture pipeline for Synapse AI Memory.

Combines heuristic classifiers with policy-based routing to decide:
store? extract triples? defer to review? ignore?

Also includes original clipboard capture functionality.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse
    from review_queue import ReviewQueue


# ---------------------------------------------------------------------------
# Constants and Types
# ---------------------------------------------------------------------------

class IngestResult(Enum):
    """Results from the memory router."""
    STORED = "stored"
    QUEUED_FOR_REVIEW = "queued_for_review"
    IGNORED_FLUFF = "ignored_fluff"
    REJECTED_SECRET = "rejected_secret"
    IGNORED_POLICY = "ignored_policy"


@dataclass
class ClassificationSignal:
    """Output from a heuristic classifier."""
    should_store: bool
    confidence: float  # 0.0 to 1.0
    category: str  # fact, preference, decision, etc.
    extracted: str  # cleaned/extracted text, or original if no extraction


# ---------------------------------------------------------------------------
# Heuristic Classifiers (no LLM, milliseconds response time)
# ---------------------------------------------------------------------------

class PreferenceDetector:
    """Detects preference statements: 'I like/hate/prefer/avoid/want/need...'"""
    
    PREFERENCE_PATTERNS = [
        r"\b(?:I|we|my|our)\s+(?:really\s+)?(?:like|love|enjoy|prefer|want|need|desire)\s+",
        r"\b(?:I|we|my|our)\s+(?:really\s+)?(?:hate|dislike|avoid|can't\s+stand|despise)\s+",
        r"\b(?:I|we)\s+(?:would\s+)?(?:rather|prefer)\s+",
        r"\b(?:I|we)\s+(?:don't|do\s+not)\s+(?:like|want|need|prefer)\s+",
        r"\bmy\s+(?:favorite|favourite|preferred)\s+",
        r"\b(?:I|we)\s+(?:am|are)\s+(?:not\s+)?(?:into|interested\s+in|fond\s+of)\s+",
        r"\b(?:I|we)\s+(?:can't\s+stand|cannot\s+stand)\s+",
        r"\b(?:I|we)\s+(?:wish|hope)\s+",
        r"\b(?:I|we)\s+(?:always|usually|often|sometimes|never)\s+(?:use|choose|pick|go\s+with)\s+",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        text_lower = text.lower()
        max_confidence = 0.0
        
        for pattern in cls.PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Higher confidence for stronger preference words
                if any(word in match.group() for word in ['love', 'hate', 'despise', 'favorite', 'favourite']):
                    confidence = 0.9
                elif any(word in match.group() for word in ['really', 'always', 'never']):
                    confidence = 0.8
                else:
                    confidence = 0.7
                max_confidence = max(max_confidence, confidence)
        
        return ClassificationSignal(
            should_store=max_confidence > 0.0,
            confidence=max_confidence,
            category="preference",
            extracted=text.strip()
        )


class FactDetector:
    """Detects factual information: names, locations, jobs, tools, entities."""
    
    # Common person name patterns
    NAME_PATTERNS = [
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # John Smith
        r"\b(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+[A-Z][a-z]+\b",  # Dr. Smith
        r"\b[A-Z][a-z]+\s+is\s+(?:a|an|the)\s+",  # Alice is a developer
        r"\bmy\s+(?:name\s+is|friend|colleague|boss|manager)\s+[A-Z][a-z]+\b",
    ]
    
    # Location patterns
    LOCATION_PATTERNS = [
        r"\b(?:in|at|from|to|near|around)\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b",
        r"\b(?:San\s+Francisco|New\s+York|Los\s+Angeles|Washington\s+DC)\b",
        r"\b[A-Z][a-z]+(?:,\s*[A-Z]{2})?\s*\d{5}\b",  # City, State ZIP
        r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Boulevard|Blvd)\b",
    ]
    
    # Job/role patterns
    JOB_PATTERNS = [
        r"\b(?:I|he|she|they|we)\s+(?:work|am|is|are)\s+(?:as\s+)?(?:a|an)?\s*(?:software\s+)?(?:engineer|developer|designer|manager|director|CEO|CTO|analyst|consultant|teacher|professor|doctor|nurse|lawyer|architect|scientist|researcher|writer|journalist|artist|musician|chef|pilot|driver|mechanic|electrician|plumber|carpenter|accountant|banker|salesperson|marketer|recruiter)\b",
        r"\b(?:my|his|her|their|our)\s+(?:job|role|position|title)\s+is\s+",
        r"\b(?:works|working)\s+(?:at|for)\s+[A-Z][a-z]+",
    ]
    
    # Tool/technology patterns
    TOOL_PATTERNS = [
        r"\b(?:use|using|used|work\s+with|built\s+with|made\s+with)\s+(?:Python|JavaScript|Java|C\+\+|React|Vue|Angular|Django|Flask|Node\.js|Docker|Kubernetes|AWS|Azure|GCP|PostgreSQL|MySQL|Redis|MongoDB|Git|GitHub|VS\s+Code|Vim|Emacs|Slack|Discord|Zoom|Teams|Figma|Photoshop|Illustrator|Sketch|Notion|Obsidian|Roam)\b",
        r"\b(?:I|we|they)\s+(?:prefer|use|chose|selected|switched\s+to)\s+[A-Z][A-Za-z0-9\.\+\-\s]*\b",        
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        confidence = 0.0
        category = "fact"
        
        # Check for person names
        for pattern in cls.NAME_PATTERNS:
            if re.search(pattern, text):
                confidence = max(confidence, 0.7)
                category = "person"
        
        # Check for locations
        for pattern in cls.LOCATION_PATTERNS:
            if re.search(pattern, text):
                confidence = max(confidence, 0.8)
                category = "location"
        
        # Check for jobs/roles
        for pattern in cls.JOB_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                confidence = max(confidence, 0.8)
                category = "job"
        
        # Check for tools/technologies
        for pattern in cls.TOOL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                confidence = max(confidence, 0.7)
                category = "tool"
        
        # Generic factual patterns
        factual_indicators = [
            r"\bis\s+(?:a|an|the)\s+[a-z]+",
            r"\bhas\s+(?:a|an|the|\d+)\s+",
            r"\blives\s+in\s+",
            r"\bborn\s+in\s+",
            r"\bgraduated\s+from\s+",
            r"\bstudied\s+at\s+",
        ]
        
        for pattern in factual_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                confidence = max(confidence, 0.6)
        
        return ClassificationSignal(
            should_store=confidence > 0.0,
            confidence=confidence,
            category=category,
            extracted=text.strip()
        )


class DecisionDetector:
    """Detects decision statements: 'let's do X', 'we decided', 'ship on', 'go with'"""
    
    DECISION_PATTERNS = [
        r"\b(?:let's|lets)\s+(?:do|go\s+with|use|try|implement|build|create)\s+",
        r"\b(?:we|I)\s+(?:decided|choose|chose|picked|selected|agreed)\s+(?:to\s+)?",
        r"\b(?:going|gonna|will\s+go)\s+with\s+",
        r"\b(?:ship|deploy|launch|release)\s+(?:on|by|this|next)\s+",
        r"\b(?:final|ultimate|official)\s+decision(?::|\s+)",
        r"\b(?:settled|agreed)\s+on\s+",
        r"\bthat's\s+(?:it|settled|decided|final)\b",
        r"\b(?:we|I)\s+(?:should|will|are\s+going\s+to|plan\s+to)\s+",
        r"\b(?:action\s+item|next\s+step|todo|task):\s*",
        r"\b(?:conclusion|verdict|outcome):\s*",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        text_lower = text.lower()
        max_confidence = 0.0
        
        for pattern in cls.DECISION_PATTERNS:
            if re.search(pattern, text_lower):
                # Higher confidence for definitive decisions
                if any(word in text_lower for word in ['decided', 'final', 'settled', 'agreed', 'conclusion']):
                    confidence = 0.9
                elif any(word in text_lower for word in ['will', 'going to', 'plan to', 'ship']):
                    confidence = 0.8
                else:
                    confidence = 0.7
                max_confidence = max(max_confidence, confidence)
        
        return ClassificationSignal(
            should_store=max_confidence > 0.0,
            confidence=max_confidence,
            category="decision",
            extracted=text.strip()
        )


class PlanDetector:
    """Detects planning statements: dates, 'tomorrow', 'next week', 'by Friday'"""
    
    DATE_PATTERNS = [
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:today|tomorrow|yesterday)\b",
        r"\b(?:next|this|last)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bin\s+(?:a\s+few\s+days|a\s+week|two\s+weeks|a\s+month)\b",
        r"\bby\s+(?:end\s+of\s+)?(?:week|month|year|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
        r"\bdue\s+(?:on|by)\s+",
        r"\bdeadline\s+is\s+",
        r"\bscheduled\s+for\s+",
    ]
    
    PLANNING_INDICATORS = [
        r"\b(?:plan|planning|scheduled|scheduled|agenda|timeline|roadmap|milestone)\b",
        r"\b(?:will|gonna|going\s+to|intend\s+to|planning\s+to)\s+",
        r"\b(?:reminder|remember\s+to|don't\s+forget\s+to)\s+",
        r"\b(?:upcoming|coming\s+up|approaching)\s+",
        r"\b(?:meeting|call|appointment|event|deadline|due\s+date)\s+",
        r"\b(?:after|before|during|while)\s+",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        text_lower = text.lower()
        confidence = 0.0
        
        # Check for date/time patterns (high confidence)
        for pattern in cls.DATE_PATTERNS:
            if re.search(pattern, text_lower):
                confidence = max(confidence, 0.85)
        
        # Check for planning indicators
        for pattern in cls.PLANNING_INDICATORS:
            if re.search(pattern, text_lower):
                confidence = max(confidence, 0.6)
        
        return ClassificationSignal(
            should_store=confidence > 0.0,
            confidence=confidence,
            category="plan",
            extracted=text.strip()
        )


class CorrectionDetector:
    """Detects corrections: 'no actually', 'that's wrong', 'correction:', contradictions"""
    
    CORRECTION_PATTERNS = [
        r"\b(?:no\s+actually|actually\s+no|wait\s+no|no\s+wait)\b",
        r"\bactually\b.*\b(?:wrong|incorrect|not\s+right|false|untrue|different|instead)\b",
        r"\b(?:that's|thats|that\s+is|this\s+is)\b(?:\s+\w+){0,2}\s*(?:wrong|incorrect|not\s+right|false|untrue)\b",
        r"\b(?:correction|clarification|amendment|update):\s*",
        r"\b(?:I\s+was\s+wrong|my\s+mistake|sorry|oops|nevermind|never\s+mind)\b",
        r"\b(?:let\s+me\s+correct|to\s+correct|need\s+to\s+fix)\b",
        r"\b(?:not|isn't|aren't|wasn't|weren't|don't|doesn't|didn't)\s+",
        r"\b(?:scratch\s+that|forget\s+what\s+I\s+said|ignore\s+that)\b",
        r"\b(?:meant\s+to\s+say|should\s+have\s+said|what\s+I\s+meant\s+was)\b",
        r"\b(?:on\s+second\s+thought|thinking\s+about\s+it)\b",
        r"\b(?:revised|updated|changed\s+my\s+mind)\b",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        text_lower = text.lower()
        max_confidence = 0.0
        
        for pattern in cls.CORRECTION_PATTERNS:
            if re.search(pattern, text_lower):
                # Higher confidence for explicit corrections
                if any(word in text_lower for word in ['correction:', 'wrong', 'mistake', 'clarification:']):
                    confidence = 0.95
                elif any(word in text_lower for word in ['actually', 'scratch that', 'nevermind']):
                    confidence = 0.8
                else:
                    confidence = 0.6
                max_confidence = max(max_confidence, confidence)
        
        return ClassificationSignal(
            should_store=max_confidence > 0.0,
            confidence=max_confidence,
            category="correction",
            extracted=text.strip()
        )


class OutcomeDetector:
    """Detects outcome statements: 'that worked', 'it failed', 'shipped', 'published', 'broke'"""
    
    OUTCOME_PATTERNS = [
        r"\b(?:that|it|the\s+\w+)\s+(?:worked|failed|succeeded|broke|crashed|died)\b",
        r"\b\w+\s+failed\b",
        r"\b(?:successfully|failed\s+to|managed\s+to|unable\s+to)\s+",
        r"\b(?:shipped|deployed|launched|released|published|went\s+live)\b",
        r"\b(?:completed|finished|done|ready|delivered|complete)\b",        
        r"\b(?:is|was|are)\s+complete\b",
        r"\b(?:bug|issue|problem|error)\s+(?:fixed|resolved|solved|closed)\b",
        r"\b(?:test|tests)\s+(?:passed|failed|passing|failing)\b",
        r"\b(?:build|deployment|release)\s+(?:succeeded|failed|passed|broken)\b",
        r"\b(?:performance|speed|load\s+time)\s+(?:improved|degraded|worse|better)\b",
        r"\b(?:metrics|numbers|stats)\s+(?:are\s+)?(?:up|down|good|bad|improved|worse)\b",
        r"\b(?:outage|downtime|incident|crash|failure)\b",
        r"\b(?:rollback|reverted|rolled\s+back)\b",
        r"\b(?:success|failure|win|loss|victory|defeat)\b",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ClassificationSignal:
        text_lower = text.lower()
        max_confidence = 0.0
        
        for pattern in cls.OUTCOME_PATTERNS:
            if re.search(pattern, text_lower):
                # Higher confidence for definitive outcomes
                if any(word in text_lower for word in ['shipped', 'failed', 'crashed', 'succeeded', 'completed']):
                    confidence = 0.9
                elif any(word in text_lower for word in ['worked', 'broke', 'fixed', 'delivered']):
                    confidence = 0.8
                else:
                    confidence = 0.6
                max_confidence = max(max_confidence, confidence)
        
        return ClassificationSignal(
            should_store=max_confidence > 0.0,
            confidence=max_confidence,
            category="outcome",
            extracted=text.strip()
        )


class SecretFilter:
    """Detects and filters sensitive information: API keys, tokens, passwords, SSNs, credit cards"""
    
    SECRET_PATTERNS = [
        # API keys and tokens
        r"\b[A-Za-z0-9]{32,}\b",  # Generic long alphanumeric strings
        r"\bsk-[a-zA-Z0-9]{10,}\b",  # OpenAI API keys
        r"\bghp_[a-zA-Z0-9]{30,}\b",  # GitHub personal access tokens
        r"\bgho_[a-zA-Z0-9]{30,}\b",  # GitHub OAuth tokens
        r"\bghs_[a-zA-Z0-9]{30,}\b",  # GitHub server tokens
        r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key IDs
        r"\b[A-Za-z0-9/+=]{40}\b",  # AWS secret access keys
        r"\bAIza[0-9A-Za-z_-]{35}\b",  # Google API keys
        r"\b[0-9]+-[0-9A-Za-z_-]{32}\\.apps\\.googleusercontent\\.com\b",  # Google OAuth client IDs
        
        # Passwords (contextual)
        r"\bpassword\s*[=:]\s*[^\s]+",
        r"\bpwd\s*[=:]\s*[^\s]+", 
        r"\bpasswd\s*[=:]\s*[^\s]+",
        r"\bpass\s*[=:]\s*[^\s]+",
        
        # Social Security Numbers
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{3}\s+\d{2}\s+\d{4}\b",
        r"\b\d{9}\b",  # 9 consecutive digits (loose SSN)
        
        # Credit card numbers
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # 16 digit card numbers
        r"\b\d{13,19}\b",  # Credit card range
        
        # Private keys
        r"-----BEGIN\s+(?:PRIVATE\s+KEY|RSA\s+PRIVATE\s+KEY|ENCRYPTED\s+PRIVATE\s+KEY)-----",
        r"-----END\s+(?:PRIVATE\s+KEY|RSA\s+PRIVATE\s+KEY|ENCRYPTED\s+PRIVATE\s+KEY)-----",
        
        # Email/phone in password contexts
        r"(?:email|phone|mobile|cell).*(?:password|pin|code)",
        
        # URLs with embedded credentials
        r"https?://[^:\s]+:[^@\s]+@[^\s]+",
    ]
    
    @classmethod
    def detect(cls, text: str) -> bool:
        """Returns True if text contains secrets that should NEVER be stored."""
        for pattern in cls.SECRET_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class FluffFilter:
    """Detects conversational fluff that should be ignored: 'ok', 'thanks', 'cool', 'sounds good'"""
    
    FLUFF_PATTERNS = [
        # Simple acknowledgments
        r"^(?:ok|okay|k|kk|alright|right|got\s+it|gotcha|sure|yep|yup|yeah|yes|no|nope)\.?$",
        
        # Thanks and politeness
        r"^(?:thanks?|thank\s+you|thx|ty|much\s+appreciated|appreciated)\.?$",
        r"^(?:please|pls)\.?$",
        r"^(?:sorry|sry|my\s+bad|oops|whoops)\.?$",
        
        # Reactions
        r"^(?:cool|nice|great|awesome|sweet|neat|wow|amazing|interesting)\.?$",
        r"^(?:good|bad|eh|meh|hmm|hmmm|uhh|umm|ummm)\.?$",
        r"^(?:lol|lmao|haha|hehe|lulz|rofl|lmfao)\.?$",
        r"^(?:😂|😅|😊|👍|👎|🙂|🙃|😄|😆|🤣|😬|🤔|🤷|🤷‍♂️|🤷‍♀️)$",
        
        # Agreement/disagreement
        r"^(?:sounds\s+good|looks\s+good|makes\s+sense|fair\s+enough|works\s+for\s+me|i\s+agree|agreed|exactly|absolutely|totally|definitely|for\s+sure|of\s+course|indeed|true|false)\.?$",
        
        # Conversation management
        r"^(?:anyway|anyways|so|well|btw|by\s+the\s+way|oh\s+well|whatever|nevermind|never\s+mind|moving\s+on)\.?$",
        r"^(?:hello|hi|hey|bye|goodbye|see\s+ya|see\s+you|later|cya|ttyl|brb|be\s+right\s+back)\.?$",
        
        # Combinations of fluff words
        r"^(?:ok|okay|cool|nice|great|yeah|yep|sure|sounds\s+good|thanks?|lol|haha)\s+(?:ok|okay|cool|nice|great|yeah|yep|sure|sounds\s+good|thanks?|lol|haha|bro|man|dude)\.?$",
        
        # Single words/chars
        r"^[.!?]+$",  # Just punctuation
        r"^\w{1,2}\.?$",  # Single letters or very short
    ]
    
    @classmethod
    def detect(cls, text: str) -> bool:
        """Returns True if text is conversational fluff that should be ignored."""
        text_clean = text.strip().lower()
        if not text_clean:
            return True
            
        for pattern in cls.FLUFF_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        return False


# ---------------------------------------------------------------------------
# Memory Router Core
# ---------------------------------------------------------------------------

# All detector classes
DETECTORS = [
    PreferenceDetector,
    FactDetector,
    DecisionDetector,
    PlanDetector,
    CorrectionDetector,
    OutcomeDetector,
]

_CATEGORY_TO_MEMORY_TYPE = {
    "preference": "preference",
    "fact": "fact",
    "person": "fact",
    "location": "fact",
    "job": "fact",
    "tool": "skill",
    "decision": "event",
    "plan": "event",
    "correction": "observation",
    "outcome": "observation",
}


def _to_memory_type(category: str) -> str:
    return _CATEGORY_TO_MEMORY_TYPE.get(category, "fact")


def ingest(
    text: str, 
    synapse: Optional["Synapse"] = None,
    review_queue: Optional["ReviewQueue"] = None,
    source: Optional[str] = None, 
    meta: Optional[Dict[str, Any]] = None, 
    policy: str = "auto"
) -> IngestResult:
    """
    Memory Router core function.
    
    Runs heuristic classifiers and routes based on policy:
    - "auto": store high-confidence (>0.6), review medium-confidence 
    - "minimal": only store very high-confidence (>0.8)
    - "review": send everything to review queue
    - "off": ignore everything
    
    Args:
        text: Input text to classify and potentially store
        synapse: Synapse instance for storing memories
        review_queue: ReviewQueue instance for deferred review
        source: Source identifier (e.g., "clipboard", "stdin", "chat")
        meta: Additional metadata to attach
        policy: Routing policy ("auto", "minimal", "review", "off")
    
    Returns:
        IngestResult indicating what happened to the text
    """
    
    # Security filters first (never store secrets)
    if SecretFilter.detect(text):
        return IngestResult.REJECTED_SECRET
    
    # Ignore conversational fluff
    if FluffFilter.detect(text):
        return IngestResult.IGNORED_FLUFF
    
    # Policy: off -> ignore everything
    if policy == "off":
        return IngestResult.IGNORED_POLICY
    
    # Run all heuristic classifiers
    signals = [detector.classify(text) for detector in DETECTORS]

    # Pick best signal (even low confidence), fallback to generic fact signal.
    best = max(signals, key=lambda s: s.confidence)
    if best.confidence <= 0.0:
        best = ClassificationSignal(
            should_store=False,
            confidence=0.0,
            category="fact",
            extracted=text.strip(),
        )

    # Route based on policy and confidence
    if policy == "minimal" and best.confidence < 0.8:
        return IngestResult.IGNORED_POLICY

    if policy == "review" or (policy == "auto" and best.confidence < 0.6):
        # Send to review queue
        if review_queue:
            metadata = meta or {}
            metadata.update({
                "source": source or "unknown",
                "router_category": best.category,
                "router_confidence": best.confidence,
                "router_signals": [{"category": s.category, "confidence": s.confidence} for s in signals if s.should_store],
            })
            review_queue.submit(best.extracted, metadata)
            return IngestResult.QUEUED_FOR_REVIEW
        else:
            # No review queue available for review path.
            return IngestResult.IGNORED_POLICY
    
    # Auto-store with high confidence
    if synapse:
        metadata = meta or {}
        metadata.update({
            "source": source or "unknown", 
            "router_category": best.category,
            "router_confidence": best.confidence,
            "router_auto_stored": True,
        })
        synapse.remember(
            best.extracted,
            memory_type=_to_memory_type(best.category),
            metadata=metadata,
        )
        return IngestResult.STORED
    else:
        # No synapse instance provided
        return IngestResult.IGNORED_POLICY


# ---------------------------------------------------------------------------
# Original Clipboard Capture Functionality (preserved)
# ---------------------------------------------------------------------------

def _get_clipboard() -> str:
    """Read clipboard text, platform-aware."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
            return result.stdout
        elif system == "Linux":
            # Try xclip first, then xsel
            for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            result = subprocess.run(["powershell", "-command", "Get-Clipboard"], capture_output=True, text=True, timeout=3)
            return result.stdout
    except Exception:
        pass
    return ""


def clip_text(synapse: "Synapse", text: str, tags: Optional[List[str]] = None, source: str = "clip", policy: str = "auto", review_queue: Optional["ReviewQueue"] = None) -> IngestResult:
    """Remember arbitrary text with optional tags using the memory router."""
    metadata: Dict[str, Any] = {"source": source}
    if tags:
        metadata["tags"] = tags
    
    result = ingest(text, synapse=synapse, review_queue=review_queue, source=source, meta=metadata, policy=policy)
    return result


def clip_stdin(synapse: "Synapse", tags: Optional[List[str]] = None, policy: str = "auto", review_queue: Optional["ReviewQueue"] = None) -> Optional[IngestResult]:
    """Read stdin and process with memory router."""
    text = sys.stdin.read().strip()
    if not text:
        print("⚠️  No input received from stdin")
        return None
        
    result = clip_text(synapse, text, tags=tags, source="stdin", policy=policy, review_queue=review_queue)
    
    # Print result feedback
    if result == IngestResult.STORED:
        print(f"✅ Stored: {text[:80]}...")
    elif result == IngestResult.QUEUED_FOR_REVIEW:
        print(f"📋 Queued for review: {text[:80]}...")
    elif result == IngestResult.IGNORED_FLUFF:
        print(f"🙄 Ignored (fluff): {text[:80]}...")
    elif result == IngestResult.REJECTED_SECRET:
        print(f"🔒 Rejected (contains secrets): {text[:20]}...")
    elif result == IngestResult.IGNORED_POLICY:
        print(f"⏸️  Ignored (policy): {text[:80]}...")
        
    return result


def clipboard_watch(
    synapse: "Synapse", 
    interval: float = 2.0, 
    tags: Optional[List[str]] = None,
    policy: str = "auto",
    review_queue: Optional["ReviewQueue"] = None
) -> None:
    """Poll clipboard every N seconds, process new content with memory router. Runs until Ctrl+C."""
    seen = set()
    # Seed with current clipboard to avoid capturing pre-existing content
    current = _get_clipboard().strip()
    if current:
        seen.add(current)

    print(f"👀 Watching clipboard (every {interval}s, policy={policy}). Press Ctrl+C to stop.")
    stored_count = 0
    queued_count = 0
    ignored_count = 0
    
    try:
        while True:
            time.sleep(interval)
            content = _get_clipboard().strip()
            if not content or content in seen:
                continue
                
            seen.add(content)
            result = clip_text(synapse, content, tags=tags, source="clipboard", policy=policy, review_queue=review_queue)
            
            preview = content[:80].replace("\n", " ")
            if result == IngestResult.STORED:
                stored_count += 1
                print(f"✅ #{stored_count} Stored: {preview}")
            elif result == IngestResult.QUEUED_FOR_REVIEW:
                queued_count += 1
                print(f"📋 #{queued_count} Queued: {preview}")
            elif result == IngestResult.IGNORED_FLUFF:
                ignored_count += 1
                print(f"🙄 #{ignored_count} Ignored: {preview}")
            elif result == IngestResult.REJECTED_SECRET:
                print(f"🔒 REJECTED (secret): {preview[:20]}...")
            elif result == IngestResult.IGNORED_POLICY:
                ignored_count += 1
                print(f"⏸️  #{ignored_count} Policy: {preview}")
                
    except KeyboardInterrupt:
        print(f"\n🛑 Clipboard watch stopped.")
        print(f"📊 Results: {stored_count} stored, {queued_count} queued, {ignored_count} ignored")