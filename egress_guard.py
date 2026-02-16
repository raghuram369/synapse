"""PII egress filtering utilities for MCP read paths."""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple


PatternSpec = Tuple[Pattern[str], str]


class EgressGuard:
    """Redact common PII from outgoing text payloads."""

    _STANDARD_PATTERNS: List[PatternSpec] = [
        (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "SSN"),
        (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "CREDIT_CARD"),
        (
            re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
            "EMAIL",
        ),
        (
            re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
            "PHONE",
        ),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP"),
        (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"), "IP"),
    ]

    _STRICT_PATTERNS: List[PatternSpec] = [
        (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "SSN"),
        (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "CREDIT_CARD"),
        (
            re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
            "EMAIL",
        ),
        (
            re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
            "PHONE",
        ),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP"),
        (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"), "IP"),
        (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "IBAN"),
        (re.compile(r"\b\d{10,}\b"), "NUMERIC_ID"),
    ]

    def __init__(self, sensitivity: str = "standard"):
        normalized = str(sensitivity or "standard").strip().lower()
        if normalized not in {"standard", "strict"}:
            raise ValueError("sensitivity must be one of: standard, strict")
        self.sensitivity = normalized
        self._patterns = (
            self._STRICT_PATTERNS if self.sensitivity == "strict" else self._STANDARD_PATTERNS
        )

    def filter_context(self, text: str) -> str:
        """Redact PII patterns from text and return cleaned output."""
        if not isinstance(text, str) or not text:
            return "" if text is None else str(text)

        cleaned = text
        for pattern, pii_type in self._patterns:
            cleaned = pattern.sub(f"[REDACTED_{pii_type}]", cleaned)
        return cleaned


class SensitiveContentDetector:
    """Heuristic detector for contextually sensitive content."""

    _HEALTH_PATTERNS: List[Pattern[str]] = [
        re.compile(r"\b(?:diagnosed|diagnosis|treatment|medication|prescription|therapy)\b", re.I),
        re.compile(r"\b(?:anxiety|depression|ptsd|diabetes|cancer|asthma|adhd|autism)\b", re.I),
        re.compile(r"\b(?:hospital|clinic|surgeon|psychiatrist|oncologist|medical record)\b", re.I),
    ]
    _SCHOOL_PATTERNS: List[Pattern[str]] = [
        re.compile(r"\b(?:elementary|middle|high)\s+school\b", re.I),
        re.compile(r"\b(?:kindergarten|daycare|preschool)\b", re.I),
        re.compile(r"\b(?:my|our)\s+(?:kid|kids|child|children|daughter|son)\b.*\b(?:school|class|teacher)\b", re.I),
    ]
    _ADDRESS_PATTERNS: List[Pattern[str]] = [
        re.compile(
            r"\b\d{1,6}[A-Za-z]?\s+[A-Za-z0-9.'-]+\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct|place|pl|way|circle|cir)\b",
            re.I,
        ),
        re.compile(r"\b(?:my|our)\s+(?:home|house|apartment|apt)\s+(?:address|is)\b", re.I),
    ]
    _RELATIONSHIP_PATTERNS: List[Pattern[str]] = [
        re.compile(r"\b(?:my|our)\s+(?:wife|husband|partner|girlfriend|boyfriend|ex|spouse)\b", re.I),
        re.compile(r"\b(?:my|our)\s+(?:kid|kids|child|children|daughter|son)\b", re.I),
        re.compile(r"\b(?:custody|divorce|separation|pregnant|pregnancy|fertility)\b", re.I),
    ]

    @classmethod
    def detect(cls, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        for patterns in (
            cls._HEALTH_PATTERNS,
            cls._SCHOOL_PATTERNS,
            cls._ADDRESS_PATTERNS,
            cls._RELATIONSHIP_PATTERNS,
        ):
            for pattern in patterns:
                if pattern.search(text):
                    return True
        return False
