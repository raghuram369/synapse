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
