"""Policy presets and lightweight PII detection/ redaction utilities."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set


PRESETS: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "description": "Only preferences + commitments. Aggressive pruning.",
        "ttl_days": 30,
        "auto_prune": True,
        "prune_min_access": 2,
        "redact_pii": True,
        "keep_tags": ["preference", "commitment", "important"],
    },
    "private": {
        "description": "Redact PII patterns aggressively. Medium retention.",
        "ttl_days": 90,
        "auto_prune": True,
        "prune_min_access": 0,
        "redact_pii": True,
        "pii_patterns": ["email", "phone", "ssn", "address", "credit_card"],
    },
    "work": {
        "description": "Project-centric. Longer retention. Tag by project.",
        "ttl_days": 365,
        "auto_prune": True,
        "prune_min_access": 0,
        "redact_pii": False,
        "auto_tag_project": True,
    },
    "ephemeral": {
        "description": "TTL on everything unless explicitly pinned.",
        "ttl_days": 7,
        "auto_prune": True,
        "prune_min_access": 0,
        "redact_pii": False,
        "pin_tag": "pinned",
    },
}


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_CC_RE = re.compile(
    r"(?<![\d-])(?:\d{4}[ -]?){3}\d{4}(?![\d-])|(?<![\d-])\d{4}[ -]?\d{6}[ -]?\d{5}(?![\d-])"
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}[A-Za-z]?\s+[A-Za-z0-9.'-]+\s+(?:"
    r"street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct|place|pl|way|circle|cir)\b",
    re.I,
)
_PROJECT_TAG_RE = re.compile(r"(?:#|project[:\s-]{0,4})([A-Za-z][A-Za-z0-9_-]{1,})", re.I)
_SUPPORTED_PII_TYPES: Set[str] = {
    "email",
    "phone",
    "ssn",
    "address",
    "credit_card",
}


class PolicyManager:
    def __init__(self, synapse):
        self.synapse = synapse
        self.active_policy: Optional[Dict[str, Any]] = None

    def apply(self, preset: str) -> Dict[str, Any]:
        """Apply a preset policy. Configures forgetting rules, PII detection, TTL."""
        key = str(preset).strip().lower() if preset is not None else ""
        if not key:
            raise ValueError("preset must be a non-empty string")

        template = PRESETS.get(key)
        if template is None:
            raise ValueError(f"Unknown policy preset: {preset!r}")

        config: Dict[str, Any] = deepcopy(template)
        config["name"] = key
        self.active_policy = config
        return deepcopy(config)

    def get_active(self) -> Optional[Dict[str, Any]]:
        return deepcopy(self.active_policy) if self.active_policy is not None else None

    def list_presets(self) -> Dict[str, Any]:
        return {name: deepcopy(config) for name, config in PRESETS.items()}

    def _coerce_pii_patterns(self, pii_patterns: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if pii_patterns is None:
            return None
        normalized: Set[str] = set()
        for value in pii_patterns:
            name = str(value).strip().lower() if value is not None else ""
            if not name:
                continue
            if name in _SUPPORTED_PII_TYPES:
                normalized.add(name)
        return normalized if normalized else None

    def detect_pii(self, text: str, pii_patterns: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Zero-dependency PII detection."""
        if not text:
            return []

        allowed_types = self._coerce_pii_patterns(pii_patterns)
        detections: List[Dict[str, Any]] = []
        seen: set[tuple] = set()

        def _add_match(pii_type: str, value: str, start: int, end: int) -> None:
            if allowed_types is not None and pii_type not in allowed_types:
                return
            key = (pii_type, start, end)
            if key in seen:
                return
            seen.add(key)
            detections.append({
                "type": pii_type,
                "value": value,
                "start": start,
                "end": end,
            })

        for match in _EMAIL_RE.finditer(text):
            _add_match("email", match.group(0), match.start(), match.end())

        for match in _PHONE_RE.finditer(text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if re.fullmatch(r"\+?\d{3}[- ]?\d{2}[- ]?\d{4}", candidate.replace("(", "").replace(")", "").strip()):
                continue
            if 7 <= len(digits) <= 15:
                _add_match("phone", candidate, match.start(), match.end())

        for match in _SSN_RE.finditer(text):
            _add_match("ssn", match.group(0), match.start(), match.end())

        for match in _CC_RE.finditer(text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if 13 <= len(digits) <= 19 and self._luhn_valid(digits):
                _add_match("credit_card", candidate, match.start(), match.end())

        for match in _ADDRESS_RE.finditer(text):
            _add_match("address", match.group(0), match.start(), match.end())

        detections.sort(key=lambda item: (item["start"], item["end"]))
        deduped: List[Dict[str, Any]] = []
        occupied: set[tuple] = set()
        for item in detections:
            span = (item["type"], item["start"], item["end"])
            if span in occupied:
                continue
            occupied.add(span)
            deduped.append(item)
        return deduped

    def auto_redact(self, text: str, pii_patterns: Optional[Iterable[str]] = None) -> str:
        """Replace detected PII with [REDACTED-TYPE]."""
        findings = self.detect_pii(text, pii_patterns=pii_patterns)
        if not findings:
            return text

        redacted = text
        for item in sorted(findings, key=lambda x: x["start"], reverse=True):
            token = f"[REDACTED-{item['type'].upper()}]"
            redacted = f"{redacted[:item['start']]}{token}{redacted[item['end']:]}"

        return redacted

    def detect_project_tags(self, text: str) -> List[str]:
        """Return normalized project tags discovered in text."""
        if not text:
            return []

        tags: set[str] = set()
        for match in _PROJECT_TAG_RE.finditer(text):
            raw = match.group(1)
            if not raw:
                continue
            tag = self._normalize_tag(raw)
            if tag:
                tags.add(f"project:{tag}")

        return sorted(tags)

    @staticmethod
    def _normalize_tag(raw: str) -> str:
        text = re.sub(r"\s+", "-", raw.strip().lower())
        text = re.sub(r"[^a-z0-9-]", "", text)
        if text.startswith("project") and len(text) > len("project"):
            text = text[len("project"):]
            text = text.lstrip("-_")
        return text

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        digits = [int(ch) for ch in number]
        if not digits:
            return False
        checksum = 0
        reverse = digits[::-1]
        for i, digit in enumerate(reverse):
            if i % 2 == 1:
                doubled = digit * 2
                checksum += doubled - 9 if doubled > 9 else doubled
            else:
                checksum += digit
        return checksum % 10 == 0
