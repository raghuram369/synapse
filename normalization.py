"""Zero-LLM entity normalization utilities.

This module centralizes simple lexical normalization for entities and triples.
It intentionally avoids any NLP runtime dependency and keeps behavior
predictable and deterministic.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


_DEFAULT_ALIAS_MAP: Dict[str, str] = {
    "ny": "new york",
    "nyc": "new york",
    "sf": "san francisco",
    "la": "los angeles",
    "usa": "united states",
    "u.s.": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "u.s.a.": "united states",
    "new york": "new york",
    "new york city": "new york city",
}


class EntityNormalizer:
    """Normalize raw entities and resolve aliases without external NLP models."""

    _ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
    _WHITESPACE_RE = re.compile(r"\s+")
    _NON_ALPHA_SUFFIX_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9']+$")
    _COREF_PROPER_NAME_RE = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9']*(?:\s+[A-Z][A-Za-z0-9']*){0,4})\b"
    )

    _PERSON_TOKENS = {
        "he", "she", "they", "i", "me", "you", "we", "us", "her", "him", "his", "hers",
    }

    _NON_PERSON_KEYWORDS = {
        "city",
        "country",
        "project",
        "software",
        "system",
        "memory",
        "database",
        "tool",
        "company",
        "platform",
        "model",
        "service",
        "app",
        "device",
        "machine",
        "server",
        "synapse",
        "node",
        "agent",
        "product",
        "team",
    }

    _NOUN_IRREGULARS: Dict[str, str] = {
        "children": "child",
    }

    _VERB_IRREGULARS: Dict[str, str] = {
        "liked": "like",
        "liked.": "like",
        "goes": "go",
        "went": "go",
    }

    def __init__(self, alias_map: Optional[Dict[str, str]] = None):
        self.alias_map: Dict[str, str] = dict(_DEFAULT_ALIAS_MAP)
        if alias_map:
            for alias, canonical in alias_map.items():
                self.register_alias(alias, canonical)

        self.user_name: Optional[str] = None

    @staticmethod
    def _normalize_alias(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower()).strip()

    @staticmethod
    def _normalize_token(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).strip()

    @staticmethod
    def _strip_article(text: str) -> str:
        return EntityNormalizer._ARTICLE_RE.sub("", text, count=1).strip()

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register a user alias that resolves to a canonical string."""
        if not alias:
            return
        if canonical is None:
            canonical = ""

        alias_key = self._normalize_alias(str(alias))
        canonical_value = self._normalize_token(str(canonical))
        self.alias_map[alias_key] = canonical_value

        stripped_alias = self._strip_article(alias_key)
        if stripped_alias and stripped_alias != alias_key:
            self.alias_map[stripped_alias] = canonical_value

        alias_min = alias_key.lower()
        if alias_min in {"the user", "user", "the customer", "customer"}:
            self.user_name = canonical_value

    def lemmatize(self, word: str, keep_proper_nouns: bool = True) -> str:
        """Reduce a single token to a simple base form.

        Keeps title-case words as-is so simple proper nouns remain capitalized.
        """
        token = self._normalize_token(word)
        if not token:
            return ""

        if keep_proper_nouns and self._is_proper_noun(token):
            return token.strip()

        lower = token.lower()
        lower = self._NON_ALPHA_SUFFIX_RE.sub("", lower)
        if not lower:
            return ""

        if lower in self._NOUN_IRREGULARS:
            return self._NOUN_IRREGULARS[lower]
        if lower in self._VERB_IRREGULARS:
            return self._VERB_IRREGULARS[lower]

        # Keep lemmatization conservative: only apply -ing stemming when we see
        # the common doubled-consonant pattern (e.g. runn+ing -> run).
        if lower.endswith("ing") and len(lower) > 4:
            stem = lower[:-3]
            if len(stem) > 1 and stem[-1] == stem[-2]:
                return stem[:-1]
            return lower

        if lower.endswith("ed") and len(lower) > 3:
            stem = lower[:-2]
            if len(stem) > 1 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem

        if lower.endswith("ies") and len(lower) > 3:
            return lower[:-3] + "y"

        if lower.endswith("es") and len(lower) > 2:
            return lower[:-2]

        if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
            return lower[:-1]

        return lower

    def canonical(self, text: str, keep_proper_nouns: bool = False) -> str:
        """Normalize text and resolve aliases into a canonical entity label."""
        cleaned = self._normalize_token(text)
        if not cleaned:
            return ""

        cleaned = cleaned.strip('"\'`')
        if not cleaned:
            return ""

        no_articles = self._strip_article(cleaned)
        no_articles = self._normalize_token(no_articles)

        alias_key = self._normalize_alias(no_articles)
        alias_value = self.alias_map.get(alias_key)
        if alias_value:
            return alias_value

        if no_articles.endswith("'s"):
            no_articles = self._normalize_token(no_articles[:-2])

        if not no_articles:
            return ""

        normalized_tokens: List[str] = []
        for raw_token in re.split(r"\s+", no_articles):
            token = self._NON_ALPHA_SUFFIX_RE.sub("", raw_token)
            if not token:
                continue
            normalized_tokens.append(
                self.lemmatize(token, keep_proper_nouns=keep_proper_nouns)
            )

        normalized = " ".join(normalized_tokens)
        return self._normalize_token(normalized)

    def coref_resolve(self, texts: List[str]) -> Dict[str, str]:
        """Resolve a few hardcoded pronouns using local mention history."""
        if not texts:
            return {}

        resolved: Dict[str, str] = {}
        recent_named_entities: List[tuple[str, bool]] = []  # (name, is_person)

        for text in texts:
            if not text:
                continue
            lowered = text.lower()

            if self.user_name:
                if "the user" in lowered:
                    resolved["the user"] = self.user_name
                if "the customer" in lowered:
                    resolved["the customer"] = self.user_name

            for entity_match in self._COREF_PROPER_NAME_RE.finditer(text):
                raw_entity = entity_match.group(0).strip()
                if not raw_entity:
                    continue
                if raw_entity.strip().lower() in {"he", "she", "they", "it", "i", "we", "you"}:
                    continue
                # Preserve case for pronoun resolution targets (e.g. "Alice"),
                # but keep canonical() default lowercased for general indexing.
                canonical_entity = self.canonical(raw_entity, keep_proper_nouns=True)
                if not canonical_entity:
                    continue
                is_person = self._looks_like_person(raw_entity, canonical_entity)
                recent_named_entities.append((canonical_entity, is_person))

            if re.search(r"\b(he|she|they)\b", lowered):
                person = self._latest_entity(recent_named_entities, prefer_person=True)
                if person:
                    for pronoun in ("he", "she", "they"):
                        if re.search(rf"\b{pronoun}\b", lowered):
                            resolved[pronoun] = person

            if re.search(r"\bit\b", lowered):
                obj = self._latest_entity(recent_named_entities, prefer_person=False)
                if obj:
                    resolved["it"] = obj

        return resolved

    @staticmethod
    def _looks_like_person(entity: str, canonical: str) -> bool:
        normalized = canonical.lower()

        # If the canonical value is fully lowercased and multi-token (usually
        # via alias/canonicalization), treat it as a non-person (e.g. "new york").
        if canonical and canonical == normalized and " " in canonical:
            return False

        if any(keyword in normalized.split() for keyword in EntityNormalizer._NON_PERSON_KEYWORDS):
            return False

        if normalized in EntityNormalizer._PERSON_TOKENS:
            return True

        if any(token in EntityNormalizer._PERSON_TOKENS for token in normalized.split()):
            return False

        return bool(
            EntityNormalizer._is_proper_noun(entity)
            and normalized.split()[0] not in {"the", "a", "an"}
            and len(normalized) > 1
        )

    @staticmethod
    def _latest_entity(entities: List[tuple[str, bool]], prefer_person: bool) -> Optional[str]:
        if prefer_person:
            for name, is_person in reversed(entities):
                if is_person:
                    return name
        else:
            for name, is_person in reversed(entities):
                if not is_person:
                    return name
        return None

    @staticmethod
    def _is_proper_noun(token: str) -> bool:
        cleaned = token.strip("\"'`,;:.!?()[]{}")
        return bool(re.match(r"[A-Z][a-z]", cleaned))


# Shared singleton used across entity_graph, triples, and Synapse
ENTITY_NORMALIZER = EntityNormalizer()
