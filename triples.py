"""Zero-LLM structured triple extraction and triple indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from normalization import ENTITY_NORMALIZER


@dataclass
class Triple:
    """Normalized fact-like triple with lightweight linguistic metadata."""

    subject: str
    predicate: str
    object: str
    polarity: str  # positive | negative
    tense: str  # past | present | future
    confidence: float  # 0.0 - 1.0
    source_span: Tuple[int, int]

ENTITY_ALIAS_MAP = ENTITY_NORMALIZER.alias_map
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|doesn't|doesn’t|didn't|didn’t|can't|can’t|won't|won’t|isn't|isn’t|aren't|aren’t|wasn't|wasn’t|weren't|weren’t)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:maybe|might|possibly|perhaps|probably|i think|i guess)\b",
    re.IGNORECASE,
)

_NOUN_PHRASE_RE = r"[A-Za-z0-9][A-Za-z0-9'&-]*(?:\s+[A-Za-z0-9'&-]+){0,10}"
_SEGMENT_RE = re.compile(r"[^.!?;]+[.!?;]?", re.MULTILINE)

_CHANGED_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})\s+changed\s+from\s+(?P<old>[^,.;!?]+?)\s+to\s+(?P<new>[^,.;!?]+)",
    re.IGNORECASE,
)
_POSS_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})'s\s+(?P<predicate>{_NOUN_PHRASE_RE})\s+(?:(?:is|was|are|were)\s+)?(?P<tail>[^,.;!?]+)",
    re.IGNORECASE,
)
_COUPULA_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})\s+(?P<copula>is|are|was|were|isn't|aren't|wasn't|weren't)\s+(?P<tail>[^,.;!?]+)",
    re.IGNORECASE,
)
_LIKES_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})\s+(?:(?:did|does|do|is|are|was|were|will|going to)\s+)?(?:(?:not|never|no longer|doesn't|doesn’t|didn't|didn’t|isn't|isn’t|aren't|aren’t|wasn't|wasn’t|weren't|weren’t|can't|can’t|won't|won’t)\s+)?(?P<verb>like|likes|prefer|prefers|want|wants)\s+(?P<tail>[^,.;!?]+)",
    re.IGNORECASE,
)
_MOVED_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})\s+(?:(?:did|does|do|is|are|was|were|will|going to)\s+)?(?:(?:not|never|no longer|doesn't|doesn’t|didn't|didn’t|isn't|isn’t|aren't|aren’t|wasn't|wasn’t|weren't|weren’t|can't|can’t|won't|won’t)\s+)?(?P<verb>moved|move)\s+to\s+(?P<tail>[^,.;!?]+)",
    re.IGNORECASE,
)
_WORKS_RE = re.compile(
    rf"(?P<subject>{_NOUN_PHRASE_RE})\s+(?:(?:did|does|do|is|are|was|were|will|going to)\s+)?(?:(?:not|never|no longer|doesn't|doesn’t|didn't|didn’t|isn't|isn’t|aren't|aren’t|wasn't|wasn’t|weren't|weren’t|can't|can’t|won't|won’t)\s+)?(?P<verb>works|work)\s+at\s+(?P<tail>[^,.;!?]+)",
    re.IGNORECASE,
)


def normalize_entity(text: str) -> str:
    """Normalize a raw entity mention for indexing and matching."""
    return ENTITY_NORMALIZER.canonical(text, keep_proper_nouns=False)


def _strip_tail(raw: str) -> str:
    raw = raw.strip(" \"'`")
    raw = re.split(r"\b(?:and|or)\b", raw, maxsplit=1)[0]
    raw = re.split(r"[.,;!?]", raw, maxsplit=1)[0]
    return raw.strip()


def _contains_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _contains_hedge(text: str) -> bool:
    return bool(_HEDGE_RE.search(text))

def _strip_leading_hedge(text: str) -> str:
    """Remove common hedge markers when they prefix a subject phrase."""
    if not text:
        return ""
    lowered = text.strip()
    # Handle multi-word hedges first.
    lowered = re.sub(r"^(?:i think|i guess)\s+", "", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"^(?:maybe|might|possibly|perhaps|probably)\s+", "", lowered, flags=re.IGNORECASE)
    return lowered.strip()

def _strip_trailing_auxiliaries(subject: str) -> str:
    """Trim auxiliary/negation tokens that can be accidentally captured as part of the subject."""
    if not subject:
        return ""
    tokens = [t for t in re.findall(r"[A-Za-z0-9']+", subject) if t]
    if not tokens:
        return subject.strip()

    strip_set = {
        "do", "does", "did",
        "is", "are", "was", "were",
        "will", "going", "to",
        "not", "never", "no", "longer",
        "can't", "cant", "won't", "wont",
        "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
        "wasn't", "wasnt", "weren't", "werent",
    }
    # Remove trailing auxiliary tokens repeatedly (e.g. "Alice does not" -> "Alice").
    while tokens and tokens[-1].lower() in strip_set:
        tokens.pop()
    return " ".join(tokens).strip()


def _derive_tense(fragment: str) -> str:
    lowered = fragment.lower()
    if "will" in lowered or "going to" in lowered:
        return "future"
    if "was" in lowered or "were" in lowered or lowered.strip().startswith("used to"):
        return "past"
    if re.search(r"\b\w+ed\b", lowered):
        return "past"
    return "present"


def _build_confidence(fragment: str, inferred: bool = False) -> float:
    if _contains_hedge(fragment):
        return 0.5
    if inferred:
        return 0.7
    return 1.0


def _polarity(fragment: str) -> str:
    return "negative" if _contains_negation(fragment) else "positive"


def extract_triples(text: str) -> List[Triple]:
    """Extract structured triples from free-form text without spaCy or external NLP.

    Returns triples with approximate subject-verb-object semantics and simple metadata.
    """
    triples: List[Triple] = []
    if not text:
        return triples

    for segment in _SEGMENT_RE.finditer(text):
        segment_text = segment.group(0).strip()
        if not segment_text:
            continue
        segment_start = segment.start()

        # X changed from Y to Z -> two triples, old (negative) and new (positive)
        for match in _CHANGED_RE.finditer(segment_text):
            subject = normalize_entity(match.group("subject"))
            old_value = normalize_entity(match.group("old"))
            new_value = normalize_entity(match.group("new"))
            if not subject or not old_value or not new_value:
                continue

            span = (segment_start + match.start(), segment_start + match.end())
            triples.append(
                Triple(
                    subject=subject,
                    predicate="changed_from",
                    object=old_value,
                    polarity="negative",
                    tense="past",
                    confidence=0.7,
                    source_span=span,
                )
            )
            triples.append(
                Triple(
                    subject=subject,
                    predicate="changed_to",
                    object=new_value,
                    polarity="positive",
                    tense="past",
                    confidence=0.7,
                    source_span=span,
                )
            )

        # X's Y is Z
        for match in _POSS_RE.finditer(segment_text):
            subject = normalize_entity(_strip_leading_hedge(match.group("subject")))
            predicate_raw = match.group("predicate")
            # The noun phrase regex can accidentally consume the copula ("laptop is").
            predicate_raw = re.sub(r"\s+(?:is|are|was|were)$", "", predicate_raw, flags=re.IGNORECASE).strip()
            predicate = normalize_entity(predicate_raw)
            obj = normalize_entity(_strip_tail(match.group("tail")))
            if not subject or not predicate or not obj:
                continue
            fragment = match.group(0)
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    polarity=_polarity(fragment),
                    tense=_derive_tense(fragment),
                    confidence=_build_confidence(fragment),
                    source_span=(segment_start + match.start(), segment_start + match.end()),
                )
            )

        # X is Y / X is not Y
        for match in _COUPULA_RE.finditer(segment_text):
            raw_subject = match.group("subject")
            if "'s" in raw_subject:
                continue
            subject = normalize_entity(_strip_leading_hedge(raw_subject))
            predicate = "is"
            obj = normalize_entity(_strip_tail(match.group("tail")))
            if not subject or not obj:
                continue
            fragment = match.group(0)
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    polarity=_polarity(fragment),
                    tense=_derive_tense(fragment),
                    confidence=_build_confidence(fragment),
                    source_span=(segment_start + match.start(), segment_start + match.end()),
                )
            )

        # X likes/prefers/wants Y
        for match in _LIKES_RE.finditer(segment_text):
            raw_subject = _strip_trailing_auxiliaries(match.group("subject"))
            subject = normalize_entity(_strip_leading_hedge(raw_subject))
            obj = normalize_entity(_strip_tail(match.group("tail")))
            if not subject or not obj:
                continue
            # Predicates should preserve their surface form ("likes", "prefers", "wants").
            verb = (match.group("verb") or "").strip().lower()
            verb = {"like": "likes", "prefer": "prefers", "want": "wants"}.get(verb, verb)
            fragment = match.group(0)
            triples.append(
                Triple(
                    subject=subject,
                    predicate=verb,
                    object=obj,
                    polarity=_polarity(fragment),
                    tense=_derive_tense(fragment),
                    confidence=_build_confidence(fragment),
                    source_span=(segment_start + match.start(), segment_start + match.end()),
                )
            )

        # X moved to Y
        for match in _MOVED_RE.finditer(segment_text):
            raw_subject = _strip_trailing_auxiliaries(match.group("subject"))
            subject = normalize_entity(_strip_leading_hedge(raw_subject))
            obj = normalize_entity(_strip_tail(match.group("tail")))
            if not subject or not obj:
                continue
            fragment = match.group(0)
            triples.append(
                Triple(
                    subject=subject,
                    predicate="moved_to",
                    object=obj,
                    polarity=_polarity(fragment),
                    tense=_derive_tense(fragment),
                    confidence=_build_confidence(fragment),
                    source_span=(segment_start + match.start(), segment_start + match.end()),
                )
            )

        # X works at Y
        for match in _WORKS_RE.finditer(segment_text):
            raw_subject = _strip_trailing_auxiliaries(match.group("subject"))
            subject = normalize_entity(_strip_leading_hedge(raw_subject))
            obj = normalize_entity(_strip_tail(match.group("tail")))
            if not subject or not obj:
                continue
            fragment = match.group(0)
            triples.append(
                Triple(
                    subject=subject,
                    predicate="works_at",
                    object=obj,
                    polarity=_polarity(fragment),
                    tense=_derive_tense(fragment),
                    confidence=_build_confidence(fragment),
                    source_span=(segment_start + match.start(), segment_start + match.end()),
                )
            )

    return triples


class TripleIndex:
    """In-memory index for triples with memory-aware storage."""

    def __init__(self):
        self.node_index: Dict[str, Set[int]] = {}
        self.edge_index: Dict[str, Set[int]] = {}
        self.inverted_triple_index: Dict[Tuple[str, str, str], Set[int]] = {}

        self._triples: Dict[int, Triple] = {}
        self._triple_to_memory: Dict[int, int] = {}
        self._memory_to_triples: Dict[int, Set[int]] = {}
        self._next_id = 0

    def add(self, memory_id: int, triples: List[Triple]) -> Set[int]:
        """Add triples tied to a memory id and update all indexes."""
        triple_ids: Set[int] = set()
        for triple in triples:
            triple_id = self._next_id
            self._next_id += 1

            self._triples[triple_id] = triple
            self._triple_to_memory[triple_id] = memory_id
            self._memory_to_triples.setdefault(memory_id, set()).add(triple_id)

            for entity in (triple.subject, triple.object):
                normalized_entity = normalize_entity(entity)
                self.node_index.setdefault(normalized_entity, set()).add(triple_id)
            self.edge_index.setdefault(normalize_entity(triple.predicate), set()).add(triple_id)
            key = (
                normalize_entity(triple.subject),
                normalize_entity(triple.predicate),
                normalize_entity(triple.object),
            )
            self.inverted_triple_index.setdefault(key, set()).add(memory_id)
            triple_ids.add(triple_id)
        return triple_ids

    def query_subject(self, subject: str) -> Set[int]:
        """Return triple IDs where normalized subject matches."""
        return set(self.node_index.get(normalize_entity(subject), set()))

    def query_predicate(self, predicate: str) -> Set[int]:
        """Return triple IDs where normalized predicate matches."""
        return set(self.edge_index.get(normalize_entity(predicate), set()))

    def query_object(self, obj: str) -> Set[int]:
        """Return triple IDs where normalized object matches."""
        return set(self.node_index.get(normalize_entity(obj), set()))

    def query_spo(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> Set[int]:
        """Return triple IDs matching the provided subject/predicate/object."""
        candidate: Set[int] | None = None

        if subject is not None:
            candidate = set(self.node_index.get(normalize_entity(subject), set()))
        if predicate is not None:
            pred_ids = self.edge_index.get(normalize_entity(predicate), set())
            candidate = pred_ids if candidate is None else candidate.intersection(pred_ids)
        if obj is not None:
            obj_ids = self.node_index.get(normalize_entity(obj), set())
            candidate = obj_ids if candidate is None else candidate.intersection(obj_ids)

        if candidate is None:
            candidate = set(self._triples.keys())
        else:
            candidate = set(candidate)

        if subject is not None:
            subject_normalized = normalize_entity(subject)
            candidate = {
                triple_id
                for triple_id in candidate
                if normalize_entity(self._triples[triple_id].subject) == subject_normalized
            }
        if predicate is not None:
            predicate_normalized = normalize_entity(predicate)
            candidate = {
                triple_id
                for triple_id in candidate
                if normalize_entity(self._triples[triple_id].predicate) == predicate_normalized
            }
        if obj is not None:
            obj_normalized = normalize_entity(obj)
            candidate = {
                triple_id
                for triple_id in candidate
                if normalize_entity(self._triples[triple_id].object) == obj_normalized
            }

        return candidate

    def get_triples_for_memory(self, memory_id: int) -> List[Triple]:
        """Return triples stored for a single memory id."""
        triple_ids = self._memory_to_triples.get(memory_id, set())
        return [self._triples[triple_id] for triple_id in sorted(triple_ids)]

    def remove_memory(self, memory_id: int) -> None:
        """Remove all triples and index entries associated with a memory id."""
        triple_ids = self._memory_to_triples.pop(memory_id, set())
        for triple_id in triple_ids:
            triple = self._triples.pop(triple_id, None)
            self._triple_to_memory.pop(triple_id, None)
            if triple is None:
                continue

            for entity in (triple.subject, triple.object):
                normalized_entity = normalize_entity(entity)
                if normalized_entity in self.node_index:
                    self.node_index[normalized_entity].discard(triple_id)
                    if not self.node_index[normalized_entity]:
                        del self.node_index[normalized_entity]

            normalized_predicate = normalize_entity(triple.predicate)
            if normalized_predicate in self.edge_index:
                self.edge_index[normalized_predicate].discard(triple_id)
                if not self.edge_index[normalized_predicate]:
                    del self.edge_index[normalized_predicate]

            key = (
                normalize_entity(triple.subject),
                normalize_entity(triple.predicate),
                normalize_entity(triple.object),
            )
            if key in self.inverted_triple_index:
                self.inverted_triple_index[key].discard(memory_id)
                if not self.inverted_triple_index[key]:
                    del self.inverted_triple_index[key]
