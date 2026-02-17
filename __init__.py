"""Synapse AI Memory — A neuroscience-inspired memory engine for AI agents.

Usage::

    from synapse import Synapse, Memory

    s = Synapse()
    s.remember("I prefer vegetarian food")
    results = s.recall("dietary preferences")
"""

from synapse import Memory, Synapse, MEMORY_TYPES, EDGE_TYPES
from context_cards import CardDeck, ContextCard
from exceptions import (
    SynapseAuthError,
    SynapseConnectionError,
    SynapseError,
    SynapseFormatError,
    SynapseValidationError,
)

__version__ = "0.13.0"
__all__ = [
    "Memory",
    "Synapse",
    "MEMORY_TYPES",
    "EDGE_TYPES",
    "SynapseError",
    "SynapseFormatError",
    "SynapseAuthError",
    "SynapseConnectionError",
    "SynapseValidationError",
    "ContextCard",
    "CardDeck",
]
