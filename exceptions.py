"""Custom exceptions for the Synapse memory engine."""

from __future__ import annotations


class SynapseError(Exception):
    """Base exception for all Synapse errors."""


class SynapseFormatError(SynapseError):
    """Raised when data format is invalid (corrupt files, bad schemas)."""


class SynapseAuthError(SynapseError):
    """Raised on authentication or authorization failures."""


class SynapseConnectionError(SynapseError):
    """Raised when a network connection to a peer or daemon fails."""


class SynapseValidationError(SynapseError):
    """Raised when input validation fails (empty content, bad types)."""
