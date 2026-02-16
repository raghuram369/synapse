"""Runtime scope policy for clamping caller-requested memory scope."""

from __future__ import annotations

from dataclasses import dataclass


MEMORY_SCOPES = ("public", "shared", "private")


@dataclass(frozen=True)
class ScopePolicy:
    """Immutable max-scope policy enforced at runtime."""

    max_scope: str

    def __post_init__(self) -> None:
        normalized = str(self.max_scope).strip().lower()
        if normalized not in MEMORY_SCOPES:
            raise ValueError(f"Invalid max_scope: {self.max_scope}. Must be one of {MEMORY_SCOPES}")
        object.__setattr__(self, "max_scope", normalized)

    def enforce(self, requested_scope: str) -> str:
        """Clamp requested scope to the policy max scope."""
        requested = str(requested_scope).strip().lower()
        if requested not in MEMORY_SCOPES:
            raise ValueError(
                f"Invalid requested_scope: {requested_scope}. Must be one of {MEMORY_SCOPES}"
            )
        return MEMORY_SCOPES[min(MEMORY_SCOPES.index(requested), MEMORY_SCOPES.index(self.max_scope))]

    @classmethod
    def owner(cls) -> "ScopePolicy":
        return cls(max_scope="private")

    @classmethod
    def external(cls) -> "ScopePolicy":
        return cls(max_scope="shared")

    @classmethod
    def anonymous(cls) -> "ScopePolicy":
        return cls(max_scope="public")
