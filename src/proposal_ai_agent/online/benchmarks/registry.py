"""Registry and immutable definitions for query qualification benchmarks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    """Declarative qualification rules for one supported query intent."""

    intent_id: str
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]
    defaults: Mapping[str, str] = field(default_factory=dict)
    validation_rules: Mapping[str, str] = field(default_factory=dict)
    confidence_threshold: float = 0.75
    clarification_policy: Mapping[str, bool] = field(default_factory=dict)
    intent_patterns: tuple[str, ...] = field(default_factory=tuple)
    is_default: bool = False

    def __post_init__(self) -> None:
        """Freeze profile mappings and validate declarative configuration."""
        if not self.intent_id:
            raise ValueError("intent_id must not be empty")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if set(self.required_parameters) & set(self.optional_parameters):
            raise ValueError("required and optional parameters must not overlap")
        object.__setattr__(self, "required_parameters", tuple(self.required_parameters))
        object.__setattr__(self, "optional_parameters", tuple(self.optional_parameters))
        object.__setattr__(self, "intent_patterns", tuple(self.intent_patterns))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))
        object.__setattr__(self, "validation_rules", MappingProxyType(dict(self.validation_rules)))
        object.__setattr__(
            self, "clarification_policy", MappingProxyType(dict(self.clarification_policy))
        )

    def matches(self, query: str) -> bool:
        """Return whether an explicitly configured intent pattern matches the query."""
        return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in self.intent_patterns)


class BenchmarkRegistry:
    """Lookup benchmark profiles without coupling qualification to concrete intents."""

    def __init__(self, profiles: Iterable[BenchmarkProfile] = ()) -> None:
        self._profiles: dict[str, BenchmarkProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: BenchmarkProfile) -> None:
        """Register one profile, rejecting duplicate intent identifiers."""
        if profile.intent_id in self._profiles:
            raise ValueError(f"benchmark profile already registered: {profile.intent_id}")
        if profile.is_default and any(item.is_default for item in self._profiles.values()):
            raise ValueError("only one benchmark profile may be the default")
        self._profiles[profile.intent_id] = profile

    def get(self, intent_id: str) -> BenchmarkProfile:
        """Return a profile by intent identifier."""
        return self._profiles[intent_id]

    def select(self, query: str) -> BenchmarkProfile:
        """Select a matching profile, falling back to the configured default."""
        for profile in self._profiles.values():
            if profile.matches(query):
                return profile
        for profile in self._profiles.values():
            if profile.is_default:
                return profile
        raise LookupError("no benchmark profile matches the query")
