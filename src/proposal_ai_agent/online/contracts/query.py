"""Immutable contract representing one received user query."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from uuid import UUID


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Copy metadata into an immutable mapping, recursively freezing containers."""
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(
        {key: _freeze_value(value) for key, value in metadata.items()}
    )


def _freeze_value(value: Any) -> Any:
    """Freeze mutable containers used in query metadata."""
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class UserQuery:
    """Normalized, immutable input to the online query pipeline."""

    request_id: UUID
    query: str
    timestamp_utc: datetime
    session_id: str | None = None
    conversation_history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    user_context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    auth_context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    trace_metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate required fields and freeze all metadata containers."""
        if not isinstance(self.request_id, UUID) or self.request_id.version != 4:
            raise ValueError("request_id must be a UUID4")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        normalized_query = self.query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not isinstance(self.timestamp_utc, datetime) or self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string or None")
        if not isinstance(self.conversation_history, Sequence):
            raise TypeError("conversation_history must be a sequence of mappings")

        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "timestamp_utc", self.timestamp_utc.astimezone(timezone.utc))
        object.__setattr__(
            self,
            "conversation_history",
            tuple(_freeze_metadata(item) for item in self.conversation_history),
        )
        object.__setattr__(self, "user_context", _freeze_metadata(self.user_context))
        object.__setattr__(self, "auth_context", _freeze_metadata(self.auth_context))
        object.__setattr__(self, "trace_metadata", _freeze_metadata(self.trace_metadata))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating extracted parameters against a benchmark profile."""

    is_valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Freeze validation errors supplied by direct contract construction."""
        object.__setattr__(self, "errors", tuple(self.errors))


@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    """One specific, machine-readable reason a query needs clarification."""

    parameter: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class QualifiedQuery:
    """Immutable result of benchmark-driven query qualification."""

    original: UserQuery
    intent: str
    benchmark_id: str
    extracted_parameters: Mapping[str, Any]
    missing_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]
    confidence_score: float
    ambiguity_flags: tuple[str, ...]
    conflict_flags: tuple[str, ...]
    clarification_required: bool
    clarification_requests: tuple[ClarificationRequest, ...]
    validation_result: ValidationResult

    def __post_init__(self) -> None:
        """Defensively freeze qualification values and validate derived state."""
        if not isinstance(self.original, UserQuery):
            raise TypeError("original must be a UserQuery")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if self.clarification_required != bool(self.clarification_requests):
            raise ValueError("clarification state must match clarification requests")
        object.__setattr__(
            self, "extracted_parameters", _freeze_metadata(self.extracted_parameters)
        )
        object.__setattr__(self, "missing_parameters", tuple(self.missing_parameters))
        object.__setattr__(self, "optional_parameters", tuple(self.optional_parameters))
        object.__setattr__(self, "ambiguity_flags", tuple(self.ambiguity_flags))
        object.__setattr__(self, "conflict_flags", tuple(self.conflict_flags))
        object.__setattr__(self, "clarification_requests", tuple(self.clarification_requests))


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    """Immutable syntactic preparation of a qualified query for downstream use."""

    qualified_query: QualifiedQuery
    normalized_query: str
    language: str
    language_confidence: float
    query_hash: str
    cache_key: str
    character_count: int
    word_count: int
    estimated_token_count: int
    processing_timestamp_utc: datetime
    processing_version: str
    processing_flags: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate scalar processing outputs and defensively freeze flags."""
        if not isinstance(self.qualified_query, QualifiedQuery):
            raise TypeError("qualified_query must be a QualifiedQuery")
        if not 0.0 <= self.language_confidence <= 1.0:
            raise ValueError("language_confidence must be between 0.0 and 1.0")
        if not isinstance(self.processing_timestamp_utc, datetime):
            raise TypeError("processing_timestamp_utc must be a datetime")
        if self.processing_timestamp_utc.tzinfo is None:
            raise ValueError("processing_timestamp_utc must be timezone-aware")
        object.__setattr__(
            self,
            "processing_timestamp_utc",
            self.processing_timestamp_utc.astimezone(timezone.utc),
        )
        object.__setattr__(self, "processing_flags", _freeze_metadata(self.processing_flags))
