"""Immutable response contracts for the final answer stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import Any


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return frozenset(freeze(item) for item in value)
        return value

    return MappingProxyType({key: freeze(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class GeneratedResponse:
    """Immutable generated response from a single selected LLM provider."""

    generated_text: str
    provider_name: str
    model_name: str
    finish_reason: str
    token_usage: Mapping[str, int]
    latency_ms: float
    generation_timestamp: datetime
    generation_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.generated_text, str):
            raise TypeError("generated_text must be a string")
        if not self.generated_text.strip():
            raise ValueError("generated_text must not be empty")
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
            raise ValueError("finish_reason must be a non-empty string")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise TypeError("latency_ms must be numeric")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not isinstance(self.generation_timestamp, datetime):
            raise TypeError("generation_timestamp must be a datetime")
        if self.generation_timestamp.tzinfo is None:
            raise ValueError("generation_timestamp must be timezone-aware")
        if not isinstance(self.token_usage, Mapping):
            raise TypeError("token_usage must be a mapping")
        if not isinstance(self.generation_metadata, Mapping):
            raise TypeError("generation_metadata must be a mapping")

        token_usage = {key: value for key, value in self.token_usage.items()}
        if "prompt_tokens" not in token_usage:
            raise ValueError("token_usage must include prompt_tokens")
        if "completion_tokens" not in token_usage:
            raise ValueError("token_usage must include completion_tokens")
        if "total_tokens" not in token_usage:
            raise ValueError("token_usage must include total_tokens")
        for key, value in token_usage.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"token_usage.{key} must be an integer")
            if value < 0:
                raise ValueError(f"token_usage.{key} must be non-negative")
        if token_usage["total_tokens"] != token_usage["prompt_tokens"] + token_usage["completion_tokens"]:
            raise ValueError("token_usage.total_tokens must equal prompt_tokens + completion_tokens")

        object.__setattr__(self, "token_usage", _freeze_mapping(token_usage))
        object.__setattr__(
            self,
            "generation_timestamp",
            self.generation_timestamp.astimezone(timezone.utc),
        )
        object.__setattr__(self, "generation_metadata", _freeze_mapping(self.generation_metadata))


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Immutable source citation for the final response."""

    citation_id: str
    chunk_id: str
    document_id: str
    document_name: str | None
    section: str | None
    hierarchy_path: tuple[str, ...]
    page: int | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.citation_id, str) or not self.citation_id:
            raise ValueError("citation_id must be a non-empty string")
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        header_path = tuple(self.hierarchy_path)
        if any(not isinstance(item, str) for item in header_path):
            raise TypeError("hierarchy_path values must be strings")
        if self.document_name is not None and not isinstance(self.document_name, str):
            raise TypeError("document_name must be a string or None")
        if self.section is not None and not isinstance(self.section, str):
            raise TypeError("section must be a string or None")
        if self.page is not None and (isinstance(self.page, bool) or not isinstance(self.page, int)):
            raise TypeError("page must be an integer or None")
        object.__setattr__(self, "hierarchy_path", header_path)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    """Immutable metadata for the generated response."""

    provider_name: str
    model_name: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    generation_timestamp: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
            raise ValueError("finish_reason must be a non-empty string")
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise TypeError("latency_ms must be numeric")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not isinstance(self.generation_timestamp, datetime):
            raise TypeError("generation_timestamp must be a datetime")
        if self.generation_timestamp.tzinfo is None:
            raise ValueError("generation_timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "generation_timestamp",
            self.generation_timestamp.astimezone(timezone.utc),
        )


@dataclass(frozen=True, slots=True)
class AuditTrace:
    """Immutable trace data for the final response."""

    request_id: Any
    session_id: str | None
    query_hash: str
    retrieval_summary: Mapping[str, Any]
    provider: str
    execution_timestamp: datetime

    def __post_init__(self) -> None:
        if self.request_id is None:
            raise ValueError("request_id must not be None")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string or None")
        if not isinstance(self.query_hash, str) or not self.query_hash.strip():
            raise ValueError("query_hash must be a non-empty string")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.retrieval_summary, Mapping):
            raise TypeError("retrieval_summary must be a mapping")
        if not isinstance(self.execution_timestamp, datetime):
            raise TypeError("execution_timestamp must be a datetime")
        if self.execution_timestamp.tzinfo is None:
            raise ValueError("execution_timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "retrieval_summary",
            _freeze_mapping(self.retrieval_summary),
        )
        object.__setattr__(
            self,
            "execution_timestamp",
            self.execution_timestamp.astimezone(timezone.utc),
        )


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    """Final answer response delivered to downstream applications."""

    final_answer: str
    citations: tuple[SourceCitation, ...]
    response_metadata: ResponseMetadata
    audit_trace: AuditTrace

    def __post_init__(self) -> None:
        if not isinstance(self.final_answer, str):
            raise TypeError("final_answer must be a string")
        if not self.final_answer.strip():
            raise ValueError("final_answer must not be empty")
        citations = tuple(self.citations)
        if any(not isinstance(citation, SourceCitation) for citation in citations):
            raise TypeError("citations must contain SourceCitation values")
        citation_ids = [citation.citation_id for citation in citations]
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation identifiers must be unique")
        if not isinstance(self.response_metadata, ResponseMetadata):
            raise TypeError("response_metadata must be a ResponseMetadata")
        if not isinstance(self.audit_trace, AuditTrace):
            raise TypeError("audit_trace must be an AuditTrace")
