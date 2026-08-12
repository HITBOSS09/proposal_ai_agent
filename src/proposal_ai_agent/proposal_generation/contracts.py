"""Shared proposal-domain contracts for generation boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ProposalPlan
from .prompt_composer import RetrievedReference


class SectionContent(BaseModel):
    """Raw generated content and provider telemetry for one proposal section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_id: str = Field(min_length=1)
    display_name: str | None = None
    generated_text: str = Field(min_length=1)
    generation_timestamp: datetime
    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    token_usage: Mapping[str, int]
    latency_ms: float = Field(ge=0.0)
    finish_reason: str = Field(min_length=1)
    generation_metadata: Mapping[str, Any]

    @field_validator("generation_timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generation_timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("latency_ms must be finite")
        return value

    @field_validator("token_usage")
    @classmethod
    def validate_token_usage(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        required = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(name not in value for name in required):
            raise ValueError("token_usage must include prompt_tokens, completion_tokens, and total_tokens")
        if any(isinstance(amount, bool) or not isinstance(amount, int) or amount < 0 for amount in value.values()):
            raise ValueError("token_usage values must be non-negative integers")
        if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
            raise ValueError("token_usage.total_tokens must equal prompt and completion tokens")
        return dict(value)


class ProposalDocument(BaseModel):
    """Immutable proposal-domain document assembled from ordered section content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_plan: ProposalPlan
    proposal_title: str = Field(min_length=1)
    sections: tuple[SectionContent, ...]
    references: tuple[RetrievedReference, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_document_metadata(value)

    @model_validator(mode="after")
    def validate_section_order(self) -> "ProposalDocument":
        expected_section_ids = tuple(section.section_id for section in self.proposal_plan.sections)
        actual_section_ids = tuple(section.section_id for section in self.sections)
        if actual_section_ids != expected_section_ids:
            raise ValueError("sections must exactly match proposal_plan section order")
        return self


class _FrozenDocumentMetadata(dict[str, Any]):
    """Dictionary-compatible metadata mapping that rejects mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("ProposalDocument metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_document_metadata(value: Any) -> Any:
    """Recursively detach mutable metadata containers."""
    if isinstance(value, Mapping):
        return _FrozenDocumentMetadata(
            {key: _freeze_document_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_document_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_document_metadata(item) for item in value)
    return value


class ProposalReviewReport(BaseModel):
    """Immutable findings-only result of reviewing one proposal document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_title: str = Field(min_length=1)
    findings: tuple[str, ...] = ()

    @field_validator("findings")
    @classmethod
    def validate_findings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(finding, str) or not finding.strip() for finding in value):
            raise ValueError("findings must contain non-empty strings")
        return tuple(value)

    @property
    def is_clean(self) -> bool:
        """Whether the review produced no findings."""
        return not self.findings
