"""Immutable contracts for context assembly in the online pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from .query import ValidationResult
from .retrieval import RetrievedContext


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Defensively freeze mapping values, including nested containers."""
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
class Citation:
    """Provenance record for one assembled context source."""

    citation_id: str
    chunk_id: str
    document_id: str
    header_path: tuple[str, ...]
    chunk_index: int
    token_count: int
    page: int | None
    truncated: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.citation_id, str) or not self.citation_id:
            raise ValueError("citation_id must be a non-empty string")
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int):
            raise TypeError("token_count must be an integer")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a bool")
        header_path = tuple(self.header_path)
        if any(not isinstance(header, str) for header in header_path):
            raise TypeError("header_path values must be strings")
        object.__setattr__(self, "header_path", header_path)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Immutable representation of the assembled context for synthesis."""

    retrieved_context: RetrievedContext
    assembled_context: str
    citations: tuple[Citation, ...]
    metadata: Mapping[str, Any]
    context_statistics: Mapping[str, Any]
    token_usage: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.retrieved_context, RetrievedContext):
            raise TypeError("retrieved_context must be a RetrievedContext")
        if not isinstance(self.assembled_context, str):
            raise TypeError("assembled_context must be a string")
        if not self.assembled_context.strip():
            raise ValueError("assembled_context must not be empty")

        citations = tuple(self.citations)
        if any(not isinstance(citation, Citation) for citation in citations):
            raise TypeError("citations must contain Citation values")
        citation_ids = [citation.citation_id for citation in citations]
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation identifiers must be unique")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.context_statistics, Mapping):
            raise TypeError("context_statistics must be a mapping")
        if not isinstance(self.token_usage, Mapping):
            raise TypeError("token_usage must be a mapping")

        token_budget = self.token_usage.get("budget_tokens")
        token_used = self.token_usage.get("used_tokens")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int):
            raise TypeError("token_usage.budget_tokens must be an integer")
        if isinstance(token_used, bool) or not isinstance(token_used, int):
            raise TypeError("token_usage.used_tokens must be an integer")
        if token_budget < 0 or token_used < 0:
            raise ValueError("token usage values must be non-negative")
        if token_used > token_budget:
            raise ValueError("used_tokens cannot exceed budget_tokens")

        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "context_statistics", _freeze_mapping(self.context_statistics))
        object.__setattr__(self, "token_usage", _freeze_mapping(self.token_usage))


@dataclass(frozen=True, slots=True)
class PromptPackage:
    """Immutable prompt package constructed from assembled context."""

    system_prompt: str
    user_prompt: str
    conversation_history: tuple[str, ...]
    assembled_context: AssembledContext
    prompt_template: str
    output_format: str
    prompt_statistics: Mapping[str, Any]
    generation_metadata: Mapping[str, Any]
    validation_result: ValidationResult

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        if not isinstance(self.conversation_history, tuple):
            raise TypeError("conversation_history must be a tuple of strings")
        if any(not isinstance(item, str) for item in self.conversation_history):
            raise TypeError("conversation_history must contain strings")
        if not isinstance(self.prompt_template, str) or not self.prompt_template.strip():
            raise ValueError("prompt_template must be a non-empty string")
        if not isinstance(self.output_format, str) or not self.output_format.strip():
            raise ValueError("output_format must be a non-empty string")
        if not isinstance(self.validation_result, ValidationResult):
            raise TypeError("validation_result must be a ValidationResult")

        object.__setattr__(self, "prompt_statistics", _freeze_mapping(self.prompt_statistics))
        object.__setattr__(self, "generation_metadata", _freeze_mapping(self.generation_metadata))

        self._validate_token_statistics(self.prompt_statistics)

    @staticmethod
    def _validate_token_statistics(statistics: Mapping[str, Any]) -> None:
        required_keys = (
            "system_tokens",
            "user_tokens",
            "history_tokens",
            "context_tokens",
            "total_tokens",
        )
        for key in required_keys:
            if key not in statistics:
                raise ValueError(f"prompt_statistics must include {key}")
            value = statistics[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"prompt_statistics.{key} must be an integer")
            if value < 0:
                raise ValueError(f"prompt_statistics.{key} must be non-negative")
        if statistics["total_tokens"] != (
            statistics["system_tokens"]
            + statistics["user_tokens"]
            + statistics["history_tokens"]
            + statistics["context_tokens"]
        ):
            raise ValueError("prompt_statistics.total_tokens must equal the sum of component tokens")
