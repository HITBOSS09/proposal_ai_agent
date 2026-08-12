"""Immutable retrieval-planning contracts with no retrieval execution behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any

from .embedding import QueryEmbedding


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


class RetrievalStrategy(str, Enum):
    """Retrieval execution strategy selected by the planner."""

    DENSE = "DENSE"
    HYBRID = "HYBRID"
    KEYWORD = "KEYWORD"


class RerankingPolicy(str, Enum):
    """Post-retrieval reranking policy selected by the planner."""

    DISABLED = "DISABLED"
    CROSS_ENCODER = "CROSS_ENCODER"


class HybridSearchPolicy(str, Enum):
    """How a future retrieval executor should combine search modes."""

    DENSE_ONLY = "DENSE_ONLY"
    HYBRID = "HYBRID"
    KEYWORD_ONLY = "KEYWORD_ONLY"


class ACLPolicy(str, Enum):
    """Placeholder ACL policy; enforcement belongs to a future execution phase."""

    PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Declarative boundaries for a future retrieval executor."""

    knowledge_base: str | None = None
    collection: str | None = None
    document: str | None = None
    version: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze future scope extensions."""
        object.__setattr__(self, "extensions", _freeze_mapping(self.extensions))


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """Limits a future retrieval execution is expected to observe."""

    max_candidates: int
    max_latency_ms: int
    max_context_tokens: int

    def __post_init__(self) -> None:
        """Reject non-positive execution budget values."""
        if min(self.max_candidates, self.max_latency_ms, self.max_context_tokens) <= 0:
            raise ValueError("retrieval budget values must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Complete immutable retrieval blueprint derived from a query embedding."""

    query_embedding: QueryEmbedding
    metadata_filters: Mapping[str, Any]
    search_scope: SearchScope
    retrieval_strategy: RetrievalStrategy
    retrieval_profile: str
    top_k: int
    candidate_budget: int
    score_threshold: float
    reranking_policy: RerankingPolicy
    hybrid_policy: HybridSearchPolicy
    acl_policy: ACLPolicy
    retrieval_budget: RetrievalBudget
    planner_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate planner values and defensively freeze mappings."""
        if not isinstance(self.query_embedding, QueryEmbedding):
            raise TypeError("query_embedding must be a QueryEmbedding")
        if not isinstance(self.search_scope, SearchScope):
            raise TypeError("search_scope must be a SearchScope")
        if not isinstance(self.retrieval_strategy, RetrievalStrategy):
            raise TypeError("retrieval_strategy must be a RetrievalStrategy")
        if not isinstance(self.reranking_policy, RerankingPolicy):
            raise TypeError("reranking_policy must be a RerankingPolicy")
        if not isinstance(self.hybrid_policy, HybridSearchPolicy):
            raise TypeError("hybrid_policy must be a HybridSearchPolicy")
        if not isinstance(self.acl_policy, ACLPolicy):
            raise TypeError("acl_policy must be an ACLPolicy")
        if not isinstance(self.retrieval_budget, RetrievalBudget):
            raise TypeError("retrieval_budget must be a RetrievalBudget")
        if not self.retrieval_profile.strip():
            raise ValueError("retrieval_profile must not be empty")
        if self.top_k <= 0 or self.candidate_budget <= 0:
            raise ValueError("top_k and candidate_budget must be positive")
        if self.candidate_budget < self.top_k:
            raise ValueError("candidate_budget must be at least top_k")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        object.__setattr__(self, "metadata_filters", _freeze_mapping(self.metadata_filters))
        object.__setattr__(self, "planner_metadata", _freeze_mapping(self.planner_metadata))


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """One repository-returned chunk, preserved without refinement."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Mapping[str, Any]
    header_path: tuple[str, ...]
    chunk_index: int
    point_id: str = ""
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Validate scalar fields and isolate caller-owned containers."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer")
        if self.point_id and not isinstance(self.point_id, str):
            raise TypeError("point_id must be a string")
        if self.page_number is not None and (isinstance(self.page_number, bool) or not isinstance(self.page_number, int)):
            raise TypeError("page_number must be an integer or None")

        header_path = tuple(self.header_path)
        if any(not isinstance(header, str) for header in header_path):
            raise TypeError("header_path values must be strings")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "header_path", header_path)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RetrievedCandidates:
    """Immutable output of knowledge retrieval with repository telemetry."""

    retrieval_request: RetrievalRequest
    candidates: tuple[RetrievedCandidate, ...]
    retrieval_time_ms: float
    candidate_count: int
    repository_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Ensure the result remains a faithful immutable retrieval record."""
        if not isinstance(self.retrieval_request, RetrievalRequest):
            raise TypeError("retrieval_request must be a RetrievalRequest")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, RetrievedCandidate) for candidate in candidates):
            raise TypeError("candidates must contain RetrievedCandidate values")
        if isinstance(self.retrieval_time_ms, bool) or not isinstance(
            self.retrieval_time_ms, (int, float)
        ):
            raise TypeError("retrieval_time_ms must be numeric")
        if not isfinite(self.retrieval_time_ms) or self.retrieval_time_ms < 0:
            raise ValueError("retrieval_time_ms must be finite and non-negative")
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
            raise TypeError("candidate_count must be an integer")
        if self.candidate_count != len(candidates):
            raise ValueError("candidate_count must equal the number of candidates")

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "retrieval_time_ms", float(self.retrieval_time_ms))
        object.__setattr__(self, "repository_metadata", _freeze_mapping(self.repository_metadata))


@dataclass(frozen=True, slots=True)
class ProcessedCandidate:
    """A validated retrieval candidate after deterministic local processing."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Mapping[str, Any]
    header_path: tuple[str, ...]
    chunk_index: int
    processing_flags: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate values and defensively freeze all mapping-based fields."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer")

        header_path = tuple(self.header_path)
        if any(not isinstance(header, str) for header in header_path):
            raise TypeError("header_path values must be strings")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "header_path", header_path)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "processing_flags", _freeze_mapping(self.processing_flags))


@dataclass(frozen=True, slots=True)
class ProcessedCandidates:
    """Immutable output of the post-retrieval candidate processor."""

    retrieval_request: RetrievalRequest
    candidates: tuple[ProcessedCandidate, ...]
    processing_summary: Mapping[str, Any]
    processing_time_ms: float

    def __post_init__(self) -> None:
        """Keep the processor result self-contained and immutable."""
        if not isinstance(self.retrieval_request, RetrievalRequest):
            raise TypeError("retrieval_request must be a RetrievalRequest")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, ProcessedCandidate) for candidate in candidates):
            raise TypeError("candidates must contain ProcessedCandidate values")
        if isinstance(self.processing_time_ms, bool) or not isinstance(
            self.processing_time_ms, (int, float)
        ):
            raise TypeError("processing_time_ms must be numeric")
        if not isfinite(self.processing_time_ms) or self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be finite and non-negative")

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "processing_time_ms", float(self.processing_time_ms))
        object.__setattr__(self, "processing_summary", _freeze_mapping(self.processing_summary))


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """Final immutable context selected from post-retrieval candidates."""

    retrieval_request: RetrievalRequest
    candidates: tuple[ProcessedCandidate, ...]
    final_context: str
    total_candidates: int
    returned_candidates: int
    reranking_applied: bool
    reranking_time_ms: float
    retrieval_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate context accounting and isolate summary metadata."""
        if not isinstance(self.retrieval_request, RetrievalRequest):
            raise TypeError("retrieval_request must be a RetrievalRequest")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, ProcessedCandidate) for candidate in candidates):
            raise TypeError("candidates must contain ProcessedCandidate values")
        if not isinstance(self.final_context, str):
            raise TypeError("final_context must be a string")
        if isinstance(self.total_candidates, bool) or not isinstance(self.total_candidates, int):
            raise TypeError("total_candidates must be an integer")
        if isinstance(self.returned_candidates, bool) or not isinstance(self.returned_candidates, int):
            raise TypeError("returned_candidates must be an integer")
        if self.total_candidates < 0 or self.returned_candidates < 0:
            raise ValueError("candidate counts must be non-negative")
        if self.returned_candidates != len(candidates):
            raise ValueError("returned_candidates must equal the number of candidates")
        if self.returned_candidates > self.total_candidates:
            raise ValueError("returned_candidates cannot exceed total_candidates")
        if not isinstance(self.reranking_applied, bool):
            raise TypeError("reranking_applied must be a bool")
        if isinstance(self.reranking_time_ms, bool) or not isinstance(
            self.reranking_time_ms, (int, float)
        ):
            raise TypeError("reranking_time_ms must be numeric")
        if not isfinite(self.reranking_time_ms) or self.reranking_time_ms < 0:
            raise ValueError("reranking_time_ms must be finite and non-negative")

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "reranking_time_ms", float(self.reranking_time_ms))
        object.__setattr__(self, "retrieval_summary", _freeze_mapping(self.retrieval_summary))
