"""Database-neutral immutable models for knowledge indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class _FrozenList(List[Any]):
    """List-compatible container that rejects mutation after model construction."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("IndexPoint values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class _FrozenDict(Dict[str, Any]):
    """Dict-compatible container that rejects mutation after model construction."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("IndexPoint values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_value(value: Any) -> Any:
    """Recursively freeze JSON-compatible list and dictionary values."""
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze_value(item) for item in value)
    return value


class _FrozenIndexModel(BaseModel):
    """Frozen base model that rejects fields outside the indexing contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentRole(str, Enum):
    """Application-owned authorization role for an indexing source."""

    REFERENCE_KNOWLEDGE = "REFERENCE_KNOWLEDGE"
    PUBLISHING_TEMPLATE = "PUBLISHING_TEMPLATE"


class IndexPoint(_FrozenIndexModel):
    """A database-neutral vector and its serializable retrieval payload."""

    id: str
    vector: List[float]
    payload: Dict[str, Any]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Require the pre-existing Phase 1 UUID used as the point identifier."""
        UUID(value)
        return value

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: List[float]) -> List[float]:
        """Reject empty vectors and non-finite components."""
        if not value:
            raise ValueError("vector must not be empty")
        if any(not isfinite(component) for component in value):
            raise ValueError("vector values must be finite")
        return _FrozenList(value)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure payload values can be sent to a JSON-based vector database."""
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ValueError("payload must be JSON serializable") from error
        return _freeze_value(value)


class IndexingResult(_FrozenIndexModel):
    """Outcome of one Qdrant indexing operation."""

    collection_name: str
    total_points: int
    inserted: int
    updated: int
    failed: int
    duration_ms: float


@dataclass(frozen=True, slots=True)
class IndexRequest:
    """Immutable command for one corpus indexing run."""

    input_path: Path
    collection_name: str
    document_role: DocumentRole | None = None
    batch_size: int = 64
    recreate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        if self.document_role is None:
            raise ValueError(
                "document_role is required for proposal-reference indexing; "
                "expected REFERENCE_KNOWLEDGE"
            )
        if not isinstance(self.document_role, DocumentRole):
            raise ValueError(
                f"invalid document_role {self.document_role!r}; "
                "expected REFERENCE_KNOWLEDGE"
            )
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """Immutable record of one chunk written to the vector collection."""

    chunk_id: str
    point_id: str
    document_id: str
    token_count: int


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Immutable per-document indexing outcome."""

    source_path: Path
    fingerprint: str
    document_id: str | None
    title: str | None
    chunks: Tuple[IndexedChunk, ...]
    skipped: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "chunks", tuple(self.chunks))


@dataclass(frozen=True, slots=True)
class IndexStatistics:
    """Aggregate immutable telemetry for an indexing run."""

    documents_discovered: int
    documents_indexed: int
    documents_skipped: int
    documents_failed: int
    chunks_indexed: int
    embeddings_generated: int
    vectors_uploaded: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Final immutable result returned by the production index pipeline."""

    collection_name: str
    collection_created: bool
    documents: Tuple[IndexedDocument, ...]
    statistics: IndexStatistics
    failures: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "failures", tuple(self.failures))
