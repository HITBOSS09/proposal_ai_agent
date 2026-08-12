"""Immutable contract for a processed query's dense embedding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import Any

from .query import ProcessedQuery


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Create an immutable, recursively frozen metadata mapping."""
    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return frozenset(freeze(item) for item in value)
        return value

    return MappingProxyType({key: freeze(value) for key, value in metadata.items()})


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """Validated, immutable embedding generated for a processed query."""

    processed_query: ProcessedQuery
    vector: tuple[float, ...]
    model_id: str
    embedding_dimension: int
    cache_hit: bool
    embedding_timestamp_utc: datetime
    embedding_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate vector integrity and defensively freeze mutable values."""
        if not isinstance(self.processed_query, ProcessedQuery):
            raise TypeError("processed_query must be a ProcessedQuery")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        vector = tuple(self.vector)
        if not vector:
            raise ValueError("vector must not be empty")
        if len(vector) != self.embedding_dimension:
            raise ValueError("vector dimension does not match embedding_dimension")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in vector
        ):
            raise ValueError("vector values must be finite numeric values")
        if not isinstance(self.embedding_timestamp_utc, datetime):
            raise TypeError("embedding_timestamp_utc must be a datetime")
        if self.embedding_timestamp_utc.tzinfo is None:
            raise ValueError("embedding_timestamp_utc must be timezone-aware")

        object.__setattr__(self, "vector", tuple(float(value) for value in vector))
        object.__setattr__(
            self,
            "embedding_timestamp_utc",
            self.embedding_timestamp_utc.astimezone(timezone.utc),
        )
        object.__setattr__(self, "embedding_metadata", _freeze_metadata(self.embedding_metadata))
