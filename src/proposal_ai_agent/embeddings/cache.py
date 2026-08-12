"""Cache interfaces for content-addressed embedding reuse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from .providers.base import Vector


class EmbeddingCache(ABC):
    """Abstract cache keyed by an enriched payload's content hash."""

    @abstractmethod
    def get(self, content_hash: str) -> Optional[Vector]:
        """Return a cached vector, if present."""

    @abstractmethod
    def set(self, content_hash: str, vector: Vector) -> None:
        """Store a vector under its content hash."""


class MemoryCache(EmbeddingCache):
    """In-process cache suitable for one embedding-engine lifetime."""

    def __init__(self) -> None:
        self._vectors: Dict[str, Vector] = {}

    def get(self, content_hash: str) -> Optional[Vector]:
        return self._vectors.get(content_hash)

    def set(self, content_hash: str, vector: Vector) -> None:
        self._vectors[content_hash] = tuple(vector)

