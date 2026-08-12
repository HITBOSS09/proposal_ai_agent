"""Provider contract for dense embedding implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

Vector = Tuple[float, ...]


class EmbeddingProvider(ABC):
    """Synchronous provider interface used by the embedding engine."""

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Embed one text value."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> Tuple[Vector, ...]:
        """Embed text values in the supplied order."""

