"""Adapter for an injected BGE-compatible embedding client."""

from __future__ import annotations

from typing import Protocol, Sequence, Tuple

from .base import EmbeddingProvider, Vector


class BGEClient(Protocol):
    """Minimal protocol implemented by sentence-transformers style clients."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts into dense vectors."""


class BGEProvider(EmbeddingProvider):
    """Embedding provider backed by an injected BGE-compatible client."""

    def __init__(self, client: BGEClient) -> None:
        self._client = client

    def embed(self, text: str) -> Vector:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> Tuple[Vector, ...]:
        return tuple(tuple(float(value) for value in vector) for vector in self._client.encode(texts))

