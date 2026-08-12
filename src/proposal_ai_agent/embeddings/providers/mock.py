"""Deterministic provider for tests and local development."""

from __future__ import annotations

from hashlib import sha256
from typing import Sequence, Tuple

from proposal_ai_agent.ingestion.metadata import normalize_text

from .base import EmbeddingProvider, Vector


class MockEmbeddingProvider(EmbeddingProvider):
    """Produce deterministic dense vectors without a model dependency."""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> Vector:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> Tuple[Vector, ...]:
        return tuple(self._embed_normalized(normalize_text(text)) for text in texts)

    def _embed_normalized(self, text: str) -> Vector:
        values = []
        counter = 0
        while len(values) < self.dimensions:
            digest = sha256(f"{text}:{counter}".encode("utf-8")).digest()
            values.extend(byte / 255.0 for byte in digest)
            counter += 1
        return tuple(values[: self.dimensions])

