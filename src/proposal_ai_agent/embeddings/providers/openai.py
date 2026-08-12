"""Adapter for an injected OpenAI embeddings client."""

from __future__ import annotations

from typing import Protocol, Sequence, Tuple

from .base import EmbeddingProvider, Vector



class OpenAIEmbeddingItem(Protocol):
    """Minimal response item returned by the OpenAI embeddings endpoint."""

    embedding: Sequence[float]


class OpenAIEmbeddingResponse(Protocol):
    """Minimal response returned by the OpenAI embeddings endpoint."""

    data: Sequence[OpenAIEmbeddingItem]


class OpenAIEmbeddingsResource(Protocol):
    """Embeddings endpoint used by the official OpenAI client."""

    def create(self, *, model: str, input: Sequence[str]) -> OpenAIEmbeddingResponse:
        """Create dense embeddings."""


class OpenAIClient(Protocol):
    """Minimal official-client surface required by ``OpenAIProvider``."""

    embeddings: OpenAIEmbeddingsResource


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider backed by an injected official OpenAI client."""

    def __init__(self, client: OpenAIClient, model: str) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self._client = client
        self._model = model

    def embed(self, text: str) -> Vector:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> Tuple[Vector, ...]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return tuple(tuple(float(value) for value in item.embedding) for item in response.data)

