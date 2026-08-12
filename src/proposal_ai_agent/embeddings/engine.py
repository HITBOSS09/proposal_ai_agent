"""Content-addressed orchestration for dense embedding providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from proposal_ai_agent.ingestion.metadata import EnrichedChunkPayload, compute_content_hash

from .cache import EmbeddingCache
from .providers.base import EmbeddingProvider, Vector
from .validators import VectorValidator


@dataclass(frozen=True, slots=True)
class EmbeddingEngineConfig:
    """Configuration that supplies the selected provider and vector dimension."""

    provider: EmbeddingProvider
    dimensions: int


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Validated vector paired with its immutable source payload."""

    payload: EnrichedChunkPayload
    vector: Vector
    cache_hit: bool


class EmbeddingEngine:
    """Embed enriched chunks with validated, content-addressed caching."""

    def __init__(
        self,
        config: EmbeddingEngineConfig,
        cache: EmbeddingCache,
        validator: Optional[VectorValidator] = None,
    ) -> None:
        self._provider = config.provider
        self._cache = cache
        self._validator = validator or VectorValidator(config.dimensions)
        if self._validator.dimensions != config.dimensions:
            raise ValueError("validator dimensions must match config dimensions")

    def embed(self, payload: EnrichedChunkPayload) -> EmbeddingResult:
        """Embed one payload using the same cache-aware batch workflow."""
        return self.embed_batch((payload,))[0]

    def embed_batch(
        self, payloads: Iterable[EnrichedChunkPayload]
    ) -> Tuple[EmbeddingResult, ...]:
        """Embed payloads in input order, invoking the provider only for misses."""
        ordered_payloads = tuple(payloads)
        cached_vectors: Dict[str, Vector] = {}
        missing_payloads: Dict[str, EnrichedChunkPayload] = {}

        for payload in ordered_payloads:
            content_hash = compute_content_hash(payload.embedding_text)
            if content_hash in cached_vectors or content_hash in missing_payloads:
                continue
            cached = self._cache.get(content_hash)
            if cached is None:
                missing_payloads[content_hash] = payload
            else:
                cached_vectors[content_hash] = self._validator.validate(cached)

        if missing_payloads:
            missing_items = tuple(missing_payloads.items())
            vectors = self._provider.embed_batch(
                tuple(payload.embedding_text for _, payload in missing_items)
            )
            if len(vectors) != len(missing_items):
                raise ValueError("provider returned a vector count different from request count")
            for (content_hash, _), vector in zip(missing_items, vectors):
                validated_vector = self._validator.validate(vector)
                self._cache.set(content_hash, validated_vector)
                cached_vectors[content_hash] = validated_vector

        return tuple(
            EmbeddingResult(
                payload=payload,
                vector=cached_vectors[compute_content_hash(payload.embedding_text)],
                cache_hit=compute_content_hash(payload.embedding_text) not in missing_payloads,
            )
            for payload in ordered_payloads
        )


__all__ = ["EmbeddingEngine", "EmbeddingEngineConfig", "EmbeddingResult"]
