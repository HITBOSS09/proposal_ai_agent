"""Unit tests for online query embedding."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from math import nan
from typing import Sequence, Tuple

import pytest

from proposal_ai_agent.embeddings.providers.base import EmbeddingProvider, Vector
from proposal_ai_agent.embeddings.validators import VectorValidationError
from proposal_ai_agent.online import QueryEmbedding, QueryEngine
from proposal_ai_agent.online.providers.embedding import MemoryCache


class CountingEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider that exposes calls for cache assertions."""

    def __init__(self, vectors: Tuple[Vector, ...] = ((0.1, 0.2, 0.3),)) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed(self, text: str) -> Vector:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> Tuple[Vector, ...]:
        self.calls.append(tuple(texts))
        return self.vectors


def _processed(engine: QueryEngine, query: str = "What policy applies?"):
    return engine.process_query(engine.qualify_query(engine.receive_query(query)))


def test_embed_query_populates_cache_on_miss_and_reuses_provider_vector() -> None:
    provider = CountingEmbeddingProvider()
    cache = MemoryCache()
    engine = QueryEngine(
        embedding_provider=provider,
        embedding_dimension=3,
        embedding_cache=cache,
        embedding_model_id="offline-compatible-model",
    )
    processed = _processed(engine)

    embedding = engine.embed_query(processed)

    assert embedding.cache_hit is False
    assert embedding.vector == (0.1, 0.2, 0.3)
    assert embedding.model_id == "offline-compatible-model"
    assert embedding.embedding_dimension == 3
    assert provider.calls == [(processed.normalized_query,)]
    assert cache.get(processed.cache_key) == embedding.vector


def test_embed_query_uses_cached_vector_without_provider_invocation() -> None:
    provider = CountingEmbeddingProvider()
    cache = MemoryCache()
    engine = QueryEngine(
        embedding_provider=provider, embedding_dimension=3, embedding_cache=cache
    )
    processed = _processed(engine)
    cache.set(processed.cache_key, (0.4, 0.5, 0.6))

    embedding = engine.embed_query(processed)

    assert embedding.cache_hit is True
    assert embedding.vector == (0.4, 0.5, 0.6)
    assert provider.calls == []


@pytest.mark.parametrize(
    "vectors",
    [((0.1, 0.2),), ((0.1, nan, 0.3),)],
)
def test_embed_query_rejects_invalid_provider_vectors(vectors: Tuple[Vector, ...]) -> None:
    provider = CountingEmbeddingProvider(vectors)
    engine = QueryEngine(embedding_provider=provider, embedding_dimension=3)

    with pytest.raises(VectorValidationError):
        engine.embed_query(_processed(engine))


def test_query_embedding_contract_rejects_dimension_mismatch() -> None:
    provider = CountingEmbeddingProvider()
    engine = QueryEngine(embedding_provider=provider, embedding_dimension=3)
    processed = _processed(engine)

    with pytest.raises(ValueError, match="dimension"):
        QueryEmbedding(
            processed_query=processed,
            vector=(0.1, 0.2),
            model_id="model",
            embedding_dimension=3,
            cache_hit=False,
            embedding_timestamp_utc=datetime.now(timezone.utc),
            embedding_metadata={},
        )


def test_query_embedding_is_immutable_and_defensively_copies_metadata() -> None:
    provider = CountingEmbeddingProvider()
    engine = QueryEngine(embedding_provider=provider, embedding_dimension=3)
    embedding = engine.embed_query(_processed(engine))
    metadata = {"source": {"cache": "local"}}
    copied = replace(embedding, embedding_metadata=metadata)
    metadata["source"]["cache"] = "changed"

    with pytest.raises(FrozenInstanceError):
        copied.model_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        copied.embedding_metadata["new"] = True  # type: ignore[index]

    assert copied.embedding_metadata["source"]["cache"] == "local"


def test_query_embedding_equality_is_value_based() -> None:
    provider = CountingEmbeddingProvider()
    engine = QueryEngine(embedding_provider=provider, embedding_dimension=3)

    first = engine.embed_query(_processed(engine))
    second = replace(first)

    assert first == second


def test_embedding_regression_preserves_processed_and_qualified_queries() -> None:
    provider = CountingEmbeddingProvider()
    engine = QueryEngine(embedding_provider=provider, embedding_dimension=3)
    processed = _processed(engine, "What applies? document: handbook")
    before = (
        processed.normalized_query,
        processed.qualified_query.intent,
        processed.qualified_query.extracted_parameters,
    )

    embedding = engine.embed_query(processed)

    assert (
        embedding.processed_query.normalized_query,
        embedding.processed_query.qualified_query.intent,
        embedding.processed_query.qualified_query.extracted_parameters,
    ) == before
