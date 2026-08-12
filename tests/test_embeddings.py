"""Unit tests for the Phase 2.2 embedding engine."""

from datetime import datetime, timezone
from math import inf, nan
from typing import Sequence, Tuple
from unittest.mock import Mock
from uuid import uuid4

import pytest

from proposal_ai_agent.embeddings import (
    BGEProvider,
    EmbeddingEngine,
    EmbeddingEngineConfig,
    EmbeddingProvider,
    MemoryCache,
    MockEmbeddingProvider,
    OpenAIProvider,
    VectorValidationError,
    VectorValidator,
)
from proposal_ai_agent.ingestion.chunk_models import Chunk
from proposal_ai_agent.ingestion.metadata import enrich_chunk


def make_payload(text: str = "Example chunk text"):
    chunk = Chunk(
        document_id=uuid4(),
        section_id=uuid4(),
        section_path=["Introduction"],
        heading="Introduction",
        document_type="proposal",
        language="en",
        source_file="proposal.docx",
        source_path="/documents/proposal.docx",
        chunk_index=1,
        order_start=0,
        order_end=0,
        text=text,
        token_count=len(text.split()),
    )
    return enrich_chunk(chunk, datetime(2026, 1, 1, tzinfo=timezone.utc))


class CountingProvider(EmbeddingProvider):
    """Test provider exposing batch calls without changing production behavior."""

    def __init__(self) -> None:
        self.batch_calls = 0

    def embed(self, text: str) -> Tuple[float, ...]:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> Tuple[Tuple[float, ...], ...]:
        self.batch_calls += 1
        return tuple((float(index), 1.0, 2.0) for index, _ in enumerate(texts))


class FakeBGEClient:
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple((float(index), 1.0) for index, _ in enumerate(texts))


class FakeEmbeddingItem:
    def __init__(self, embedding: Sequence[float]) -> None:
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data: Sequence[FakeEmbeddingItem]) -> None:
        self.data = data


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.model = ""
        self.input: Sequence[str] = ()

    def create(self, *, model: str, input: Sequence[str]) -> FakeEmbeddingResponse:
        self.model = model
        self.input = input
        return FakeEmbeddingResponse(
            tuple(FakeEmbeddingItem((float(index), 1.0)) for index, _ in enumerate(input))
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


def make_engine(provider: EmbeddingProvider, dimensions: int = 3) -> EmbeddingEngine:
    return EmbeddingEngine(
        EmbeddingEngineConfig(provider=provider, dimensions=dimensions), MemoryCache()
    )


def test_memory_cache_stores_vectors_by_content_hash() -> None:
    cache = MemoryCache()
    assert cache.get("hash") is None
    cache.set("hash", (1.0, 2.0))
    assert cache.get("hash") == (1.0, 2.0)


def test_mock_provider_is_deterministic_and_implements_interface() -> None:
    provider = MockEmbeddingProvider(dimensions=4)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.embed("same text") == provider.embed("same text")
    assert len(provider.embed_batch(("one", "two"))) == 2


def test_bge_provider_delegates_to_injected_client() -> None:
    provider = BGEProvider(FakeBGEClient())
    assert provider.embed_batch(("one", "two")) == ((0.0, 1.0), (1.0, 1.0))


def test_openai_provider_delegates_to_injected_client() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIProvider(client, model="text-embedding-model")
    assert provider.embed("one") == (0.0, 1.0)
    assert client.embeddings.model == "text-embedding-model"
    assert client.embeddings.input == ("one",)


def test_embed_returns_validated_vector() -> None:
    payload = make_payload()
    result = make_engine(MockEmbeddingProvider(dimensions=3)).embed(payload)
    assert result.payload == payload
    assert len(result.vector) == 3
    assert result.cache_hit is False


def test_cache_hit_avoids_provider_call() -> None:
    provider = CountingProvider()
    engine = make_engine(provider)
    payload = make_payload()
    first = engine.embed(payload)
    second = engine.embed(payload)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert provider.batch_calls == 1
    assert first.vector == second.vector


def test_cache_miss_uses_provider_batch_and_preserves_order() -> None:
    provider = CountingProvider()
    engine = make_engine(provider)
    first = make_payload("first")
    second = make_payload("second")
    results = engine.embed_batch((first, second))
    assert provider.batch_calls == 1
    assert [result.payload for result in results] == [first, second]
    assert [result.cache_hit for result in results] == [False, False]


def test_batch_deduplicates_matching_content_hashes() -> None:
    provider = CountingProvider()
    engine = make_engine(provider)
    first = make_payload("duplicate")
    second = make_payload("duplicate")
    results = engine.embed_batch((first, second))
    assert provider.batch_calls == 1
    assert results[0].vector == results[1].vector


@pytest.mark.parametrize(
    "vector",
    [
        (1.0, 2.0),
        (1.0, nan, 3.0),
        (1.0, inf, 3.0),
        (1.0, True, 3.0),
    ],
)
def test_vector_validator_rejects_invalid_vectors(vector: Tuple[float, ...]) -> None:
    with pytest.raises(VectorValidationError):
        VectorValidator(3).validate(vector)


def test_vector_validator_returns_immutable_finite_vector() -> None:
    vector = VectorValidator(3).validate((1, 2.5, 3))
    assert vector == (1.0, 2.5, 3.0)
    assert isinstance(vector, tuple)


def test_engine_rejects_invalid_provider_vector() -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.embed_batch.return_value = ((1.0, nan, 3.0),)
    engine = make_engine(provider)
    with pytest.raises(VectorValidationError):
        engine.embed(make_payload())
