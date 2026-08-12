"""Tests for Phase 11 LLM routing and generation."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts.response import GeneratedResponse
from proposal_ai_agent.online.contracts.retrieval import ProcessedCandidate, RetrievedContext
from proposal_ai_agent.online.contracts.synthesis import PromptPackage
from proposal_ai_agent.online.engines.response_engine import ResponseEngine
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine
from proposal_ai_agent.online.providers.llm.factory import LLMProviderFactory
from proposal_ai_agent.online.providers.llm.provider import (
    LLMProvider,
    ProviderUnavailableError,
    LLMGenerationError,
)
from proposal_ai_agent.online.providers.llm.router import LLMRouter, NoAvailableProviderError


class DummyLLMProvider:
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        provider_metadata: dict | None = None,
        available: bool = True,
        fail_on_generate: bool = False,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._provider_metadata = provider_metadata or {}
        self._available = available
        self._fail_on_generate = fail_on_generate
        self.generated_calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_metadata(self) -> dict:
        return self._provider_metadata

    def health_check(self) -> bool:
        return self._available

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        self.generated_calls += 1
        if not self._available:
            raise ProviderUnavailableError("provider unavailable")
        if self._fail_on_generate:
            raise LLMGenerationError("generation failed")
        return GeneratedResponse(
            generated_text=f"response from {self.provider_name}",
            provider_name=self.provider_name,
            model_name=self.model_name,
            finish_reason="completed",
            token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            latency_ms=12.5,
            generation_timestamp=datetime.now(timezone.utc),
            generation_metadata={"provider_metadata": self.provider_metadata},
        )


def _request():
    query_engine = QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )
    user_query = query_engine.receive_query("What applies? document: handbook, department: legal, version: v2")
    return query_engine.plan_retrieval(
        query_engine.embed_query(query_engine.process_query(query_engine.qualify_query(user_query)))
    )


def _processed_candidate(*, chunk_id: str, text: str) -> ProcessedCandidate:
    return ProcessedCandidate(
        chunk_id=chunk_id,
        document_id="document-1",
        text=text,
        score=0.5,
        metadata={"page": 1},
        header_path=("Policies",),
        chunk_index=0,
        processing_flags={"metadata_validated": True, "merged_chunk_count": 1},
    )


def _retrieved_context(request, *candidates):
    return RetrievedContext(
        retrieval_request=request,
        candidates=candidates,
        final_context="\n\n".join(candidate.text for candidate in candidates),
        total_candidates=len(candidates),
        returned_candidates=len(candidates),
        reranking_applied=False,
        reranking_time_ms=0.0,
        retrieval_summary={
            "processing_summary": {},
            "candidate_budget": request.candidate_budget,
            "top_k": request.top_k,
            "reranking_pool_size": 0,
        },
    )


def _prompt_package(request):
    assembled = SynthesisEngine().assemble(_retrieved_context(request, _processed_candidate(chunk_id="chunk-1", text="Content")))
    return PromptPackage(
        system_prompt="Enterprise assistant.",
        user_prompt=f"Question: {request.query_embedding.processed_query.normalized_query}",
        conversation_history=("User: Hello",),
        assembled_context=assembled,
        prompt_template="{system_prompt}\n{context_block}\n{user_prompt}",
        output_format="markdown",
        prompt_statistics={"system_tokens": 1, "user_tokens": 2, "history_tokens": 3, "context_tokens": 4, "total_tokens": 10},
        generation_metadata={"benchmark_id": request.query_embedding.processed_query.qualified_query.benchmark_id},
        validation_result=request.query_embedding.processed_query.qualified_query.validation_result,
    )


def test_generated_response_is_immutable_equal_and_defensively_copied() -> None:
    response = GeneratedResponse(
        generated_text="Answer",
        provider_name="provider-a",
        model_name="model-x",
        finish_reason="completed",
        token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        latency_ms=5.0,
        generation_timestamp=datetime.now(timezone.utc),
        generation_metadata={"foo": "bar"},
    )
    metadata = {"foo": "bar"}
    response = GeneratedResponse(
        generated_text="Answer",
        provider_name="provider-a",
        model_name="model-x",
        finish_reason="completed",
        token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        latency_ms=5.0,
        generation_timestamp=datetime.now(timezone.utc),
        generation_metadata=metadata,
    )
    metadata["foo"] = "changed"

    assert response == replace(response)
    with pytest.raises(FrozenInstanceError):
        response.generated_text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        response.generation_metadata["foo"] = "value"  # type: ignore[index]


def test_llm_router_selects_preferred_provider_and_falls_back() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider_a = DummyLLMProvider(
        provider_name="alpha",
        model_name="a1",
        provider_metadata={"priority": 10},
        available=True,
    )
    provider_b = DummyLLMProvider(
        provider_name="beta",
        model_name="b1",
        provider_metadata={"priority": 20},
        available=True,
    )
    router = LLMRouter((provider_a, provider_b))

    assert router.route(prompt).provider_name == "alpha"

    prompt_preferred = replace(prompt, generation_metadata={**prompt.generation_metadata, "preferred_provider": "beta"})
    assert router.route(prompt_preferred).provider_name == "beta"

    provider_a_unavailable = DummyLLMProvider(
        provider_name="alpha",
        model_name="a1",
        provider_metadata={"priority": 10},
        available=False,
    )
    router_fallback = LLMRouter((provider_a_unavailable, provider_b))
    assert router_fallback.route(prompt).provider_name == "beta"


def test_llm_router_uses_fallback_when_generation_fails() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider_a = DummyLLMProvider(
        provider_name="alpha",
        model_name="a1",
        provider_metadata={"priority": 10},
        available=True,
        fail_on_generate=True,
    )
    provider_b = DummyLLMProvider(
        provider_name="beta",
        model_name="b1",
        provider_metadata={"priority": 20},
        available=True,
    )
    router = LLMRouter((provider_a, provider_b))

    response = router.generate(prompt)

    assert response.provider_name == "beta"
    assert provider_a.generated_calls == 1
    assert provider_b.generated_calls == 1


def test_llm_provider_factory_registers_and_resolves_providers() -> None:
    factory = LLMProviderFactory()

    class MinimalProvider(DummyLLMProvider):
        pass

    factory.register_provider("alpha", MinimalProvider)
    provider = factory.create("alpha", provider_name="alpha", model_name="a1")

    assert provider.provider_name == "alpha"
    providers = factory.supported_providers()
    assert providers[:2] == ("stub", "ollama")
    assert "alpha" in providers


def test_response_engine_delegates_to_router() -> None:
    request = _request()
    prompt = _prompt_package(request)

    class RecordingRouter:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
            self.calls.append(prompt_package)
            return GeneratedResponse(
                generated_text="Answer",
                provider_name="alpha",
                model_name="a1",
                finish_reason="completed",
                token_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                latency_ms=1.0,
                generation_timestamp=datetime.now(timezone.utc),
                generation_metadata={"router": "recording"},
            )

    router = RecordingRouter()
    engine = ResponseEngine(router)

    result = engine.generate(prompt)

    assert result.provider_name == "alpha"
    assert router.calls == [prompt]


def test_router_raises_when_no_providers_are_available() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = DummyLLMProvider(
        provider_name="alpha",
        model_name="a1",
        provider_metadata={"priority": 10},
        available=False,
    )
    router = LLMRouter((provider,))

    with pytest.raises(NoAvailableProviderError):
        router.route(prompt)
