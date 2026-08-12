"""Tests for the StubLLMProvider integration with the router and response engine."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts.response import GeneratedResponse
from proposal_ai_agent.online.contracts.retrieval import ProcessedCandidate, RetrievedContext
from proposal_ai_agent.online.contracts.synthesis import PromptPackage
from proposal_ai_agent.online.engines.response_engine import ResponseEngine
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine
from proposal_ai_agent.online.providers.llm.stub import StubLLMProvider
from proposal_ai_agent.online.providers.llm.router import LLMRouter, NoAvailableProviderError


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


def test_router_selects_stub_provider_and_response_is_deterministic() -> None:
    request = _request()
    prompt = _prompt_package(request)
    stub = StubLLMProvider()
    router = LLMRouter((stub,))

    selected = router.route(prompt)
    assert selected.provider_name == "stub"

    response = router.generate(prompt)
    assert response.provider_name == "stub"
    assert "stub" in response.generated_text


def test_response_engine_delegates_to_stub_once() -> None:
    request = _request()
    prompt = _prompt_package(request)
    stub = StubLLMProvider()
    router = LLMRouter((stub,))
    engine = ResponseEngine(router)

    response = engine.generate(prompt)
    assert response.provider_name == "stub"
    assert stub.generated_calls == 1


def test_stub_generated_response_is_immutable_and_defensive() -> None:
    request = _request()
    prompt = _prompt_package(request)
    stub = StubLLMProvider()

    response = stub.generate(prompt)
    # equality and defensive copying
    assert response == replace(response)
    with pytest.raises(AttributeError):
        response.generated_text = "changed"

    with pytest.raises(TypeError):
        response.generation_metadata["foo"] = "value"  # type: ignore[index]


def test_router_raises_when_stub_unavailable() -> None:
    request = _request()
    prompt = _prompt_package(request)
    stub = StubLLMProvider(available=False)
    router = LLMRouter((stub,))

    with pytest.raises(NoAvailableProviderError):
        router.route(prompt)
