"""Tests for the final response builder phase."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts.response import AnswerResponse, GeneratedResponse
from proposal_ai_agent.online.contracts.retrieval import ProcessedCandidate, RetrievedContext
from proposal_ai_agent.online.contracts.synthesis import PromptPackage
from proposal_ai_agent.online.engines.response_builder import ResponseBuilder
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine
from proposal_ai_agent.online.providers.llm.stub import StubLLMProvider


def _request():
    query_engine = QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )
    user_query = query_engine.receive_query("What applies? document: handbook")
    return query_engine.plan_retrieval(
        query_engine.embed_query(query_engine.process_query(query_engine.qualify_query(user_query)))
    )


def _processed_candidate(*, chunk_id: str, text: str) -> ProcessedCandidate:
    return ProcessedCandidate(
        chunk_id=chunk_id,
        document_id="document-1",
        text=text,
        score=0.5,
        metadata={"page": 1, "document_title": "Handbook"},
        header_path=("Policies", "Retention"),
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
    assembled = SynthesisEngine().assemble(
        _retrieved_context(request, _processed_candidate(chunk_id="chunk-1", text="Content"))
    )
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


def test_response_builder_constructs_answer_response() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = StubLLMProvider()
    generated = provider.generate(prompt)

    response = ResponseBuilder().build_response(prompt, generated)

    assert isinstance(response, AnswerResponse)
    assert response.final_answer == generated.generated_text
    assert response.response_metadata.provider_name == generated.provider_name
    assert response.response_metadata.total_tokens == generated.token_usage["total_tokens"]
    assert response.audit_trace.request_id == request.query_embedding.processed_query.qualified_query.original.request_id
    assert response.audit_trace.query_hash == request.query_embedding.processed_query.query_hash
    assert response.audit_trace.provider == generated.provider_name
    assert len(response.citations) == len(prompt.assembled_context.citations)
    assert response.citations[0].document_name == "Handbook"
    assert response.citations[0].section == "Retention"


def test_answer_response_is_immutable_and_equal() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = StubLLMProvider()
    generated = provider.generate(prompt)

    response = ResponseBuilder().build_response(prompt, generated)
    assert response == replace(response)
    with pytest.raises(AttributeError):
        response.final_answer = "changed"  # type: ignore[misc]


def test_response_builder_rejects_invalid_inputs() -> None:
    builder = ResponseBuilder()
    with pytest.raises(TypeError):
        builder.build_response("not a prompt", GeneratedResponse(
            generated_text="Answer",
            provider_name="stub",
            model_name="stub",
            finish_reason="completed",
            token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_ms=1.0,
            generation_timestamp=datetime.now(timezone.utc),
            generation_metadata={"provider": "stub"},
        ))

    request = _request()
    prompt = _prompt_package(request)
    with pytest.raises(TypeError):
        builder.build_response(prompt, "not a response")
