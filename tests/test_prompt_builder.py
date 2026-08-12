"""Tests for Phase 10 prompt construction."""

from dataclasses import FrozenInstanceError, replace

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.benchmarks.registry import BenchmarkProfile, BenchmarkRegistry
from proposal_ai_agent.online.contracts import RerankingPolicy
from proposal_ai_agent.online.contracts.retrieval import ProcessedCandidate, RetrievedContext
from proposal_ai_agent.online.contracts.synthesis import PromptPackage
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine


def _request(conversation_history=None):
    query_engine = QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )
    user_query = query_engine.receive_query(
        "What applies? document: handbook, department: legal, version: v2",
        conversation_history=conversation_history,
    )
    return query_engine.plan_retrieval(
        query_engine.embed_query(query_engine.process_query(query_engine.qualify_query(user_query)))
    )


def _processed_candidate(
    *,
    chunk_id: str,
    text: str,
    document_id: str = "document-1",
    score: float = 0.5,
    header_path: tuple[str, ...] = ("Policies",),
    chunk_index: int = 0,
    metadata=None,
    processing_flags=None,
) -> ProcessedCandidate:
    return ProcessedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        metadata={} if metadata is None else metadata,
        header_path=header_path,
        chunk_index=chunk_index,
        processing_flags={
            "metadata_validated": True,
            "merged_chunk_count": 1,
            **({} if processing_flags is None else processing_flags),
        },
    )


def _retrieved_context(request, *candidates: ProcessedCandidate) -> RetrievedContext:
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


def test_prompt_builder_constructs_prompt_package_with_context_and_history() -> None:
    request = _request(
        conversation_history=(
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        )
    )
    candidates = (
        _processed_candidate(chunk_id="chunk-1", text="First source text."),
        _processed_candidate(chunk_id="chunk-2", text="Second source text."),
    )
    retrieved_context = _retrieved_context(request, *candidates)
    synthesis_engine = SynthesisEngine()
    assembled_context = synthesis_engine.assemble(retrieved_context)
    prompt = synthesis_engine.build_prompt(assembled_context)

    assert "enterprise knowledge assistant" in prompt.system_prompt
    assert prompt.prompt_template.startswith("{system_prompt}")
    assert prompt.output_format == "markdown"
    assert prompt.conversation_history == (
        "User: Hello",
        "Assistant: Hi, how can I help?",
    )
    assert "Context:" in prompt.prompt_template or "context_block" in prompt.prompt_template
    assert prompt.assembled_context is assembled_context
    assert prompt.validation_result == request.query_embedding.processed_query.qualified_query.validation_result
    assert prompt.prompt_statistics["total_tokens"] == (
        prompt.prompt_statistics["system_tokens"]
        + prompt.prompt_statistics["user_tokens"]
        + prompt.prompt_statistics["history_tokens"]
        + prompt.prompt_statistics["context_tokens"]
    )


def test_prompt_builder_resolves_profile_template_and_output_format() -> None:
    custom_template = "SYSTEM:\n{system_prompt}\nHISTORY:\n{conversation_history}\nCONTEXT:\n{context_block}\nUSER:\n{user_prompt}"
    profile = BenchmarkProfile(
        intent_id="RAG_QA",
        required_parameters=("question",),
        optional_parameters=("document", "department", "version"),
        defaults={"prompt_template": custom_template, "output_format": "json"},
    )
    registry = BenchmarkRegistry((profile,))
    request = _request()
    candidates = (_processed_candidate(chunk_id="chunk-1", text="Only source text."),)
    retrieved_context = _retrieved_context(request, *candidates)
    synthesis_engine = SynthesisEngine(registry=registry)
    assembled_context = synthesis_engine.assemble(retrieved_context)
    prompt = synthesis_engine.build_prompt(assembled_context)

    assert prompt.prompt_template == custom_template
    assert prompt.output_format == "json"
    assert "Respond in json format" in prompt.system_prompt
    assert prompt.generation_metadata["benchmark_id"] == "RAG_QA"


def test_prompt_package_is_immutable_and_defensively_copied() -> None:
    request = _request()
    candidate = _processed_candidate(chunk_id="chunk-1", text="Content")
    retrieved_context = _retrieved_context(request, candidate)
    synthesis_engine = SynthesisEngine()
    assembled_context = synthesis_engine.assemble(retrieved_context)
    prompt = synthesis_engine.build_prompt(assembled_context)

    assert prompt == replace(prompt)
    with pytest.raises(FrozenInstanceError):
        prompt.system_prompt = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        prompt.prompt_statistics["system_tokens"] = 999  # type: ignore[index]
