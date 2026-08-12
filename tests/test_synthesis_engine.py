"""Tests for Phase 9 context assembly."""

from dataclasses import FrozenInstanceError, replace

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts import RetrievalBudget, RerankingPolicy
from proposal_ai_agent.online.contracts.retrieval import (
    ProcessedCandidate,
    RetrievedContext,
)
from proposal_ai_agent.online.contracts.synthesis import AssembledContext, Citation
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine


def _request():
    query_engine = QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )
    user_query = query_engine.receive_query(
        "What applies? document: handbook, department: legal, version: v2"
    )
    return query_engine.plan_retrieval(
        query_engine.embed_query(query_engine.process_query(query_engine.qualify_query(user_query)))
    )


def _processed_candidate(
    *,
    chunk_id: str,
    document_id: str = "document-1",
    text: str,
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


def test_synthesis_engine_enforces_token_budget_and_truncates_last_chunk() -> None:
    request = replace(
        _request(),
        retrieval_budget=RetrievalBudget(max_candidates=2, max_latency_ms=1_000, max_context_tokens=5),
    )
    candidates = (
        _processed_candidate(chunk_id="chunk-1", text="one two three four"),
        _processed_candidate(chunk_id="chunk-2", text="five six seven"),
    )
    engine = SynthesisEngine()

    assembled = engine.assemble(_retrieved_context(request, *candidates))

    assert assembled.assembled_context == "one two three four\n\nfive"
    assert assembled.token_usage["budget_tokens"] == 5
    assert assembled.token_usage["used_tokens"] == 5
    assert assembled.token_usage["remaining_tokens"] == 0
    assert assembled.token_usage["truncated"] is True
    assert assembled.token_usage["last_truncated_citation_id"] == "c2"
    assert assembled.citations[1].truncated is True


def test_synthesis_engine_preserves_retrieval_order_and_metadata() -> None:
    request = replace(
        _request(),
        retrieval_budget=RetrievalBudget(max_candidates=3, max_latency_ms=1_000, max_context_tokens=100),
    )
    candidates = (
        _processed_candidate(
            chunk_id="chunk-1",
            document_id="document-1",
            text="First source text.",
            metadata={"page": 7, "document_title": "Handbook"},
            header_path=("Policies", "Retention"),
            chunk_index=0,
        ),
        _processed_candidate(
            chunk_id="chunk-2",
            document_id="document-2",
            text="Second source text.",
            metadata={"page": 3, "document_title": "Guidance"},
            header_path=("Overview",),
            chunk_index=1,
        ),
    )
    engine = SynthesisEngine()
    assembled = engine.assemble(_retrieved_context(request, *candidates))

    assert [citation.citation_id for citation in assembled.citations] == ["c1", "c2"]
    assert assembled.citations[0].document_id == "document-1"
    assert assembled.citations[1].metadata["source_metadata"]["document_title"] == "Guidance"
    assert assembled.metadata["sources"][0]["document_id"] == "document-1"
    assert assembled.metadata["sources"][1]["chunk_id"] == "chunk-2"
    assert assembled.context_statistics["total_chunks"] == 2
    assert assembled.context_statistics["merged_chunks"] == 0


def test_synthesis_engine_merges_already_merged_chunks_and_counts_statistics() -> None:
    request = replace(
        _request(),
        retrieval_budget=RetrievalBudget(max_candidates=1, max_latency_ms=1_000, max_context_tokens=100),
    )
    candidate = _processed_candidate(
        chunk_id="merged-chunk",
        text="First merged chunk. Second merged chunk.",
        processing_flags={
            "metadata_validated": True,
            "merged_chunk_count": 2,
            "merged_chunk_ids": ("chunk-a", "chunk-b"),
        },
    )
    engine = SynthesisEngine()
    assembled = engine.assemble(_retrieved_context(request, candidate))

    assert assembled.assembled_context == "First merged chunk. Second merged chunk."
    assert assembled.context_statistics["merged_chunks"] == 1
    assert assembled.citations[0].metadata["processing_flags"]["merged_chunk_count"] == 2


def test_synthesis_engine_rejects_empty_retrieved_context() -> None:
    request = replace(
        _request(),
        retrieval_budget=RetrievalBudget(max_candidates=1, max_latency_ms=1_000, max_context_tokens=100),
    )
    engine = SynthesisEngine()

    with pytest.raises(ValueError, match="assembled_context must not be empty"):
        engine.assemble(
            RetrievedContext(
                retrieval_request=request,
                candidates=(),
                final_context="",
                total_candidates=0,
                returned_candidates=0,
                reranking_applied=False,
                reranking_time_ms=0.0,
                retrieval_summary={},
            )
        )


def test_assembled_context_is_immutable_equal_and_defensively_copied() -> None:
    request = _request()
    candidate = _processed_candidate(chunk_id="chunk-1", text="Content")
    citation = Citation(
        citation_id="c1",
        chunk_id="chunk-1",
        document_id="document-1",
        header_path=("Policies",),
        chunk_index=0,
        token_count=1,
        page=None,
        truncated=False,
        metadata={"source_metadata": {"document_title": "Handbook"}},
    )
    assembled = AssembledContext(
        retrieved_context=_retrieved_context(request, candidate),
        assembled_context="Content",
        citations=(citation,),
        metadata={"sources": ({"citation_id": "c1"},)},
        context_statistics={"total_chunks": 1, "merged_chunks": 0, "total_tokens": 1, "total_sources": 1},
        token_usage={"budget_tokens": 100, "used_tokens": 1, "remaining_tokens": 99, "truncated": False, "last_truncated_citation_id": None},
    )
    with pytest.raises(FrozenInstanceError):
        assembled.assembled_context = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        assembled.token_usage["used_tokens"] = 2  # type: ignore[index]
    assert assembled == replace(assembled)
