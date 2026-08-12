"""Tests for Phase 6 knowledge-retrieval execution."""

from dataclasses import FrozenInstanceError, replace

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts import RerankingPolicy, RetrievalStrategy
from proposal_ai_agent.online.contracts.retrieval import (
    ProcessedCandidate,
    ProcessedCandidates,
    RetrievedCandidate,
    RetrievedCandidates,
    RetrievedContext,
)
from proposal_ai_agent.online.engines.retrieval_engine import (
    RepositorySearchResult,
    RetrievalEngine,
)
from proposal_ai_agent.online.providers.reranking.provider import RerankingScore


class RecordingRepository:
    """Repository fake proving the engine delegates the whole frozen request."""

    def __init__(self, result: RepositorySearchResult) -> None:
        self.result = result
        self.requests = []

    def search(self, retrieval_request):
        self.requests.append(retrieval_request)
        return self.result


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


def _candidate(
    metadata=None,
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "document-1",
    text: str = "Retention rules apply.",
    score: float = 0.91,
    header_path: tuple[str, ...] = ("Policies", "Retention"),
    chunk_index: int = 4,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        metadata={} if metadata is None else metadata,
        header_path=header_path,
        chunk_index=chunk_index,
    )


def _retrieved(request, *candidates: RetrievedCandidate) -> RetrievedCandidates:
    return RetrievedCandidates(
        retrieval_request=request,
        candidates=candidates,
        retrieval_time_ms=1.0,
        candidate_count=len(candidates),
        repository_metadata={"collection": "proposal_chunks"},
    )


def _processed_candidate(
    *,
    chunk_id: str,
    text: str,
    score: float = 0.5,
    chunk_index: int = 0,
    metadata=None,
) -> ProcessedCandidate:
    return ProcessedCandidate(
        chunk_id=chunk_id,
        document_id="document-1",
        text=text,
        score=score,
        metadata={} if metadata is None else metadata,
        header_path=("Policies",),
        chunk_index=chunk_index,
        processing_flags={"metadata_validated": True, "merged_chunk_count": 1},
    )


def _processed(request, *candidates: ProcessedCandidate) -> ProcessedCandidates:
    return ProcessedCandidates(
        retrieval_request=request,
        candidates=candidates,
        processing_summary={"output_count": len(candidates)},
        processing_time_ms=1.0,
    )


def test_retrieval_executes_the_unchanged_request_and_preserves_candidates() -> None:
    request = _request()
    candidate = _candidate()
    repository = RecordingRepository(
        RepositorySearchResult((candidate,), {"collection": "proposal_chunks"})
    )

    result = RetrievalEngine(repository).retrieve(request)

    assert repository.requests == [request]
    assert repository.requests[0].metadata_filters == {
        "document": "handbook",
        "department": "legal",
        "version": "v2",
    }
    assert repository.requests[0].search_scope.document == "handbook"
    assert repository.requests[0].search_scope.extensions == {"department": "legal"}
    assert repository.requests[0].retrieval_strategy is RetrievalStrategy.DENSE
    assert result.retrieval_request is request
    assert result.candidates == (candidate,)
    assert result.candidate_count == 1
    assert result.repository_metadata == {"collection": "proposal_chunks"}
    assert result.retrieval_time_ms >= 0


def test_retrieval_preserves_an_empty_repository_result() -> None:
    repository = RecordingRepository(RepositorySearchResult((), {"collection": "empty"}))

    result = RetrievalEngine(repository).retrieve(_request())

    assert result.candidates == ()
    assert result.candidate_count == 0


def test_retrieval_dtos_are_immutable_equal_and_defensively_copied() -> None:
    metadata = {"source": {"tags": ["policy"]}}
    repository_metadata = {"shard": {"name": "a"}}
    candidate = _candidate(metadata)
    repository = RecordingRepository(RepositorySearchResult((candidate,), repository_metadata))
    result = RetrievalEngine(repository).retrieve(_request())
    metadata["source"]["tags"].append("changed")
    repository_metadata["shard"]["name"] = "changed"

    assert candidate.metadata["source"]["tags"] == ("policy",)
    assert result.repository_metadata["shard"]["name"] == "a"
    assert result == replace(result)
    with pytest.raises(FrozenInstanceError):
        candidate.text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.repository_metadata["shard"] = {}  # type: ignore[index]


def test_retrieval_rejects_non_repository_results() -> None:
    class InvalidRepository:
        def search(self, retrieval_request):
            return ()

    with pytest.raises(TypeError, match="RepositorySearchResult"):
        RetrievalEngine(InvalidRepository()).retrieve(_request())


def test_candidate_processing_removes_exact_and_structural_near_duplicates() -> None:
    request = _request()
    first = _candidate(metadata={"content_hash": "first"})
    exact_duplicate = _candidate(metadata={"content_hash": "first"})
    same_document_location = _candidate(
        chunk_id="different-id", metadata={"content_hash": "second"}
    )
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})))

    result = engine.process(_retrieved(request, first, exact_duplicate, same_document_location))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-1"]
    assert result.processing_summary["exact_duplicates_removed"] == 1
    assert result.processing_summary["near_duplicates_removed"] == 1


def test_candidate_processing_merges_adjacent_chunks_without_reordering_documents() -> None:
    request = _request()
    first = _candidate(
        metadata={"document_id": "document-1", "chunk_index": 4, "section_path": ["Policies", "Retention"]}
    )
    adjacent = _candidate(
        chunk_id="chunk-2",
        text="Records must be retained.",
        metadata={"document_id": "document-1", "chunk_index": 5, "section_path": ["Policies", "Retention"]},
        chunk_index=5,
    )
    next_document = _candidate(
        chunk_id="chunk-3",
        document_id="document-2",
        text="Other document.",
        metadata={"document_id": "document-2", "chunk_index": 0, "section_path": ["Overview"]},
        header_path=("Overview",),
        chunk_index=0,
    )
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})))

    result = engine.process(_retrieved(request, first, adjacent, next_document))

    assert [candidate.document_id for candidate in result.candidates] == ["document-1", "document-2"]
    assert result.candidates[0].text == "Retention rules apply.\n\nRecords must be retained."
    assert result.candidates[0].header_path == ("Policies", "Retention")
    assert result.candidates[0].score == first.score
    assert result.candidates[0].processing_flags["merged_chunk_ids"] == ("chunk-1", "chunk-2")
    assert result.processing_summary["adjacent_merges"] == 1


def test_candidate_processing_rejects_invalid_metadata_and_low_scores() -> None:
    request = replace(_request(), score_threshold=0.8)
    invalid = _candidate(metadata={"document_id": "wrong-document"})
    below_threshold = _candidate(
        chunk_id="chunk-2", chunk_index=5, score=0.79, metadata={"content_hash": "low"}
    )
    accepted = _candidate(
        chunk_id="chunk-3", chunk_index=6, score=0.8, metadata={"content_hash": "kept"}
    )
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})))

    result = engine.process(_retrieved(request, invalid, below_threshold, accepted))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-3"]
    assert result.processing_summary["invalid_metadata_removed"] == 1
    assert result.processing_summary["score_threshold_removed"] == 1


def test_processed_dtos_are_immutable_defensive_and_have_no_vector_field() -> None:
    metadata = {"source": {"tags": ["policy"]}, "vector": [0.1, 0.2]}
    flags = {"merged_chunk_ids": ["chunk-1"]}
    direct = ProcessedCandidate(
        chunk_id="chunk-1",
        document_id="document-1",
        text="Content",
        score=0.5,
        metadata=metadata,
        header_path=("Policies",),
        chunk_index=1,
        processing_flags=flags,
    )
    metadata["source"]["tags"].append("changed")
    flags["merged_chunk_ids"].append("changed")
    result = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {}))).process(
        _retrieved(_request(), _candidate(metadata={"content_hash": "kept"}))
    )

    assert direct.metadata["source"]["tags"] == ("policy",)
    assert direct.processing_flags["merged_chunk_ids"] == ("chunk-1",)
    assert direct == replace(direct)
    assert not hasattr(result.candidates[0], "vector")
    with pytest.raises(FrozenInstanceError):
        direct.text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        direct.metadata["new"] = "value"  # type: ignore[index]


def test_candidate_processing_does_not_call_the_phase_six_repository() -> None:
    repository = RecordingRepository(RepositorySearchResult((), {}))
    engine = RetrievalEngine(repository)
    retrieved = _retrieved(_request(), _candidate(metadata={"content_hash": "kept"}))

    result = engine.process(retrieved)

    assert repository.requests == []
    assert result.retrieval_request is retrieved.retrieval_request


def test_reranking_disabled_preserves_order_and_does_not_invoke_provider() -> None:
    class FailingProvider:
        def rerank(self, query, candidates):
            raise AssertionError("disabled reranking must not call the provider")

    request = replace(_request(), top_k=2, candidate_budget=3)
    candidates = (
        _processed_candidate(chunk_id="chunk-1", text="First"),
        _processed_candidate(chunk_id="chunk-2", text="Second", chunk_index=1),
        _processed_candidate(chunk_id="chunk-3", text="Third", chunk_index=2),
    )
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})), FailingProvider())

    result = engine.rerank(_processed(request, *candidates))

    assert result.candidates == candidates[:2]
    assert result.final_context == "First\n\nSecond"
    assert result.reranking_applied is False
    assert result.reranking_time_ms == 0.0


def test_reranking_uses_only_candidate_budget_and_returns_top_k() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.calls = []

        def rerank(self, query, candidates):
            self.calls.append((query, tuple(candidates)))
            return tuple(
                RerankingScore(index, relevance)
                for index, relevance in enumerate((0.1, 0.9, 0.5))
            )

    request = replace(
        _request(), reranking_policy=RerankingPolicy.CROSS_ENCODER, top_k=2, candidate_budget=3
    )
    candidates = tuple(
        _processed_candidate(chunk_id=f"chunk-{index}", text=f"Candidate {index}", chunk_index=index)
        for index in range(4)
    )
    reranker = RecordingReranker()
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})), reranker)

    result = engine.rerank(_processed(request, *candidates))

    assert reranker.calls == [
        (request.query_embedding.processed_query.normalized_query, candidates[:3])
    ]
    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-1", "chunk-2"]
    assert result.reranking_applied is True
    assert result.total_candidates == 4
    assert result.returned_candidates == 2
    assert result.retrieval_summary["reranking_pool_size"] == 3


def test_reranking_preserves_processed_candidate_content_and_metadata() -> None:
    class ReverseReranker:
        def rerank(self, query, candidates):
            return tuple(RerankingScore(index, -index) for index in range(len(candidates)))

    metadata = {"citation": {"page": 7}}
    candidate = _processed_candidate(chunk_id="chunk-1", text="Unchanged", metadata=metadata)
    request = replace(_request(), reranking_policy=RerankingPolicy.CROSS_ENCODER)
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})), ReverseReranker())

    result = engine.rerank(_processed(request, candidate))
    metadata["citation"]["page"] = 8

    assert result.candidates[0] is candidate
    assert result.candidates[0].text == "Unchanged"
    assert result.candidates[0].metadata["citation"]["page"] == 7
    assert result.candidates[0].processing_flags["metadata_validated"] is True


def test_retrieved_context_is_immutable_equal_and_defensively_copied() -> None:
    summary = {"nested": {"value": "original"}}
    candidate = _processed_candidate(chunk_id="chunk-1", text="Content")
    context = RetrievedContext(
        retrieval_request=_request(),
        candidates=(candidate,),
        final_context="Content",
        total_candidates=1,
        returned_candidates=1,
        reranking_applied=False,
        reranking_time_ms=0.0,
        retrieval_summary=summary,
    )
    summary["nested"]["value"] = "changed"

    assert context.retrieval_summary["nested"]["value"] == "original"
    assert context == replace(context)
    with pytest.raises(FrozenInstanceError):
        context.final_context = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.retrieval_summary["new"] = "value"  # type: ignore[index]


def test_reranking_rejects_incomplete_provider_scores() -> None:
    class IncompleteReranker:
        def rerank(self, query, candidates):
            return (RerankingScore(0, 1.0),)

    request = replace(_request(), reranking_policy=RerankingPolicy.CROSS_ENCODER)
    engine = RetrievalEngine(RecordingRepository(RepositorySearchResult((), {})), IncompleteReranker())

    with pytest.raises(ValueError, match="score every candidate"):
        engine.rerank(
            _processed(
                request,
                _processed_candidate(chunk_id="chunk-1", text="First"),
                _processed_candidate(chunk_id="chunk-2", text="Second", chunk_index=1),
            )
        )
