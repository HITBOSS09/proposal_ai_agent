"""Tests for proposal query conversion into online retrieval requests."""

from dataclasses import FrozenInstanceError

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online.contracts import (
    HybridSearchPolicy,
    RerankingPolicy,
    RetrievalStrategy as OnlineRetrievalStrategy,
    SearchScope,
)
from proposal_ai_agent.online.engines import QueryEngine
from proposal_ai_agent.proposal_generation import (
    ProposalRetrievalRequestBuilder,
    ReferenceType,
    RetrievalStrategy,
    SectionRetrievalQuery,
)


def _builder(*, search_scope: SearchScope | None = None) -> ProposalRetrievalRequestBuilder:
    return ProposalRetrievalRequestBuilder(
        _query_engine(),
        search_scope=search_scope,
    )


def _query_engine() -> QueryEngine:
    return QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="proposal-test-embedding",
    )


def _query(
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE, *, rerank_enabled: bool = False
) -> SectionRetrievalQuery:
    return SectionRetrievalQuery(
        section_id="solution-overview",
        reference_type=ReferenceType.TECHNICAL,
        query_text="Northstar security requirements",
        metadata_filters={"industry": "energy", "approved": True},
        max_results=3,
        retrieval_strategy=strategy,
        rerank_enabled=rerank_enabled,
    )


def test_builder_constructs_request_with_query_embedding_and_scope() -> None:
    scope = SearchScope(knowledge_base="proposals", collection="proposal_chunks")

    request = _builder(search_scope=scope).build(_query())

    assert request.query_embedding.model_id == "proposal-test-embedding"
    assert request.query_embedding.vector
    assert request.top_k == request.candidate_budget == 3
    assert request.retrieval_budget.max_candidates == 3
    assert request.search_scope == scope


def test_builder_propagates_metadata_without_aliasing_input() -> None:
    filters = {"source_document": "reference.docx"}
    query = _query().model_copy(update={"metadata_filters": filters})

    request = _builder().build(query)
    filters["source_document"] = "changed.docx"

    assert request.metadata_filters == {"source_document": "reference.docx"}
    with pytest.raises(TypeError):
        request.metadata_filters["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("proposal_strategy", "expected_strategy", "expected_hybrid_policy"),
    [
        (RetrievalStrategy.DENSE, OnlineRetrievalStrategy.DENSE, HybridSearchPolicy.DENSE_ONLY),
        (RetrievalStrategy.HYBRID, OnlineRetrievalStrategy.HYBRID, HybridSearchPolicy.HYBRID),
        (RetrievalStrategy.KEYWORD, OnlineRetrievalStrategy.KEYWORD, HybridSearchPolicy.KEYWORD_ONLY),
    ],
)
def test_builder_maps_retrieval_strategies(
    proposal_strategy: RetrievalStrategy,
    expected_strategy: OnlineRetrievalStrategy,
    expected_hybrid_policy: HybridSearchPolicy,
) -> None:
    request = _builder().build(_query(proposal_strategy))

    assert request.retrieval_strategy is expected_strategy
    assert request.hybrid_policy is expected_hybrid_policy


def test_builder_maps_reranking_policy() -> None:
    assert _builder().build(_query(rerank_enabled=False)).reranking_policy is RerankingPolicy.DISABLED
    assert _builder().build(_query(rerank_enabled=True)).reranking_policy is RerankingPolicy.CROSS_ENCODER


def test_builder_rejects_invalid_configuration() -> None:
    with pytest.raises(TypeError, match="query_engine"):
        ProposalRetrievalRequestBuilder(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget"):
        ProposalRetrievalRequestBuilder(_query_engine(), max_latency_ms=0)


def test_builder_output_is_immutable() -> None:
    request = _builder().build(_query())

    with pytest.raises(FrozenInstanceError):
        request.top_k = 4  # type: ignore[misc]
