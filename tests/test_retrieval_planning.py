"""Unit tests for online retrieval planning without retrieval execution."""

from dataclasses import FrozenInstanceError, replace

import pytest

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine, RetrievalRequest
from proposal_ai_agent.online.benchmarks import BenchmarkProfile, BenchmarkRegistry
from proposal_ai_agent.online.contracts import (
    ACLPolicy,
    HybridSearchPolicy,
    RerankingPolicy,
    RetrievalStrategy,
)


def _engine(registry: BenchmarkRegistry | None = None) -> QueryEngine:
    return QueryEngine(
        registry=registry,
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )


def _embedding(engine: QueryEngine, query: str):
    user_query = engine.receive_query(query)
    qualified = engine.qualify_query(user_query)
    processed = engine.process_query(qualified)
    return engine.embed_query(processed)


def test_plan_retrieval_generates_generic_filters_and_scope() -> None:
    engine = _engine()

    request = engine.plan_retrieval(
        _embedding(
            engine,
            "What are retention rules? document: handbook, department: legal, version: v2",
        )
    )

    assert isinstance(request, RetrievalRequest)
    assert request.metadata_filters == {
        "document": "handbook",
        "department": "legal",
        "version": "v2",
    }
    assert request.search_scope.document == "handbook"
    assert request.search_scope.version == "v2"
    assert request.search_scope.extensions == {"department": "legal"}


def test_plan_retrieval_resolves_profile_defaults_and_enums() -> None:
    profile = BenchmarkProfile(
        intent_id="RAG_QA",
        required_parameters=("question",),
        optional_parameters=("document", "department", "version"),
        defaults={
            "knowledge_base": "proposals",
            "collection": "proposal_chunks",
            "retrieval_profile": "strict-rag",
            "top_k": "7",
            "candidate_budget": "28",
            "max_candidates": "30",
            "max_latency_ms": "800",
            "max_context_tokens": "3500",
            "score_threshold": "0.42",
            "retrieval_strategy": "HYBRID",
            "reranking_policy": "CROSS_ENCODER",
            "hybrid_policy": "HYBRID",
        },
        is_default=True,
    )
    engine = _engine(BenchmarkRegistry((profile,)))

    request = engine.plan_retrieval(_embedding(engine, "What applies?"))

    assert request.search_scope.knowledge_base == "proposals"
    assert request.search_scope.collection == "proposal_chunks"
    assert request.retrieval_profile == "strict-rag"
    assert request.retrieval_strategy is RetrievalStrategy.HYBRID
    assert request.reranking_policy is RerankingPolicy.CROSS_ENCODER
    assert request.hybrid_policy is HybridSearchPolicy.HYBRID
    assert request.acl_policy is ACLPolicy.PLACEHOLDER
    assert request.top_k == 7
    assert request.candidate_budget == 28
    assert request.score_threshold == 0.42
    assert request.retrieval_budget.max_candidates == 30
    assert request.retrieval_budget.max_latency_ms == 800
    assert request.retrieval_budget.max_context_tokens == 3500
    assert request.planner_metadata["benchmark_id"] == "RAG_QA"


def test_retrieval_request_rejects_non_enum_policy_values() -> None:
    engine = _engine()
    request = engine.plan_retrieval(_embedding(engine, "What applies?"))

    with pytest.raises(TypeError, match="retrieval_strategy"):
        replace(request, retrieval_strategy="DENSE")  # type: ignore[arg-type]


def test_retrieval_request_is_immutable_and_defensively_copies_mappings() -> None:
    engine = _engine()
    request = engine.plan_retrieval(_embedding(engine, "What applies? document: handbook"))
    metadata = {"plan": {"owner": "engine-1"}}
    copied = replace(request, planner_metadata=metadata)
    metadata["plan"]["owner"] = "changed"

    with pytest.raises(FrozenInstanceError):
        copied.top_k = 10  # type: ignore[misc]
    with pytest.raises(TypeError):
        copied.metadata_filters["new"] = "value"  # type: ignore[index]

    assert copied.planner_metadata["plan"]["owner"] == "engine-1"


def test_retrieval_request_equality_is_value_based() -> None:
    engine = _engine()
    request = engine.plan_retrieval(_embedding(engine, "What applies?"))

    assert request == replace(request)


def test_retrieval_planning_regression_preserves_phases_one_to_four() -> None:
    engine = _engine()
    embedding = _embedding(engine, "What applies? document: handbook")
    before = (
        embedding.processed_query.normalized_query,
        embedding.processed_query.qualified_query.intent,
        embedding.vector,
    )

    request = engine.plan_retrieval(embedding)

    assert (
        request.query_embedding.processed_query.normalized_query,
        request.query_embedding.processed_query.qualified_query.intent,
        request.query_embedding.vector,
    ) == before
