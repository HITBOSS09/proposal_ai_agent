"""Provider-neutral conversion of proposal queries into retrieval requests."""

from __future__ import annotations

from proposal_ai_agent.online.contracts import (
    ACLPolicy,
    HybridSearchPolicy,
    QueryEmbedding,
    RerankingPolicy,
    RetrievalBudget,
    RetrievalRequest,
    RetrievalStrategy as OnlineRetrievalStrategy,
    SearchScope,
)
from proposal_ai_agent.online.engines import QueryEngine

from .retrieval_query import RetrievalStrategy, SectionRetrievalQuery


class ProposalRetrievalRequestBuilder:
    """Build immutable online retrieval requests from proposal-domain queries."""

    RETRIEVAL_PROFILE = "proposal-section"
    DEFAULT_MAX_LATENCY_MS = 1_000
    DEFAULT_MAX_CONTEXT_TOKENS = 4_000

    def __init__(
        self,
        query_engine: QueryEngine,
        *,
        search_scope: SearchScope | None = None,
        max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        if not isinstance(query_engine, QueryEngine):
            raise TypeError("query_engine must be a QueryEngine")
        if search_scope is not None and not isinstance(search_scope, SearchScope):
            raise TypeError("search_scope must be a SearchScope")
        if max_latency_ms <= 0 or max_context_tokens <= 0:
            raise ValueError("retrieval budget values must be positive")
        self._query_engine = query_engine
        self._search_scope = search_scope or SearchScope()
        self._max_latency_ms = max_latency_ms
        self._max_context_tokens = max_context_tokens

    def build(self, query: SectionRetrievalQuery) -> RetrievalRequest:
        """Embed one section query and map its policy into a retrieval request."""
        if not isinstance(query, SectionRetrievalQuery):
            raise TypeError("query must be a SectionRetrievalQuery")

        embedding = self._embed(query.query_text)
        strategy, hybrid_policy = self._strategy_policy(query.retrieval_strategy)
        return RetrievalRequest(
            query_embedding=embedding,
            metadata_filters=query.metadata_filters,
            search_scope=self._search_scope,
            retrieval_strategy=strategy,
            retrieval_profile=self.RETRIEVAL_PROFILE,
            top_k=query.max_results,
            candidate_budget=query.max_results,
            score_threshold=0.0,
            reranking_policy=(
                RerankingPolicy.CROSS_ENCODER if query.rerank_enabled else RerankingPolicy.DISABLED
            ),
            hybrid_policy=hybrid_policy,
            acl_policy=ACLPolicy.PLACEHOLDER,
            retrieval_budget=RetrievalBudget(
                max_candidates=query.max_results,
                max_latency_ms=self._max_latency_ms,
                max_context_tokens=self._max_context_tokens,
            ),
            planner_metadata={
                "proposal_section_id": query.section_id,
                "proposal_reference_type": query.reference_type.value,
            },
        )

    def _embed(self, query_text: str) -> QueryEmbedding:
        user_query = self._query_engine.receive_query(query_text)
        qualified_query = self._query_engine.qualify_query(user_query)
        processed_query = self._query_engine.process_query(qualified_query)
        return self._query_engine.embed_query(processed_query)

    @staticmethod
    def _strategy_policy(
        strategy: RetrievalStrategy,
    ) -> tuple[OnlineRetrievalStrategy, HybridSearchPolicy]:
        policies = {
            RetrievalStrategy.DENSE: (
                OnlineRetrievalStrategy.DENSE,
                HybridSearchPolicy.DENSE_ONLY,
            ),
            RetrievalStrategy.HYBRID: (
                OnlineRetrievalStrategy.HYBRID,
                HybridSearchPolicy.HYBRID,
            ),
            RetrievalStrategy.KEYWORD: (
                OnlineRetrievalStrategy.KEYWORD,
                HybridSearchPolicy.KEYWORD_ONLY,
            ),
        }
        return policies[strategy]
