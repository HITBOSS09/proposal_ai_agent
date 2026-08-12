"""Qdrant-backed implementation of the proposal reference-retrieval port."""

from __future__ import annotations

from dataclasses import replace

from proposal_ai_agent.indexing import DocumentRole
from proposal_ai_agent.online.repositories import QdrantVectorRepository

from ..prompt_composer import RetrievedReference
from ..retrieval_query import RetrievalStrategy, SectionRetrievalQuery
from ..retrieval_request_builder import ProposalRetrievalRequestBuilder


class QdrantProposalRetriever:
    """Retrieve proposal references through the certified Qdrant repository adapter."""

    def __init__(
        self,
        request_builder: ProposalRetrievalRequestBuilder,
        repository: QdrantVectorRepository,
    ) -> None:
        if not isinstance(request_builder, ProposalRetrievalRequestBuilder):
            raise TypeError("request_builder must be a ProposalRetrievalRequestBuilder")
        if not isinstance(repository, QdrantVectorRepository):
            raise TypeError("repository must be a QdrantVectorRepository")
        self._request_builder = request_builder
        self._repository = repository

    def retrieve(self, query: SectionRetrievalQuery) -> tuple[RetrievedReference, ...]:
        """Retrieve ordered references for one proposal section query."""
        if not isinstance(query, SectionRetrievalQuery):
            raise TypeError("query must be a SectionRetrievalQuery")
        if query.retrieval_strategy is not RetrievalStrategy.DENSE:
            raise ValueError(
                f"QdrantProposalRetriever does not support '{query.retrieval_strategy.value}' retrieval"
            )

        mandatory_role = DocumentRole.REFERENCE_KNOWLEDGE.value
        caller_role = query.metadata_filters.get("document_role")
        if caller_role is not None and caller_role != mandatory_role:
            raise ValueError(
                "proposal retrieval requires document_role=REFERENCE_KNOWLEDGE"
            )

        retrieval_request = self._request_builder.build(query)
        authorized_filters = dict(retrieval_request.metadata_filters)
        authorized_filters["document_role"] = mandatory_role
        retrieval_request = replace(
            retrieval_request,
            metadata_filters=authorized_filters,
        )
        try:
            result = self._repository.search(retrieval_request)
        except Exception as error:
            raise RuntimeError("proposal reference retrieval failed") from error

        return tuple(
            RetrievedReference(
                reference_id=candidate.point_id or candidate.chunk_id,
                reference_type=query.reference_type.value,
                chunk_id=candidate.chunk_id,
                source_document=candidate.metadata["source_document"],
                score=candidate.score,
                content=candidate.text,
                metadata=candidate.metadata,
            )
            for candidate in result.candidates
        )
