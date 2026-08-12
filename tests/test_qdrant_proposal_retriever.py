"""Tests for Qdrant-to-proposal reference adaptation."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.indexing import DocumentRole
from proposal_ai_agent.online.contracts import SearchScope
from proposal_ai_agent.online.contracts.retrieval import RetrievedCandidate
from proposal_ai_agent.online.engines import QueryEngine
from proposal_ai_agent.online.engines.retrieval_engine import RepositorySearchResult
from proposal_ai_agent.online.repositories import QdrantVectorRepository
from proposal_ai_agent.proposal_generation import (
    ProposalRetrievalRequestBuilder,
    ReferenceType,
    RetrievalStrategy,
    SectionRetrievalQuery,
)
from proposal_ai_agent.proposal_generation.retrievers import QdrantProposalRetriever


class FakeQdrantClient:
    """Unused client because repository.search is replaced for adapter tests."""


def _query(
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE,
    metadata_filters=None,
) -> SectionRetrievalQuery:
    return SectionRetrievalQuery(
        section_id="solution-overview",
        reference_type=ReferenceType.TECHNICAL,
        query_text="Northstar security requirements",
        metadata_filters=(
            {"source_document": "reference.docx"}
            if metadata_filters is None
            else metadata_filters
        ),
        max_results=3,
        retrieval_strategy=strategy,
    )


def _builder() -> ProposalRetrievalRequestBuilder:
    return ProposalRetrievalRequestBuilder(
        QueryEngine(
            embedding_provider=MockEmbeddingProvider(dimensions=3),
            embedding_dimension=3,
            embedding_model_id="proposal-test-embedding",
        )
    )


def _repository() -> QdrantVectorRepository:
    return QdrantVectorRepository(FakeQdrantClient(), "proposal_chunks")


def _candidate(identifier: str, *, score: float = 0.9) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=f"chunk-{identifier}",
        document_id="proposal-1",
        text=f"Reference content {identifier}",
        score=score,
        metadata={
            "source_document": "reference.docx",
            "document": {"document_id": "proposal-1"},
            "section": {"section_path": ["Solution", "Security"]},
            "chunk": {"chunk_index": 2},
        },
        header_path=("Solution", "Security"),
        chunk_index=2,
        point_id=f"point-{identifier}",
    )


def _result(*candidates: RetrievedCandidate) -> RepositorySearchResult:
    return RepositorySearchResult(candidates=candidates, repository_metadata={"repository": "qdrant"})


def test_retriever_maps_successful_results_and_preserves_nested_metadata() -> None:
    repository = _repository()
    with patch.object(repository, "search", return_value=_result(_candidate("one"))) as search:
        references = QdrantProposalRetriever(_builder(), repository).retrieve(_query())

    assert len(references) == 1
    assert references[0].reference_id == "point-one"
    assert references[0].reference_type == "technical"
    assert references[0].chunk_id == "chunk-one"
    assert references[0].source_document == "reference.docx"
    assert references[0].metadata["document"] == {"document_id": "proposal-1"}
    assert references[0].metadata["section"] == {"section_path": ("Solution", "Security")}
    assert search.call_args.args[0].metadata_filters == {
        "source_document": "reference.docx",
        "document_role": DocumentRole.REFERENCE_KNOWLEDGE.value,
    }


def test_retriever_returns_empty_tuple_for_empty_repository_result() -> None:
    repository = _repository()
    with patch.object(repository, "search", return_value=_result()):
        assert QdrantProposalRetriever(_builder(), repository).retrieve(_query()) == ()


def test_retriever_preserves_repository_order_and_invokes_builder_once() -> None:
    builder = _builder()
    repository = _repository()
    with patch.object(builder, "build", wraps=builder.build) as build, patch.object(
        repository, "search", return_value=_result(_candidate("second"), _candidate("first"))
    ):
        references = QdrantProposalRetriever(builder, repository).retrieve(_query())

    assert [reference.reference_id for reference in references] == ["point-second", "point-first"]
    build.assert_called_once_with(_query())


def test_retriever_wraps_repository_errors_as_proposal_retrieval_errors() -> None:
    repository = _repository()
    with patch.object(repository, "search", side_effect=OSError("Qdrant unavailable")):
        with pytest.raises(RuntimeError, match="proposal reference retrieval failed") as error:
            QdrantProposalRetriever(_builder(), repository).retrieve(_query())

    assert isinstance(error.value.__cause__, OSError)


def test_retriever_rejects_repository_unsupported_strategies() -> None:
    repository = _repository()

    with pytest.raises(ValueError, match="does not support 'hybrid'"):
        QdrantProposalRetriever(_builder(), repository).retrieve(
            _query(strategy=RetrievalStrategy.HYBRID)
        )


def test_retriever_outputs_are_immutable() -> None:
    repository = _repository()
    with patch.object(repository, "search", return_value=_result(_candidate("one"))):
        reference = QdrantProposalRetriever(_builder(), repository).retrieve(_query())[0]

    with pytest.raises(Exception):
        reference.content = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        reference.metadata["document"]["document_id"] = "changed"  # type: ignore[index]


def test_proposal_retrieval_returns_only_reference_knowledge_points() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "proposal_chunks",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    def payload(identifier: str, role_marker=...) -> dict:
        value = {
            "chunk_id": f"chunk-{identifier}",
            "document_id": f"document-{identifier}",
            "original_text": f"Synthetic content {identifier}",
            "source_document": f"source-{identifier}",
            "section_path": ["Technical"],
            "chunk_index": 0,
            "section_type": "technical",
        }
        if role_marker is not ...:
            value["document_role"] = role_marker
        return value

    client.upsert(
        "proposal_chunks",
        points=[
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                payload=payload("A", DocumentRole.REFERENCE_KNOWLEDGE.value),
            ),
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                payload=payload("B", DocumentRole.PUBLISHING_TEMPLATE.value),
            ),
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0], payload=payload("C"),
            ),
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                payload=payload("D", "UNKNOWN"),
            ),
        ],
    )

    repository = QdrantVectorRepository(client, "proposal_chunks")
    references = QdrantProposalRetriever(_builder(), repository).retrieve(
        _query(metadata_filters={})
    )

    assert [reference.chunk_id for reference in references] == ["chunk-A"]


def test_empty_caller_filter_cannot_remove_mandatory_role() -> None:
    repository = _repository()
    with patch.object(repository, "search", return_value=_result()) as search:
        QdrantProposalRetriever(_builder(), repository).retrieve(
            _query(metadata_filters={})
        )

    assert search.call_args.args[0].metadata_filters == {
        "document_role": DocumentRole.REFERENCE_KNOWLEDGE.value
    }


@pytest.mark.parametrize(
    "role",
    [DocumentRole.PUBLISHING_TEMPLATE.value, "UNKNOWN"],
)
def test_conflicting_or_unknown_caller_role_fails_closed(role: str) -> None:
    repository = _repository()
    with patch.object(repository, "search") as search:
        with pytest.raises(ValueError, match="requires document_role=REFERENCE_KNOWLEDGE"):
            QdrantProposalRetriever(_builder(), repository).retrieve(
                _query(metadata_filters={"document_role": role})
            )

    search.assert_not_called()


def test_additional_caller_filters_are_anded_with_mandatory_role() -> None:
    repository = _repository()
    with patch.object(repository, "search", return_value=_result()) as search:
        QdrantProposalRetriever(_builder(), repository).retrieve(
            _query(metadata_filters={"section_type": "technical"})
        )

    assert search.call_args.args[0].metadata_filters == {
        "section_type": "technical",
        "document_role": DocumentRole.REFERENCE_KNOWLEDGE.value,
    }


def test_mandatory_role_filter_is_enforced_through_collection_alias() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "reference-v1",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )

    def payload(identifier: str, role_marker=...) -> dict:
        value = {
            "chunk_id": f"chunk-{identifier}",
            "document_id": f"document-{identifier}",
            "original_text": f"Synthetic content {identifier}",
            "source_document": f"source-{identifier}",
            "section_path": ["Technical"],
            "chunk_index": 0,
        }
        if role_marker is not ...:
            value["document_role"] = role_marker
        return value

    client.upsert(
        "reference-v1",
        points=[
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                payload=payload("reference", DocumentRole.REFERENCE_KNOWLEDGE.value),
            ),
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                payload=payload("template", DocumentRole.PUBLISHING_TEMPLATE.value),
            ),
            PointStruct(
                id=str(uuid4()), vector=[1.0, 0.0, 0.0], payload=payload("missing"),
            ),
        ],
    )
    client.update_collection_aliases(
        (
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name="reference-v1", alias_name="reference"
                )
            ),
        )
    )

    repository = QdrantVectorRepository(client, "reference")
    references = QdrantProposalRetriever(
        ProposalRetrievalRequestBuilder(
            QueryEngine(
                embedding_provider=MockEmbeddingProvider(dimensions=3),
                embedding_dimension=3,
                embedding_model_id="proposal-test-embedding",
            ),
            search_scope=SearchScope(collection="reference"),
        ),
        repository,
    ).retrieve(_query(metadata_filters={}))

    assert [reference.chunk_id for reference in references] == ["chunk-reference"]

    generic_request = ProposalRetrievalRequestBuilder(
        QueryEngine(
            embedding_provider=MockEmbeddingProvider(dimensions=3),
            embedding_dimension=3,
            embedding_model_id="proposal-test-embedding",
        ),
        search_scope=SearchScope(collection="reference"),
    ).build(
        _query(
            metadata_filters={
                "document_role": DocumentRole.PUBLISHING_TEMPLATE.value
            }
        )
    )
    assert [candidate.chunk_id for candidate in repository.search(generic_request).candidates] == [
        "chunk-template"
    ]
