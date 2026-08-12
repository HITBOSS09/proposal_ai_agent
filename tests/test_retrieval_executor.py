"""Unit tests for provider-neutral proposal retrieval execution."""

import pytest

from proposal_ai_agent.proposal_generation import (
    ProposalReferenceRetriever,
    ReferenceType,
    RetrievedContext,
    RetrievedReference,
    RetrievalExecutor,
    SectionRetrievalQuery,
)


def _query(section_id: str, reference_type: ReferenceType) -> SectionRetrievalQuery:
    return SectionRetrievalQuery(
        section_id=section_id,
        reference_type=reference_type,
        query_text=f"{section_id} {reference_type.value}",
        max_results=3,
    )


def _reference(reference_type: ReferenceType, identifier: str = "reference-1") -> RetrievedReference:
    return RetrievedReference(
        reference_id=identifier,
        reference_type=reference_type.value,
        chunk_id=f"chunk-{identifier}",
        source_document="reference.docx",
        score=0.9,
        content="Reference content",
    )


class MockRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query: SectionRetrievalQuery) -> tuple[RetrievedReference, ...]:
        self.calls.append(query)
        return tuple(self.responses.get((query.section_id, query.reference_type), ()))


def test_executor_preserves_query_and_section_order_and_groups_references() -> None:
    queries = (
        _query("solution", ReferenceType.TECHNICAL),
        _query("summary", ReferenceType.AUTHORING),
        _query("solution", ReferenceType.BLUEPRINT),
    )
    retriever = MockRetriever(
        {
            ("solution", ReferenceType.TECHNICAL): (_reference(ReferenceType.TECHNICAL),),
            ("summary", ReferenceType.AUTHORING): (_reference(ReferenceType.AUTHORING),),
            ("solution", ReferenceType.BLUEPRINT): (_reference(ReferenceType.BLUEPRINT),),
        }
    )

    contexts = RetrievalExecutor(retriever).execute(queries)

    assert retriever.calls == list(queries)
    assert [context.section_id for context in contexts] == ["solution", "summary"]
    assert contexts[0].technical_references[0].reference_type == "technical"
    assert contexts[0].blueprint_references[0].reference_type == "blueprint"
    assert contexts[1].style_references[0].reference_type == "authoring"


def test_executor_returns_empty_tuple_without_queries() -> None:
    retriever = MockRetriever({})

    assert RetrievalExecutor(retriever).execute(()) == ()
    assert retriever.calls == []


def test_executor_propagates_retriever_errors() -> None:
    class FailingRetriever(MockRetriever):
        def retrieve(self, query: SectionRetrievalQuery) -> tuple[RetrievedReference, ...]:
            raise RuntimeError("retrieval failed")

    with pytest.raises(RuntimeError, match="retrieval failed"):
        RetrievalExecutor(FailingRetriever({})).execute((_query("summary", ReferenceType.AUTHORING),))


def test_executor_deduplicates_references_by_type_and_identifier() -> None:
    query = _query("summary", ReferenceType.AUTHORING)
    repeated_query = _query("summary", ReferenceType.AUTHORING)
    duplicate = _reference(ReferenceType.AUTHORING, "reference-1")
    context = RetrievalExecutor(
        MockRetriever({("summary", ReferenceType.AUTHORING): (duplicate, duplicate)})
    ).execute((query, repeated_query))[0]

    assert context.style_references == (duplicate,)


def test_retrieved_context_is_immutable() -> None:
    query = _query("summary", ReferenceType.AUTHORING)
    context = RetrievalExecutor(MockRetriever({})).execute((query,))[0]

    assert isinstance(context, RetrievedContext)
    with pytest.raises(Exception):
        context.section_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.metadata["query_count"] = 2  # type: ignore[index]
