"""Contract coverage for proposal-domain retrieval queries."""

import pytest

from proposal_ai_agent.proposal_generation import (
    ReferenceType,
    RetrievalStrategy,
    SectionRetrievalQuery,
)


def test_section_retrieval_query_is_immutable_and_provider_neutral() -> None:
    query = SectionRetrievalQuery(
        section_id="solution-overview",
        reference_type=ReferenceType.TECHNICAL,
        query_text="Northstar project requirements",
        metadata_filters={"industry": "energy", "approved": True},
        max_results=5,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        rerank_enabled=True,
    )

    assert query.reference_type is ReferenceType.TECHNICAL
    assert query.retrieval_strategy is RetrievalStrategy.HYBRID
    assert query.metadata_filters == {"industry": "energy", "approved": True}
    with pytest.raises(TypeError):
        query.metadata_filters["region"] = "global"  # type: ignore[index]
    with pytest.raises(Exception):
        query.max_results = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("section_id", ""),
        ("query_text", ""),
        ("max_results", 0),
    ],
)
def test_section_retrieval_query_rejects_invalid_required_values(field, value) -> None:
    values = {
        "section_id": "section",
        "reference_type": ReferenceType.AUTHORING,
        "query_text": "project query",
        "max_results": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        SectionRetrievalQuery(**values)
