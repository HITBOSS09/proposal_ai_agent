"""Tests for provider-neutral Proposal Compiler transport DTOs."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.transport_contract import (
    BulletListResponse,
    CalloutResponse,
    HeadingResponse,
    KnowledgeReferenceResponse,
    ParagraphResponse,
    ProposalMetadataResponse,
    ProposalResponse,
    RequirementMatrixEntryResponse,
    RequirementMatrixResponse,
    SectionResponse,
    TableResponse,
    VisualPlaceholderResponse,
)


def _response() -> ProposalResponse:
    return ProposalResponse(
        proposal_id="proposal-001",
        title="Autonomous Perimeter Monitoring Proposal",
        metadata=ProposalMetadataResponse(source_model="qwen2.5:3b", request_id="request-001"),
        sections=(
            SectionResponse(
                section_id="solution",
                heading=HeadingResponse(text="Solution", level=1),
                blocks=(
                    ParagraphResponse(text="The platform provides persistent monitoring.", reference_ids=("REF-1",)),
                    BulletListResponse(items=("EO/IR observation", "Encrypted communications")),
                    TableResponse(headers=("Requirement", "Response"), rows=(("R-1", "Covered"),)),
                    VisualPlaceholderResponse(visual_id="VIS-1", description="System architecture"),
                    CalloutResponse(label="Operational note", text="Field deployment requires site survey."),
                    RequirementMatrixResponse(entries=(
                        RequirementMatrixEntryResponse(
                            requirement_id="R-1",
                            requirement="Provide EO/IR observation",
                            response="EO/IR payload included",
                            evidence_reference_ids=("REF-1",),
                        ),
                    )),
                ),
                children=(
                    SectionResponse(
                        section_id="solution-subsection",
                        heading=HeadingResponse(text="Payload", level=2),
                    ),
                ),
            ),
        ),
        references=(KnowledgeReferenceResponse(
            reference_id="REF-1", title="Payload specification", source="Technical library", locator="Section 4"
        ),),
    )


def test_transport_contract_serializes_and_deserializes_deterministically() -> None:
    response = _response()

    serialized = response.model_dump(mode="json")

    assert ProposalResponse.model_validate(serialized) == response
    assert [block.type for block in response.sections[0].blocks] == [
        "paragraph", "bullet_list", "table", "visual_placeholder", "callout", "requirement_matrix",
    ]
    assert serialized["sections"][0]["children"][0]["section_id"] == "solution-subsection"


def test_transport_contract_supports_optional_fields_and_defaults() -> None:
    response = ProposalResponse(proposal_id="proposal-001", title="Proposal")
    visual = VisualPlaceholderResponse(visual_id="VIS-1", description="Architecture")
    reference = KnowledgeReferenceResponse(reference_id="REF-1", title="Source", source="Library")

    assert response.metadata.transport_version == "1.0"
    assert response.sections == ()
    assert visual.caption is None
    assert reference.locator is None


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "Missing proposal ID"},
        {"proposal_id": "proposal-001"},
    ],
)
def test_transport_contract_requires_proposal_identity_and_title(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ProposalResponse.model_validate(payload)


def test_transport_contract_rejects_unknown_dto_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ParagraphResponse(text="Semantic text", font="Aptos")  # type: ignore[call-arg]
