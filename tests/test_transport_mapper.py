"""Tests for deterministic transport DTO to Proposal IR mapping."""

from uuid import UUID

from proposal_ai_agent.proposal_generation.proposal_ir import (
    BulletList,
    Callout,
    Paragraph,
    RequirementMatrix,
    Table,
    VisualPlaceholder,
)
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
from proposal_ai_agent.proposal_generation.transport_mapper import ProposalTransportMapper, map_proposal_transport


def _response() -> ProposalResponse:
    return ProposalResponse(
        proposal_id="proposal-001",
        title="Autonomous Perimeter Monitoring Proposal",
        metadata=ProposalMetadataResponse(transport_version="1.0", source_model="qwen2.5:3b"),
        sections=(
            SectionResponse(
                section_id="solution",
                heading=HeadingResponse(text="Solution", level=1),
                blocks=(
                    ParagraphResponse(text="The platform provides persistent monitoring.", reference_ids=("REF-1",)),
                    BulletListResponse(items=("EO/IR observation", "Encrypted communications")),
                    TableResponse(headers=("Requirement", "Response"), rows=(("R-1", "Covered"),)),
                    VisualPlaceholderResponse(visual_id="VIS-1", description="System architecture", caption="Overview"),
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
                    SectionResponse(section_id="payload", heading=HeadingResponse(text="Payload", level=2)),
                ),
            ),
        ),
        references=(KnowledgeReferenceResponse(
            reference_id="REF-1", title="Payload specification", source="Technical library", locator="Section 4"
        ),),
    )


def test_mapper_maps_every_transport_node_and_preserves_ordering() -> None:
    document = ProposalTransportMapper().map(_response())
    section = document.sections[0]

    assert document.proposal_id == "proposal-001"
    assert document.title == "Autonomous Perimeter Monitoring Proposal"
    assert document.metadata.ir_version == "1.0"
    assert section.section_id == "solution"
    assert section.children[0].section_id == "payload"
    assert [type(block) for block in section.blocks] == [
        Paragraph, BulletList, Table, VisualPlaceholder, Callout, RequirementMatrix,
    ]
    assert section.blocks[0].reference_ids == ("REF-1",)
    assert section.blocks[2].rows == (("R-1", "Covered"),)
    assert section.blocks[3].visual_id == "VIS-1"
    assert section.blocks[5].entries[0].evidence_reference_ids == ("REF-1",)
    assert document.references[0].reference_id == "REF-1"
    assert document.references[0].locator == "Section 4"


def test_mapper_generates_ir_uuid_identity_without_overwriting_business_identifiers() -> None:
    document = map_proposal_transport(_response())
    section = document.sections[0]

    assert isinstance(document.document_id, UUID)
    assert isinstance(section.node_id, UUID)
    assert isinstance(section.heading.node_id, UUID)
    assert all(isinstance(block.node_id, UUID) for block in section.blocks)
    assert isinstance(section.blocks[-1].entries[0].node_id, UUID)
    assert isinstance(document.references[0].node_id, UUID)
    assert document.proposal_id == "proposal-001"
    assert section.section_id == "solution"
    assert document.references[0].reference_id == "REF-1"
    assert section.blocks[-1].entries[0].requirement_id == "R-1"


def test_mapper_output_round_trips_as_ir_value() -> None:
    document = map_proposal_transport(_response())

    assert type(document).model_validate(document.model_dump(mode="json")) == document
