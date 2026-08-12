"""Unit tests for the format-independent Proposal IR."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.proposal_ir import (
    BulletList,
    Callout,
    BlockNode,
    FigurePlaceholder,
    Heading,
    Paragraph,
    ProposalDocument,
    KnowledgeReference,
    ProposalMetadata,
    Reference,
    RequirementMatrix,
    RequirementMatrixEntry,
    Section,
    Table,
    VisualPlaceholder,
)


def _document() -> ProposalDocument:
    return ProposalDocument(
        proposal_id="proposal-001",
        title="Autonomous Perimeter Monitoring Proposal",
        sections=(
            Section(
                section_id="solution",
                heading=Heading(text="Solution", level=1),
                blocks=(
                    Paragraph(text="The platform provides persistent monitoring.", reference_ids=("REF-1",)),
                    BulletList(items=("EO/IR observation", "Encrypted communications")),
                    Table(headers=("Requirement", "Response"), rows=(("R-1", "Covered"),)),
                    VisualPlaceholder(visual_id="VIS-1", description="System architecture", caption="Platform overview"),
                    Callout(label="Operational note", text="Field deployment requires site survey."),
                    RequirementMatrix(entries=(
                        RequirementMatrixEntry(
                            requirement_id="R-1",
                            requirement="Provide EO/IR observation",
                            response="EO/IR payload included",
                            evidence_reference_ids=("REF-1",),
                        ),
                    )),
                ),
            ),
        ),
        references=(KnowledgeReference(reference_id="REF-1", title="Payload specification", source="Technical library"),),
    )


def test_proposal_ir_is_immutable_and_round_trips() -> None:
    document = _document()

    with pytest.raises(ValidationError):
        document.title = "Changed"  # type: ignore[misc]

    assert ProposalDocument.model_validate(document.model_dump(mode="json")) == document
    assert [block.kind for block in document.sections[0].blocks] == [
        "paragraph", "bullet_list", "table", "visual_placeholder", "callout", "requirement_matrix",
    ]


def test_proposal_ir_rejects_invalid_semantic_shapes_and_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="header count"):
        Table(headers=("A", "B"), rows=(("only one cell",),))

    with pytest.raises(ValidationError, match="unique section_id"):
        ProposalDocument(
            proposal_id="proposal-001",
            title="Proposal",
            sections=(
                Section(section_id="summary", heading=Heading(text="Summary", level=1)),
                Section(section_id="summary", heading=Heading(text="Repeated", level=1)),
            ),
        )

    with pytest.raises(ValidationError, match="unique requirement_id"):
        RequirementMatrix(entries=(
            RequirementMatrixEntry(requirement_id="R-1", requirement="A", response="A"),
            RequirementMatrixEntry(requirement_id="R-1", requirement="B", response="B"),
        ))


def test_proposal_ir_contains_no_presentation_configuration() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Paragraph(text="Semantic text", font="Aptos")  # type: ignore[call-arg]


def test_proposal_ir_v1_metadata_node_identity_and_compatibility_aliases() -> None:
    document = _document()
    visual = FigurePlaceholder(figure_id="FIG-1", description="Legacy figure input")
    reference = Reference(reference_id="REF-2", title="Legacy reference", source="Knowledge base")

    assert document.metadata == ProposalMetadata(ir_version="1.0")
    assert document.document_id
    assert document.sections[0].node_id
    assert document.sections[0].heading.node_id
    assert all(isinstance(block, BlockNode) for block in document.sections[0].blocks)
    assert visual.visual_id == "FIG-1"
    assert visual.kind == "visual_placeholder"
    assert isinstance(reference, KnowledgeReference)
    assert document.sections[0].blocks[-1].entries[0].node_id
    assert document.references[0].node_id


def test_proposal_ir_preserves_nested_section_hierarchy_and_rejects_invalid_depth() -> None:
    nested = Section(
        section_id="parent",
        heading=Heading(text="Parent", level=1),
        children=(Section(section_id="child", heading=Heading(text="Child", level=2)),),
    )
    document = ProposalDocument(proposal_id="proposal-001", title="Proposal", sections=(nested,))

    assert document.sections[0].children[0].section_id == "child"
    with pytest.raises(ValidationError, match="deeper"):
        Section(
            section_id="parent",
            heading=Heading(text="Parent", level=1),
            children=(Section(section_id="child", heading=Heading(text="Child", level=1)),),
        )
