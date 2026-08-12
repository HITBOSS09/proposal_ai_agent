"""Tests for deterministic Proposal Plan plus Proposal IR composition."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.composition import (
    ComponentContent,
    CompositionEngine,
    CompositionPlanMismatch,
)
from proposal_ai_agent.proposal_generation.document_plan import ProposalPlan, SectionPlan, SectionRole
from proposal_ai_agent.proposal_generation.proposal_ir import (
    BulletList,
    Callout,
    Heading,
    KnowledgeReference,
    Paragraph,
    ProposalDocument,
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
            Section(section_id="cover", heading=Heading(text="Proposal", level=1)),
            Section(section_id="contents", heading=Heading(text="Contents", level=1)),
            Section(
                section_id="module-solution",
                heading=Heading(text="Solution Module", level=1),
                children=(
                    Section(
                        section_id="solution",
                        heading=Heading(text="Solution Overview", level=2),
                        blocks=(
                            Paragraph(text="The platform provides persistent monitoring.", reference_ids=("REF-1",)),
                            BulletList(items=("EO/IR observation", "Encrypted communications")),
                            Table(headers=("Requirement", "Response"), rows=(("R-1", "Covered"),)),
                            VisualPlaceholder(visual_id="VIS-1", description="System architecture"),
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
            ),
            Section(section_id="appendix-a", heading=Heading(text="Appendix A", level=1)),
            Section(section_id="references", heading=Heading(text="References", level=1)),
        ),
        references=(KnowledgeReference(reference_id="REF-1", title="Payload specification", source="Technical library"),),
    )


def _plan() -> ProposalPlan:
    return ProposalPlan(
        proposal_id="proposal-001",
        sections=(
            SectionPlan(section_id="cover", role=SectionRole.COVER, include_in_toc=False, numbering_enabled=False),
            SectionPlan(section_id="contents", role=SectionRole.TABLE_OF_CONTENTS, numbering_enabled=False),
            SectionPlan(section_id="module-solution", role=SectionRole.MODULE),
            SectionPlan(section_id="solution", role=SectionRole.BODY),
            SectionPlan(section_id="appendix-a", role=SectionRole.APPENDIX),
            SectionPlan(section_id="references", role=SectionRole.REFERENCES),
        ),
    )


def test_composition_engine_preserves_order_hierarchy_components_and_semantic_content() -> None:
    composition = CompositionEngine().compose(_plan(), _document())
    module = composition.components[2]
    body = module.children[0]

    assert composition.proposal_id == "proposal-001"
    assert [component.component_name for component in composition.components] == [
        "cover_page", "table_of_contents", "module_banner", "appendix", "references",
    ]
    assert module.section_id == "module-solution"
    assert body.component_name == "heading"
    assert body.section_id == "solution"
    assert [content.content.kind for content in body.slots[1].contents] == [
        "paragraph", "bullet_list", "table", "visual_placeholder", "callout", "requirement_matrix",
    ]
    assert body.slots[1].contents[2].content.rows == (("R-1", "Covered"),)
    assert body.slots[1].contents[3].content.visual_id == "VIS-1"
    assert body.slots[1].contents[5].content.entries[0].requirement_id == "R-1"
    assert composition.references[0].reference_id == "REF-1"
    assert composition.hierarchy.root_component_ids == tuple(component.component_id for component in composition.components)
    assert body.metadata == (
        ("section_role", "body"),
        ("page_break_before", False),
        ("include_in_toc", True),
        ("numbering_enabled", True),
    )


def test_composition_is_deterministic_for_the_same_plan_and_document() -> None:
    plan = _plan()
    document = _document()

    assert CompositionEngine().compose(plan, document) == CompositionEngine().compose(plan, document)


def test_composition_model_is_immutable_and_round_trips() -> None:
    composition = CompositionEngine().compose(_plan(), _document())

    with pytest.raises(ValidationError):
        composition.title = "Changed"  # type: ignore[misc]
    assert type(composition).model_validate(composition.model_dump(mode="json")) == composition
    assert isinstance(composition.components[2].children[0].slots[1].contents[0], ComponentContent)


def test_composition_rejects_plan_document_section_mismatches() -> None:
    plan = ProposalPlan(proposal_id="proposal-001", sections=(SectionPlan(section_id="cover", role=SectionRole.COVER),))

    with pytest.raises(CompositionPlanMismatch, match="section order"):
        CompositionEngine().compose(plan, _document())
