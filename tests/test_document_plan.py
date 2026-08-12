"""Tests for the immutable, renderer-independent document planning contract."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.document_plan import (
    DocumentPolicy,
    ProposalPlan,
    SectionPlan,
    SectionRole,
)


def _plan() -> ProposalPlan:
    return ProposalPlan(
        proposal_id="proposal-001",
        sections=(
            SectionPlan(section_id="cover", role=SectionRole.COVER, include_in_toc=False, numbering_enabled=False),
            SectionPlan(section_id="contents", role=SectionRole.TABLE_OF_CONTENTS, numbering_enabled=False),
            SectionPlan(section_id="solution", role=SectionRole.BODY, page_break_before=True),
            SectionPlan(section_id="appendix-a", role=SectionRole.APPENDIX),
            SectionPlan(section_id="references", role=SectionRole.REFERENCES),
        ),
    )


def test_document_policy_defaults_are_structural_and_serializable() -> None:
    policy = DocumentPolicy()

    assert policy.model_dump(mode="json") == {
        "toc_enabled": True,
        "header_enabled": True,
        "footer_enabled": True,
        "numbering_enabled": True,
        "revision_history_enabled": False,
        "page_numbering_format": "arabic",
        "landscape_tables_allowed": True,
        "appendix_numbering_enabled": True,
    }


def test_proposal_plan_preserves_order_and_round_trips_json() -> None:
    plan = _plan()
    serialized = plan.model_dump(mode="json")

    assert [section["section_id"] for section in serialized["sections"]] == [
        "cover", "contents", "solution", "appendix-a", "references",
    ]
    assert ProposalPlan.model_validate(serialized) == plan


def test_document_plan_models_are_immutable() -> None:
    plan = _plan()

    with pytest.raises(ValidationError):
        plan.proposal_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        plan.sections[0].role = SectionRole.BODY  # type: ignore[misc]


def test_section_role_serializes_to_provider_neutral_values() -> None:
    section = SectionPlan(section_id="annex-a", role=SectionRole.ANNEX)

    assert section.model_dump(mode="json")["role"] == "annex"
    assert tuple(SectionRole) == (
        SectionRole.COVER,
        SectionRole.TABLE_OF_CONTENTS,
        SectionRole.MODULE,
        SectionRole.BODY,
        SectionRole.APPENDIX,
        SectionRole.ANNEX,
        SectionRole.REFERENCES,
    )
