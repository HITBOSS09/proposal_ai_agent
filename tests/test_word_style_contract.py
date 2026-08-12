"""Tests for semantic, renderer-independent Word style contracts."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.word_style_contract import (
    BodyStyle,
    BulletStyle,
    CalloutStyle,
    CaptionStyle,
    CoverTitleStyle,
    FooterStyle,
    HeaderStyle,
    Heading1Style,
    Heading2Style,
    Heading3Style,
    ModuleBannerStyle,
    PageNumberStyle,
    ProposalTitleStyle,
    RequirementMatrixStyle,
    TableCellStyle,
    TableHeaderStyle,
    WordStyleContract,
)


def test_all_required_style_contracts_are_immutable_semantic_values() -> None:
    styles = (
        ProposalTitleStyle(), CoverTitleStyle(), ModuleBannerStyle(), Heading1Style(), Heading2Style(), Heading3Style(),
        BodyStyle(), BulletStyle(), TableHeaderStyle(), TableCellStyle(), RequirementMatrixStyle(), CalloutStyle(),
        CaptionStyle(), HeaderStyle(), FooterStyle(), PageNumberStyle(),
    )

    assert all(isinstance(style, WordStyleContract) for style in styles)
    assert [style.paragraph_type for style in styles] == [
        "proposal_title", "cover_title", "module_banner", "heading", "heading", "heading", "body", "bullet",
        "table_header", "table_cell", "requirement_matrix", "callout", "caption", "header", "footer", "page_number",
    ]
    with pytest.raises(ValidationError):
        styles[0].paragraph_type = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        styles[0].template_style_name = "Changed"  # type: ignore[misc]


def test_every_semantic_style_has_a_non_empty_template_style_name() -> None:
    styles = (
        ProposalTitleStyle(), CoverTitleStyle(), ModuleBannerStyle(), Heading1Style(), Heading2Style(), Heading3Style(),
        BodyStyle(), BulletStyle(), TableHeaderStyle(), TableCellStyle(), RequirementMatrixStyle(), CalloutStyle(),
        CaptionStyle(), HeaderStyle(), FooterStyle(), PageNumberStyle(),
    )

    assert all(style.template_style_name for style in styles)
    assert [style.template_style_name for style in styles] == [
        "Proposal_Title", "Proposal_CoverTitle", "Proposal_ModuleHeader", "Proposal_Heading1", "Proposal_Heading2",
        "Proposal_Heading3", "Proposal_BodyText", "Proposal_Bullet", "Proposal_TableHeader", "Proposal_TableCell",
        "Proposal_RequirementMatrix", "Proposal_CalloutText", "Proposal_Caption", "Proposal_Header", "Proposal_Footer",
        "Proposal_PageNumber",
    ]


def test_heading_style_contracts_use_distinct_template_style_names() -> None:
    assert {
        Heading1Style().template_style_name,
        Heading2Style().template_style_name,
        Heading3Style().template_style_name,
    } == {"Proposal_Heading1", "Proposal_Heading2", "Proposal_Heading3"}


def test_heading_and_table_contracts_express_semantic_behavior() -> None:
    assert [Heading1Style().outline_level, Heading2Style().outline_level, Heading3Style().outline_level] == [1, 2, 3]
    assert TableHeaderStyle().table_behavior == "header"
    assert TableCellStyle().table_behavior == "cell"
    assert RequirementMatrixStyle().table_behavior == "requirement_matrix"
    assert CaptionStyle().caption_position == "below"


def test_style_contracts_serialize_and_reject_presentation_fields() -> None:
    style = BodyStyle()

    assert type(style).model_validate(style.model_dump(mode="json")) == style
    assert style.model_dump(mode="json")["template_style_name"] == "Proposal_BodyText"
    with pytest.raises(ValidationError, match="at least 1 character"):
        BodyStyle(template_style_name="")
    with pytest.raises(ValidationError, match="Extra inputs"):
        BodyStyle(font_name="Aptos")  # type: ignore[call-arg]
