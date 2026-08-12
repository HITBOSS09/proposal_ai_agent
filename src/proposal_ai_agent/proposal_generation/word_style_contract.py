"""Semantic Word style contracts consumed by a future DOCX backend."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WordStyleContract(BaseModel):
    """Immutable semantic formatting intent without presentation attributes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_style_name: str = Field(default="Normal", min_length=1)
    paragraph_type: str
    outline_level: int | None
    keep_with_next: bool
    keep_together: bool
    allow_page_break: bool
    table_behavior: str | None
    caption_position: str | None


class ProposalTitleStyle(WordStyleContract):
    """Semantic style intent for the proposal document title."""

    template_style_name: str = Field(default="Proposal_Title", min_length=1)
    paragraph_type: str = "proposal_title"
    outline_level: int | None = None
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = None
    caption_position: str | None = None


class CoverTitleStyle(WordStyleContract):
    """Semantic style intent for a cover-page title."""

    template_style_name: str = Field(default="Proposal_CoverTitle", min_length=1)
    paragraph_type: str = "cover_title"
    outline_level: int | None = None
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = None
    caption_position: str | None = None


class ModuleBannerStyle(WordStyleContract):
    """Semantic style intent for a module boundary."""

    template_style_name: str = Field(default="Proposal_ModuleHeader", min_length=1)
    paragraph_type: str = "module_banner"
    outline_level: int | None = 1
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class Heading1Style(WordStyleContract):
    """Semantic style intent for first-level headings."""

    template_style_name: str = Field(default="Proposal_Heading1", min_length=1)
    paragraph_type: str = "heading"
    outline_level: int | None = 1
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class Heading2Style(WordStyleContract):
    """Semantic style intent for second-level headings."""

    template_style_name: str = Field(default="Proposal_Heading2", min_length=1)
    paragraph_type: str = "heading"
    outline_level: int | None = 2
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class Heading3Style(WordStyleContract):
    """Semantic style intent for third-level headings."""

    template_style_name: str = Field(default="Proposal_Heading3", min_length=1)
    paragraph_type: str = "heading"
    outline_level: int | None = 3
    keep_with_next: bool = True
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class BodyStyle(WordStyleContract):
    """Semantic style intent for prose body content."""

    template_style_name: str = Field(default="Proposal_BodyText", min_length=1)
    paragraph_type: str = "body"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = False
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class BulletStyle(WordStyleContract):
    """Semantic style intent for bullet-list items."""

    template_style_name: str = Field(default="Proposal_Bullet", min_length=1)
    paragraph_type: str = "bullet"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class TableHeaderStyle(WordStyleContract):
    """Semantic style intent for table-header content."""

    template_style_name: str = Field(default="Proposal_TableHeader", min_length=1)
    paragraph_type: str = "table_header"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = "header"
    caption_position: str | None = None


class TableCellStyle(WordStyleContract):
    """Semantic style intent for table-cell content."""

    template_style_name: str = Field(default="Proposal_TableCell", min_length=1)
    paragraph_type: str = "table_cell"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = False
    allow_page_break: bool = True
    table_behavior: str | None = "cell"
    caption_position: str | None = None


class RequirementMatrixStyle(WordStyleContract):
    """Semantic style intent for requirement traceability matrices."""

    template_style_name: str = Field(default="Proposal_RequirementMatrix", min_length=1)
    paragraph_type: str = "requirement_matrix"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = "requirement_matrix"
    caption_position: str | None = None


class CalloutStyle(WordStyleContract):
    """Semantic style intent for callout content."""

    template_style_name: str = Field(default="Proposal_CalloutText", min_length=1)
    paragraph_type: str = "callout"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = None


class CaptionStyle(WordStyleContract):
    """Semantic style intent for visual captions."""

    template_style_name: str = Field(default="Proposal_Caption", min_length=1)
    paragraph_type: str = "caption"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = True
    table_behavior: str | None = None
    caption_position: str | None = "below"


class HeaderStyle(WordStyleContract):
    """Semantic style intent for document-header content."""

    template_style_name: str = Field(default="Proposal_Header", min_length=1)
    paragraph_type: str = "header"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = None
    caption_position: str | None = None


class FooterStyle(WordStyleContract):
    """Semantic style intent for document-footer content."""

    template_style_name: str = Field(default="Proposal_Footer", min_length=1)
    paragraph_type: str = "footer"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = None
    caption_position: str | None = None


class PageNumberStyle(WordStyleContract):
    """Semantic style intent for page-number content."""

    template_style_name: str = Field(default="Proposal_PageNumber", min_length=1)
    paragraph_type: str = "page_number"
    outline_level: int | None = None
    keep_with_next: bool = False
    keep_together: bool = True
    allow_page_break: bool = False
    table_behavior: str | None = None
    caption_position: str | None = None
