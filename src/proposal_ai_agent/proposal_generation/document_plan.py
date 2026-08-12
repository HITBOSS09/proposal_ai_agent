"""Format-independent planning contract for proposal document intent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _DocumentPlanModel(BaseModel):
    """Immutable base configuration for document-plan contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SectionRole(str, Enum):
    """Semantic role of a planned proposal section."""

    COVER = "cover"
    TABLE_OF_CONTENTS = "table_of_contents"
    MODULE = "module"
    BODY = "body"
    APPENDIX = "appendix"
    ANNEX = "annex"
    REFERENCES = "references"


class DocumentPolicy(_DocumentPlanModel):
    """Proposal-wide structural behavior, independent of presentation details."""

    toc_enabled: bool = True
    header_enabled: bool = True
    footer_enabled: bool = True
    numbering_enabled: bool = True
    revision_history_enabled: bool = False
    page_numbering_format: str = "arabic"
    landscape_tables_allowed: bool = True
    appendix_numbering_enabled: bool = True


class SectionPlan(_DocumentPlanModel):
    """Structural intent for one ordered semantic proposal section."""

    section_id: str
    role: SectionRole
    page_break_before: bool = False
    include_in_toc: bool = True
    numbering_enabled: bool = True


class ProposalPlan(_DocumentPlanModel):
    """Ordered document-intent contract consumed by a future composer."""

    proposal_id: str
    document_policy: DocumentPolicy = Field(default_factory=DocumentPolicy)
    sections: tuple[SectionPlan, ...] = ()
