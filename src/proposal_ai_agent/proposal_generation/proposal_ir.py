"""Format-independent semantic intermediate representation for proposals.

This module deliberately has no dependency on proposal assembly, rendering, or
provider code.  Its ``ProposalDocument`` is namespaced here so the certified
assembled-document contract in :mod:`proposal_ai_agent.proposal_generation.contracts`
remains unchanged.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class _ProposalIRModel(BaseModel):
    """Immutable base for semantic Proposal IR nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProposalMetadata(_ProposalIRModel):
    """Version metadata for the stable Proposal IR contract."""

    ir_version: Literal["1.0"] = "1.0"


class BlockNode(_ProposalIRModel):
    """Shared identity contract for semantic section blocks."""

    node_id: UUID = Field(default_factory=uuid4)


class Heading(_ProposalIRModel):
    """A document-outline heading, independent of any presentation style."""

    kind: Literal["heading"] = "heading"
    node_id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1)
    level: int = Field(ge=1)


class Paragraph(BlockNode):
    """One semantic prose paragraph with optional source references."""

    kind: Literal["paragraph"] = "paragraph"
    text: str = Field(min_length=1)
    reference_ids: tuple[str, ...] = ()

    @field_validator("reference_ids")
    @classmethod
    def validate_reference_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reference_id.strip() for reference_id in value):
            raise ValueError("reference_ids must contain non-empty values")
        return tuple(value)


class BulletList(BlockNode):
    """An ordered semantic collection of peer statements."""

    kind: Literal["bullet_list"] = "bullet_list"
    items: tuple[str, ...] = Field(min_length=1)

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("items must contain non-empty values")
        return tuple(value)


class Table(BlockNode):
    """A semantic rectangular data table without layout attributes."""

    kind: Literal["table"] = "table"
    headers: tuple[str, ...] = Field(min_length=1)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> "Table":
        if any(not header.strip() for header in self.headers):
            raise ValueError("headers must contain non-empty values")
        if any(len(row) != len(self.headers) for row in self.rows):
            raise ValueError("every row must match the header count")
        return self


class VisualPlaceholder(BlockNode):
    """A required visual described semantically until an asset is supplied."""

    kind: Literal["visual_placeholder"] = "visual_placeholder"
    visual_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("visual_id", "figure_id"),
        serialization_alias="visual_id",
    )
    description: str = Field(min_length=1)
    caption: str | None = None


FigurePlaceholder = VisualPlaceholder


class Callout(BlockNode):
    """A semantically distinct statement requiring special reader attention."""

    kind: Literal["callout"] = "callout"
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    reference_ids: tuple[str, ...] = ()

    @field_validator("reference_ids")
    @classmethod
    def validate_reference_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reference_id.strip() for reference_id in value):
            raise ValueError("reference_ids must contain non-empty values")
        return tuple(value)


class RequirementMatrixEntry(_ProposalIRModel):
    """One requirement-to-response relationship within a requirement matrix."""

    node_id: UUID = Field(default_factory=uuid4)
    requirement_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    response: str = Field(min_length=1)
    evidence_reference_ids: tuple[str, ...] = ()

    @field_validator("evidence_reference_ids")
    @classmethod
    def validate_reference_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not reference_id.strip() for reference_id in value):
            raise ValueError("evidence_reference_ids must contain non-empty values")
        return tuple(value)


class RequirementMatrix(BlockNode):
    """A semantic traceability matrix between requirements and responses."""

    kind: Literal["requirement_matrix"] = "requirement_matrix"
    entries: tuple[RequirementMatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_requirement_ids(self) -> "RequirementMatrix":
        identifiers = tuple(entry.requirement_id for entry in self.entries)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("requirement matrix entries must have unique requirement_id values")
        return self


class KnowledgeReference(_ProposalIRModel):
    """A knowledge source cited by semantic proposal content."""

    node_id: UUID = Field(default_factory=uuid4)
    reference_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    locator: str | None = None


Reference = KnowledgeReference


SemanticBlock: TypeAlias = Annotated[
    Paragraph | BulletList | Table | VisualPlaceholder | Callout | RequirementMatrix,
    Field(discriminator="kind"),
]


class Section(_ProposalIRModel):
    """An ordered proposal section containing semantic content blocks."""

    node_id: UUID = Field(default_factory=uuid4)
    section_id: str = Field(min_length=1)
    heading: Heading
    blocks: tuple[SemanticBlock, ...] = ()
    children: tuple[Section, ...] = ()

    @model_validator(mode="after")
    def validate_children(self) -> "Section":
        child_ids = tuple(child.section_id for child in self.children)
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child sections must have unique section_id values")
        if any(child.heading.level <= self.heading.level for child in self.children):
            raise ValueError("child section headings must be deeper than the parent heading")
        return self


class ProposalDocument(_ProposalIRModel):
    """A complete semantic proposal, independent of every output format."""

    document_id: UUID = Field(default_factory=uuid4)
    proposal_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sections: tuple[Section, ...] = ()
    references: tuple[KnowledgeReference, ...] = ()
    metadata: ProposalMetadata = Field(default_factory=ProposalMetadata)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "ProposalDocument":
        section_ids = tuple(
            section.section_id
            for root_section in self.sections
            for section in self._walk_sections(root_section)
        )
        reference_ids = tuple(reference.reference_id for reference in self.references)
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("sections must have unique section_id values")
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("references must have unique reference_id values")
        return self

    @staticmethod
    def _walk_sections(section: Section):
        yield section
        for child in section.children:
            yield from ProposalDocument._walk_sections(child)


Section.model_rebuild()
