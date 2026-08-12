"""Provider-neutral JSON DTOs for LLM proposal responses.

These transport objects are intentionally independent of Proposal IR, mapping,
validation, and rendering concerns.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class _TransportDTO(BaseModel):
    """Immutable base configuration shared by transport response DTOs."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProposalMetadataResponse(_TransportDTO):
    """Version and origin information supplied with one transport response."""

    transport_version: str = "1.0"
    source_model: str | None = None
    request_id: str | None = None


class HeadingResponse(_TransportDTO):
    """A transport representation of a semantic heading."""

    text: str
    level: int


class ParagraphResponse(_TransportDTO):
    """A prose block returned by an LLM."""

    type: Literal["paragraph"] = "paragraph"
    text: str
    reference_ids: tuple[str, ...] = ()


class BulletListResponse(_TransportDTO):
    """A list block returned by an LLM."""

    type: Literal["bullet_list"] = "bullet_list"
    items: tuple[str, ...]


class TableResponse(_TransportDTO):
    """A tabular data block returned by an LLM."""

    type: Literal["table"] = "table"
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


class VisualPlaceholderResponse(_TransportDTO):
    """A requested visual asset described by the LLM."""

    type: Literal["visual_placeholder"] = "visual_placeholder"
    visual_id: str
    description: str
    caption: str | None = None


class CalloutResponse(_TransportDTO):
    """A semantically distinguished statement returned by an LLM."""

    type: Literal["callout"] = "callout"
    label: str
    text: str
    reference_ids: tuple[str, ...] = ()


class RequirementMatrixEntryResponse(_TransportDTO):
    """One requirement-to-response statement from the LLM."""

    requirement_id: str
    requirement: str
    response: str
    evidence_reference_ids: tuple[str, ...] = ()


class RequirementMatrixResponse(_TransportDTO):
    """A requirement traceability matrix returned by an LLM."""

    type: Literal["requirement_matrix"] = "requirement_matrix"
    entries: tuple[RequirementMatrixEntryResponse, ...]


class KnowledgeReferenceResponse(_TransportDTO):
    """A knowledge source cited by a transport response."""

    reference_id: str
    title: str
    source: str
    locator: str | None = None


TransportBlock: TypeAlias = Annotated[
    ParagraphResponse
    | BulletListResponse
    | TableResponse
    | VisualPlaceholderResponse
    | CalloutResponse
    | RequirementMatrixResponse,
    Field(discriminator="type"),
]


class SectionResponse(_TransportDTO):
    """An ordered, recursively nestable section in a proposal response."""

    section_id: str
    heading: HeadingResponse
    blocks: tuple[TransportBlock, ...] = ()
    children: tuple[SectionResponse, ...] = ()


class ProposalResponse(_TransportDTO):
    """Complete provider-neutral JSON response for one proposal."""

    proposal_id: str
    title: str
    metadata: ProposalMetadataResponse = Field(default_factory=ProposalMetadataResponse)
    sections: tuple[SectionResponse, ...] = ()
    references: tuple[KnowledgeReferenceResponse, ...] = ()


SectionResponse.model_rebuild()
