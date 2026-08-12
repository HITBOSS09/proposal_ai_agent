"""Deterministic mapping from validated transport DTOs to Proposal IR."""

from __future__ import annotations

from .proposal_ir import (
    BulletList,
    Callout,
    Heading,
    KnowledgeReference,
    Paragraph,
    ProposalDocument,
    ProposalMetadata,
    RequirementMatrix,
    RequirementMatrixEntry,
    Section,
    Table,
    VisualPlaceholder,
)
from .transport_contract import (
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
    TransportBlock,
    VisualPlaceholderResponse,
)


class ProposalTransportMapper:
    """Map validated proposal transport DTOs to immutable Proposal IR values."""

    def map(self, response: ProposalResponse) -> ProposalDocument:
        """Map one validated proposal response without changing its semantics."""
        if not isinstance(response, ProposalResponse):
            raise TypeError("response must be a ProposalResponse")
        return ProposalDocument(
            proposal_id=response.proposal_id,
            title=response.title,
            metadata=self._map_metadata(response.metadata),
            sections=tuple(self._map_section(section) for section in response.sections),
            references=tuple(self._map_reference(reference) for reference in response.references),
        )

    @staticmethod
    def _map_metadata(response: ProposalMetadataResponse) -> ProposalMetadata:
        return ProposalMetadata(ir_version=response.transport_version)

    def _map_section(self, response: SectionResponse) -> Section:
        return Section(
            section_id=response.section_id,
            heading=self._map_heading(response.heading),
            blocks=tuple(self._map_block(block) for block in response.blocks),
            children=tuple(self._map_section(child) for child in response.children),
        )

    @staticmethod
    def _map_heading(response: HeadingResponse) -> Heading:
        return Heading(text=response.text, level=response.level)

    def _map_block(
        self,
        response: TransportBlock,
    ) -> Paragraph | BulletList | Table | VisualPlaceholder | Callout | RequirementMatrix:
        if isinstance(response, ParagraphResponse):
            return Paragraph(text=response.text, reference_ids=response.reference_ids)
        if isinstance(response, BulletListResponse):
            return BulletList(items=response.items)
        if isinstance(response, TableResponse):
            return Table(headers=response.headers, rows=response.rows)
        if isinstance(response, VisualPlaceholderResponse):
            return VisualPlaceholder(
                visual_id=response.visual_id,
                description=response.description,
                caption=response.caption,
            )
        if isinstance(response, CalloutResponse):
            return Callout(label=response.label, text=response.text, reference_ids=response.reference_ids)
        if isinstance(response, RequirementMatrixResponse):
            return RequirementMatrix(entries=tuple(self._map_requirement_entry(entry) for entry in response.entries))
        raise TypeError("response must be a supported transport block")

    @staticmethod
    def _map_requirement_entry(response: RequirementMatrixEntryResponse) -> RequirementMatrixEntry:
        return RequirementMatrixEntry(
            requirement_id=response.requirement_id,
            requirement=response.requirement,
            response=response.response,
            evidence_reference_ids=response.evidence_reference_ids,
        )

    @staticmethod
    def _map_reference(response: KnowledgeReferenceResponse) -> KnowledgeReference:
        return KnowledgeReference(
            reference_id=response.reference_id,
            title=response.title,
            source=response.source,
            locator=response.locator,
        )


def map_proposal_transport(response: ProposalResponse) -> ProposalDocument:
    """Map one validated transport response into Proposal IR."""
    return ProposalTransportMapper().map(response)
