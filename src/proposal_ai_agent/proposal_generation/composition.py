"""Deterministic semantic composition model and assembly engine."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from .component_registry import ComponentRegistry
from .document_plan import ProposalPlan, SectionPlan
from .proposal_ir import (
    BulletList,
    Callout,
    Heading,
    KnowledgeReference,
    Paragraph,
    ProposalDocument,
    RequirementMatrix,
    Section,
    Table,
    VisualPlaceholder,
)


class _CompositionModel(BaseModel):
    """Immutable base for renderer-independent composition contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


CompositionNode: TypeAlias = Annotated[
    Heading | Paragraph | BulletList | Table | VisualPlaceholder | Callout | RequirementMatrix,
    Field(discriminator="kind"),
]


class ComponentContent(_CompositionModel):
    """One unchanged semantic IR node assigned to a component slot."""

    content: CompositionNode


class ComponentSlot(_CompositionModel):
    """An ordered semantic-content location within one component instance."""

    slot_name: str
    contents: tuple[ComponentContent, ...] = ()


class ComponentReference(_CompositionModel):
    """A source reference preserved for a composed document."""

    reference_node_id: UUID
    reference_id: str
    title: str
    source: str
    locator: str | None = None


class ComponentInstance(_CompositionModel):
    """One resolved semantic component instance in document order."""

    component_name: str
    component_id: UUID
    section_id: str
    slots: tuple[ComponentSlot, ...]
    children: tuple[ComponentInstance, ...] = ()
    metadata: tuple[tuple[str, str | bool], ...] = ()


class ComponentHierarchy(_CompositionModel):
    """Ordered root-component identities for the composed document tree."""

    root_component_ids: tuple[UUID, ...] = ()


class CompositionDocument(_CompositionModel):
    """Deterministic semantic assembly consumed by a future document backend."""

    composition_id: UUID
    proposal_id: str
    title: str
    components: tuple[ComponentInstance, ...]
    hierarchy: ComponentHierarchy
    references: tuple[ComponentReference, ...] = ()


class CompositionPlanMismatch(ValueError):
    """Proposal Plan and Proposal IR do not describe the same section tree."""


class CompositionEngine:
    """Assemble Proposal Plan and Proposal IR into a semantic composition model."""

    def __init__(self, component_registry: ComponentRegistry | None = None) -> None:
        self._component_registry = component_registry or ComponentRegistry.standard()

    def compose(self, plan: ProposalPlan, document: ProposalDocument) -> CompositionDocument:
        """Compose matching plan and IR documents without changing semantic content."""
        if not isinstance(plan, ProposalPlan):
            raise TypeError("plan must be a ProposalPlan")
        if not isinstance(document, ProposalDocument):
            raise TypeError("document must be a ProposalDocument")
        if plan.proposal_id != document.proposal_id:
            raise CompositionPlanMismatch("plan.proposal_id must match document.proposal_id")

        section_plans = self._section_plans_by_id(plan)
        document_section_ids = tuple(section.section_id for section in self._walk_sections(document.sections))
        if tuple(section.section_id for section in plan.sections) != document_section_ids:
            raise CompositionPlanMismatch("plan sections must exactly match document section order and hierarchy")

        components = tuple(
            self._compose_section(section, section_plans, document)
            for section in document.sections
        )
        return CompositionDocument(
            composition_id=uuid5(NAMESPACE_URL, f"composition:{document.document_id}:{plan.proposal_id}"),
            proposal_id=document.proposal_id,
            title=document.title,
            components=components,
            hierarchy=ComponentHierarchy(root_component_ids=tuple(component.component_id for component in components)),
            references=tuple(self._compose_reference(reference) for reference in document.references),
        )

    @staticmethod
    def _section_plans_by_id(plan: ProposalPlan) -> dict[str, SectionPlan]:
        section_plans = {section.section_id: section for section in plan.sections}
        if len(section_plans) != len(plan.sections):
            raise CompositionPlanMismatch("plan contains duplicate section_id values")
        return section_plans

    def _compose_section(
        self,
        section: Section,
        section_plans: dict[str, SectionPlan],
        document: ProposalDocument,
    ) -> ComponentInstance:
        section_plan = section_plans[section.section_id]
        definition = self._component_registry.resolve(section_plan.role)
        children = tuple(
            self._compose_section(child, section_plans, document)
            for child in section.children
        )
        return ComponentInstance(
            component_name=definition.component_name,
            component_id=uuid5(
                NAMESPACE_URL,
                f"component:{document.document_id}:{section.node_id}:{definition.component_name}",
            ),
            section_id=section.section_id,
            slots=(
                ComponentSlot(slot_name="heading", contents=(ComponentContent(content=section.heading),)),
                ComponentSlot(
                    slot_name="content",
                    contents=tuple(ComponentContent(content=block) for block in section.blocks),
                ),
            ),
            children=children,
            metadata=(
                ("section_role", section_plan.role.value),
                ("page_break_before", section_plan.page_break_before),
                ("include_in_toc", section_plan.include_in_toc),
                ("numbering_enabled", section_plan.numbering_enabled),
            ),
        )

    @staticmethod
    def _walk_sections(sections: tuple[Section, ...]):
        for section in sections:
            yield section
            yield from CompositionEngine._walk_sections(section.children)

    @staticmethod
    def _compose_reference(reference: KnowledgeReference) -> ComponentReference:
        return ComponentReference(
            reference_node_id=reference.node_id,
            reference_id=reference.reference_id,
            title=reference.title,
            source=reference.source,
            locator=reference.locator,
        )
