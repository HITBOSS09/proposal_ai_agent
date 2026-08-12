"""Deterministic planning contracts for enterprise proposal composition."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import ProjectModel, ProposalPlan, SectionPlan


class _PlanningModel(BaseModel):
    """Frozen base model for planning-only values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BlueprintSelection(_PlanningModel):
    """A future-ready blueprint candidate selected without loading a template."""

    blueprint_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    selection_reason: str = Field(min_length=1)


class VisualSlot(_PlanningModel):
    """A planned visual location; rendering is deliberately outside this layer."""

    slot_id: str = Field(min_length=1)
    slot_type: str = Field(min_length=1)
    mandatory: bool = False
    placement_hint: str | None = None
    caption_required: bool = False
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProposalPlanner:
    """Create deterministic, domain-agnostic proposal plans from project data."""

    _SECTION_DEFINITIONS: tuple[Mapping[str, Any], ...] = (
        {
            "section_id": "cover",
            "title": "Proposal Cover",
            "authoring_reference": False,
            "technical_reference": False,
            "template_reference": False,
            "document_role": "cover",
        },
        {
            "section_id": "executive-summary",
            "title": "Executive Summary",
            "authoring_reference": True,
            "technical_reference": False,
            "template_reference": True,
            "document_role": "body",
        },
        {
            "section_id": "solution-overview",
            "title": "Solution Overview",
            "authoring_reference": True,
            "technical_reference": True,
            "template_reference": True,
            "document_role": "body",
            "visual_slots": (
                VisualSlot(
                    slot_id="solution-overview-visual",
                    slot_type="diagram",
                    placement_hint="within-section",
                ),
            ),
        },
        {
            "section_id": "scope-and-deliverables",
            "title": "Scope and Deliverables",
            "authoring_reference": True,
            "technical_reference": False,
            "template_reference": True,
            "document_role": "body",
        },
        {
            "section_id": "delivery-plan",
            "title": "Delivery Plan",
            "authoring_reference": True,
            "technical_reference": False,
            "template_reference": True,
            "document_role": "body",
        },
        {
            "section_id": "commercials",
            "title": "Commercials",
            "authoring_reference": False,
            "technical_reference": False,
            "template_reference": True,
            "document_role": "body",
        },
        {
            "section_id": "references",
            "title": "References",
            "authoring_reference": False,
            "technical_reference": False,
            "template_reference": False,
            "document_role": "references",
        },
    )

    def plan(self, project: ProjectModel) -> ProposalPlan:
        """Create a stable section sequence without I/O, retrieval, or generation."""
        if not isinstance(project, ProjectModel):
            raise TypeError("project must be a ProjectModel")

        proposal_type = self._proposal_type(project.metadata)
        blueprint = self._select_blueprint(project.metadata)
        sections = tuple(
            self._section_plan(definition, order)
            for order, definition in enumerate(self._SECTION_DEFINITIONS, start=1)
        )
        return ProposalPlan(
            project=project,
            sections=sections,
            metadata={
                "proposal_type": proposal_type,
                "blueprint_selection": blueprint.model_dump(mode="json"),
                "planning_version": "1.0",
            },
        )

    @staticmethod
    def _proposal_type(metadata: Mapping[str, Any]) -> str:
        proposal_type = metadata.get("proposal_type")
        return proposal_type.strip() if isinstance(proposal_type, str) and proposal_type.strip() else "general"

    @staticmethod
    def _select_blueprint(metadata: Mapping[str, Any]) -> BlueprintSelection:
        blueprint_id = metadata.get("blueprint_id")
        if isinstance(blueprint_id, str) and blueprint_id.strip():
            return BlueprintSelection(
                blueprint_id=blueprint_id.strip(),
                confidence=1.0,
                selection_reason="explicit blueprint identifier supplied in project metadata",
            )
        return BlueprintSelection(
            confidence=0.0,
            selection_reason="no blueprint candidate supplied in project metadata",
        )

    @staticmethod
    def _section_plan(definition: Mapping[str, Any], order: int) -> SectionPlan:
        visual_slots = tuple(definition.get("visual_slots", ()))
        return SectionPlan(
            section_id=str(definition["section_id"]),
            title=str(definition["title"]),
            input_keys=("project_data", "requirements", "constraints"),
            authoring_reference=bool(definition["authoring_reference"]),
            technical_reference=bool(definition["technical_reference"]),
            template_reference=bool(definition["template_reference"]),
            metadata={
                "order": order,
                "display_name": definition["title"],
                "requires_user_input": True,
                "required_visual_slots": tuple(slot.model_dump(mode="json") for slot in visual_slots),
                "document_role": str(definition.get("document_role", "body")),
            },
        )
