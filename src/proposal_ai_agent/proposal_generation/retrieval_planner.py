"""Deterministic retrieval-policy planning for proposal sections."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import ProposalPlan, SectionPlan
from .retrieval_query import ReferenceType, RetrievalStrategy, SectionRetrievalQuery


class _RetrievalPlanningModel(BaseModel):
    """Frozen base model for retrieval instructions only."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SectionRetrievalPolicy(_RetrievalPlanningModel):
    """Retrieval requirements for a single planned proposal section."""

    section_id: str = Field(min_length=1)
    authoring_references_required: bool
    technical_references_required: bool
    blueprint_references_required: bool
    user_input_only: bool
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class RetrievalPlan(_RetrievalPlanningModel):
    """Ordered retrieval instructions; this model never performs retrieval."""

    proposal_plan: ProposalPlan
    section_policies: tuple[SectionRetrievalPolicy, ...]


class RetrievalPlanner:
    """Convert section flags into deterministic, non-executing retrieval policies."""

    def plan(self, proposal_plan: ProposalPlan) -> RetrievalPlan:
        if not isinstance(proposal_plan, ProposalPlan):
            raise TypeError("proposal_plan must be a ProposalPlan")
        return RetrievalPlan(
            proposal_plan=proposal_plan,
            section_policies=tuple(self._policy(section) for section in proposal_plan.sections),
        )

    def queries(
        self,
        retrieval_plan: RetrievalPlan,
        *,
        max_results: int = 3,
    ) -> tuple[SectionRetrievalQuery, ...]:
        """Create deterministic typed queries for every required reference class."""
        if not isinstance(retrieval_plan, RetrievalPlan):
            raise TypeError("retrieval_plan must be a RetrievalPlan")
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        sections = {section.section_id: section for section in retrieval_plan.proposal_plan.sections}
        queries: list[SectionRetrievalQuery] = []
        for policy in retrieval_plan.section_policies:
            section = sections[policy.section_id]
            reference_types = (
                (ReferenceType.AUTHORING, policy.authoring_references_required),
                (ReferenceType.TECHNICAL, policy.technical_references_required),
                (ReferenceType.BLUEPRINT, policy.blueprint_references_required),
            )
            for reference_type, required in reference_types:
                if required:
                    queries.append(
                        SectionRetrievalQuery(
                            section_id=section.section_id,
                            reference_type=reference_type,
                            query_text=self._query_text(retrieval_plan.proposal_plan, section, reference_type),
                            max_results=max_results,
                            retrieval_strategy=RetrievalStrategy.DENSE,
                            rerank_enabled=False,
                        )
                    )
        return tuple(queries)

    @staticmethod
    def _query_text(
        proposal_plan: ProposalPlan,
        section: SectionPlan,
        reference_type: ReferenceType,
    ) -> str:
        project = proposal_plan.project
        parts = [section.title, reference_type.value, str(proposal_plan.metadata.get("proposal_type", "general"))]
        if reference_type is ReferenceType.TECHNICAL:
            parts.extend(requirement.description for requirement in project.requirements)
            parts.extend(constraint.description for constraint in project.constraints)
        return " | ".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _policy(section: SectionPlan) -> SectionRetrievalPolicy:
        requires_user_input = bool(section.metadata.get("requires_user_input", False))
        requires_reference = (
            section.authoring_reference
            or section.technical_reference
            or section.template_reference
        )
        return SectionRetrievalPolicy(
            section_id=section.section_id,
            authoring_references_required=section.authoring_reference,
            technical_references_required=section.technical_reference,
            blueprint_references_required=section.template_reference,
            user_input_only=requires_user_input and not requires_reference,
            metadata={"order": section.metadata.get("order")},
        )
