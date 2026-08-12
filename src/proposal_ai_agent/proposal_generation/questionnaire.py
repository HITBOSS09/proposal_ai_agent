"""Deterministic mapping from questionnaire submissions to project contracts."""

from __future__ import annotations

from .models import ProjectModel, ProposalRequest


def map_proposal_request(request: ProposalRequest) -> ProjectModel:
    """Normalize one validated questionnaire submission without inference or I/O."""
    if not isinstance(request, ProposalRequest):
        raise TypeError("request must be a ProposalRequest")
    return ProjectModel(
        source_request_id=request.request_id,
        client=request.client,
        proposal_title=request.proposal_title,
        project_data=request.project_data,
        requirements=request.requirements,
        constraints=request.constraints,
        metadata=request.metadata,
    )


class QuestionnaireMapper:
    """Explicit mapper boundary reserved for future questionnaire transport layers."""

    @staticmethod
    def map(request: ProposalRequest) -> ProjectModel:
        """Map a proposal request using the deterministic contract transformation."""
        return map_proposal_request(request)
