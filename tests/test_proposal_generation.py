"""Contracts and deterministic questionnaire mapping for proposal generation."""

from uuid import uuid4

import pytest

from proposal_ai_agent.proposal_generation import (
    ClientInformation,
    Constraint,
    ProjectModel,
    ProposalRequest,
    QuestionnaireMapper,
    Requirement,
    map_proposal_request,
)


def _request() -> ProposalRequest:
    return ProposalRequest(
        request_id=uuid4(),
        client=ClientInformation(name="Example Client", attributes={"industry": "energy"}),
        proposal_title="Modernization Proposal",
        project_data={"project_name": "Northstar", "budget": {"currency": "USD", "amount": 500000}},
        requirements=(Requirement(identifier="R-1", description="Provide an implementation plan", priority="high"),),
        constraints=(Constraint(identifier="C-1", description="Complete within the agreed timeline"),),
        metadata={"submitted_by": "proposal-team"},
    )


def test_proposal_request_is_generic_and_immutable() -> None:
    request = _request()

    assert request.client.name == "Example Client"
    assert request.project_data["project_name"] == "Northstar"
    assert request.requirements[0].identifier == "R-1"
    with pytest.raises(Exception):
        request.proposal_title = "Changed"  # type: ignore[misc]


def test_questionnaire_mapping_preserves_user_supplied_project_data() -> None:
    request = _request()

    project = map_proposal_request(request)

    assert isinstance(project, ProjectModel)
    assert project.source_request_id == request.request_id
    assert project.client == request.client
    assert project.project_data == request.project_data
    assert project.requirements == request.requirements
    assert QuestionnaireMapper.map(request) == project


def test_questionnaire_mapping_rejects_non_request_input() -> None:
    with pytest.raises(TypeError, match="ProposalRequest"):
        map_proposal_request(object())  # type: ignore[arg-type]


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(Exception):
        ClientInformation(name="Example Client", unsupported=True)  # type: ignore[call-arg]
