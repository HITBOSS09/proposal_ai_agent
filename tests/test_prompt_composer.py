"""Tests for complete structured-transport prompt composition."""

from pathlib import Path

import pytest

from proposal_ai_agent.proposal_generation.models import ClientInformation, ProjectModel, Requirement
from proposal_ai_agent.proposal_generation.prompt_composer import PromptComposer, SectionPromptPackage
from proposal_ai_agent.proposal_generation.proposal_planner import ProposalPlanner
from proposal_ai_agent.proposal_generation.retrieval_context import RetrievedContext
from proposal_ai_agent.proposal_generation.retrieval_query import ReferenceType, SectionRetrievalQuery
from proposal_ai_agent.proposal_generation.prompt_composer import RetrievedReference
from proposal_ai_agent.proposal_generation.transport_contract import ProposalResponse


def _project() -> ProjectModel:
    return ProjectModel(
        source_request_id="7d2e5f0f-dcca-4c93-9de8-2d7e04e33142",
        client=ClientInformation(name="Example Client"),
        proposal_title="Example Proposal",
        project_data={"project_name": "Northstar"},
        requirements=(Requirement(identifier="REQ-1", description="Provide an implementation plan"),),
    )


def test_composer_returns_one_complete_proposal_transport_prompt() -> None:
    package = PromptComposer().compose(_project())

    assert isinstance(package, SectionPromptPackage)
    assert package.proposal_id == "7d2e5f0f-dcca-4c93-9de8-2d7e04e33142"
    assert package.expected_output_model is ProposalResponse
    assert package.metadata["section_count"] == 7
    assert "section_id: executive-summary" in package.user_prompt
    assert "ProposalResponse" in package.user_prompt
    assert "Markdown" in package.system_prompt


def test_composer_preserves_user_facts_without_legacy_output_contract() -> None:
    package = PromptComposer().compose(_project())

    assert "Example Client" in package.user_prompt
    assert "Northstar" in package.user_prompt
    assert "Provide an implementation plan" in package.user_prompt
    assert "ProposalContent" not in package.user_prompt
    assert "[TECHNICAL_REFERENCE]" not in package.user_prompt


def test_composer_rejects_non_project_input() -> None:
    with pytest.raises(TypeError, match="ProjectModel"):
        PromptComposer().compose(object())  # type: ignore[arg-type]


def test_composer_includes_typed_references_without_weakening_user_fact_priority() -> None:
    project = _project()
    plan = ProposalPlanner().plan(project)
    query = SectionRetrievalQuery(
        section_id="executive-summary",
        reference_type=ReferenceType.AUTHORING,
        query_text="summary style",
        max_results=1,
    )
    reference = RetrievedReference(
        reference_id="ref-1",
        reference_type="authoring",
        chunk_id="chunk-1",
        source_document="sample.docx",
        score=0.8,
        content="Different Client, different project, budget 10 million, quantity 99.",
    )
    context = RetrievedContext(
        section_id="executive-summary",
        queries=(query,),
        style_references=(reference,),
    )

    package = PromptComposer().compose(project, plan, (context,))

    assert package.style_references == (reference,)
    assert "Example Client" in package.user_prompt
    assert "Northstar" in package.user_prompt
    assert "[AUTHORING_REFERENCES]" in package.user_prompt
    assert "Different Client" in package.user_prompt
    assert "never copy their client names" in package.system_prompt
    assert package.metadata["section_count"] == len(plan.sections)


def test_composer_uses_only_dynamic_plan_ids_for_exact_section_contract() -> None:
    project = _project()
    plan = ProposalPlanner().plan(project)

    package = PromptComposer().compose(project, plan)

    expected_ids = tuple(section.section_id for section in plan.sections)
    contract = package.user_prompt.split("[PLANNED_SECTION_CONTRACT]", 1)[1].split("[OUTPUT_CONTRACT]", 1)[0]
    positions = tuple(contract.index(f"section_id: {section_id}") for section_id in expected_ids)
    assert positions == tuple(sorted(positions))
    assert "exactly once, in exactly this order" in contract
    assert "Do not rename, add, omit, duplicate, nest, or reorder sections" in contract

    source = (Path(__file__).resolve().parents[1] / "src/proposal_ai_agent/proposal_generation/prompt_composer.py").read_text()
    assert not any(f'("{section_id}",' in source for section_id in expected_ids)
