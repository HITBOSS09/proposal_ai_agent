"""Unit tests for structured proposal generation through the provider port."""

import pytest

from proposal_ai_agent.proposal_generation.prompt_composer import PromptComposer
from proposal_ai_agent.proposal_generation.models import ClientInformation, ProjectModel
from proposal_ai_agent.proposal_generation.providers import ProposalGenerationError
from proposal_ai_agent.proposal_generation.section_generator import SectionGenerator
from proposal_ai_agent.proposal_generation.transport_contract import HeadingResponse, ProposalResponse, SectionResponse


def _prompt():
    return PromptComposer().compose(
        ProjectModel(source_request_id="00000000-0000-0000-0000-000000000001", client=ClientInformation(name="Client"), proposal_title="Proposal")
    )


class MockProposalProvider:
    provider_name = "mock"
    model_name = "mock-model"
    provider_metadata = {}

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt) -> ProposalResponse:
        self.calls += 1
        return ProposalResponse(
            proposal_id=prompt.proposal_id,
            title="Proposal",
            sections=(SectionResponse(section_id="cover", heading=HeadingResponse(text="Proposal", level=1)),),
        )


def test_section_generator_calls_provider_once_and_returns_transport_response() -> None:
    provider = MockProposalProvider()

    response = SectionGenerator(provider).generate(_prompt())

    assert provider.calls == 1
    assert isinstance(response, ProposalResponse)
    assert response.proposal_id == "00000000-0000-0000-0000-000000000001"


def test_section_generator_propagates_provider_errors() -> None:
    class FailingProvider(MockProposalProvider):
        def generate(self, prompt) -> ProposalResponse:
            raise ProposalGenerationError("provider output invalid")

    with pytest.raises(ProposalGenerationError, match="provider output invalid"):
        SectionGenerator(FailingProvider()).generate(_prompt())
