"""Unit tests for the structured ProposalResponse Ollama adapter."""

import json

import pytest

from proposal_ai_agent.proposal_generation.models import ClientInformation, ProjectModel
from proposal_ai_agent.proposal_generation.prompt_composer import PromptComposer
from proposal_ai_agent.proposal_generation.providers import ProposalGenerationError, ProposalLLMFactory, ProposalOllamaAdapter
from proposal_ai_agent.proposal_generation.transport_contract import ProposalResponse


class DummyResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def getcode(self) -> int:
        return self.status_code

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _prompt():
    return PromptComposer().compose(
        ProjectModel(source_request_id="00000000-0000-0000-0000-000000000001", client=ClientInformation(name="Client"), proposal_title="Proposal")
    )


def _transport_payload() -> dict[str, object]:
    return {
        "proposal_id": "00000000-0000-0000-0000-000000000001",
        "title": "Proposal",
        "sections": [{"section_id": "cover", "heading": {"text": "Proposal", "level": 1}}],
    }


def test_adapter_maps_ollama_structured_response_and_request_contract(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse(200, json.dumps({"response": json.dumps(_transport_payload())}).encode("utf-8"))

    monkeypatch.setattr("proposal_ai_agent.proposal_generation.providers.ollama.urlopen", fake_urlopen)
    response = ProposalOllamaAdapter(base_url="http://localhost:11434", model="proposal-model").generate(_prompt())

    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["payload"]["stream"] is False  # type: ignore[index]
    response_schema = seen["payload"]["format"]  # type: ignore[index]
    section_schema = response_schema["properties"]["sections"]
    expected_ids = _prompt().planned_section_ids
    assert section_schema["minItems"] == section_schema["maxItems"] == len(expected_ids)
    assert "sections" in response_schema["required"]
    assert response_schema["properties"]["proposal_id"]["const"] == _prompt().proposal_id
    assert response_schema["$defs"]["SectionResponse"]["properties"]["section_id"]["enum"] == list(expected_ids)
    assert response_schema["$defs"]["SectionResponse"]["properties"]["children"]["maxItems"] == 0
    assert isinstance(response, ProposalResponse)
    assert response.title == "Proposal"
    assert response.proposal_id == _prompt().proposal_id


@pytest.mark.parametrize("model_value", [None, "", "model-invented-id"])
def test_adapter_assigns_application_owned_proposal_id_before_transport_construction(
    monkeypatch, model_value,
) -> None:
    payload = _transport_payload()
    if model_value is None:
        payload.pop("proposal_id")
    else:
        payload["proposal_id"] = model_value

    def fake_urlopen(request, timeout):
        return DummyResponse(
            200,
            json.dumps({"response": json.dumps(payload)}).encode("utf-8"),
        )

    monkeypatch.setattr("proposal_ai_agent.proposal_generation.providers.ollama.urlopen", fake_urlopen)
    prompt = _prompt()
    response = ProposalOllamaAdapter().generate(prompt)

    assert response.proposal_id == prompt.proposal_id


def test_application_metadata_binding_does_not_patch_other_missing_llm_fields(monkeypatch) -> None:
    payload = _transport_payload()
    payload.pop("title")

    def fake_urlopen(request, timeout):
        return DummyResponse(
            200,
            json.dumps({"response": json.dumps(payload)}).encode("utf-8"),
        )

    monkeypatch.setattr("proposal_ai_agent.proposal_generation.providers.ollama.urlopen", fake_urlopen)
    with pytest.raises(ProposalGenerationError, match="invalid ProposalResponse"):
        ProposalOllamaAdapter().generate(_prompt())


def test_adapter_surfaces_invalid_transport_and_factory_resolves_adapter(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return DummyResponse(200, json.dumps({"response": "not-a-proposal"}).encode("utf-8"))

    monkeypatch.setattr("proposal_ai_agent.proposal_generation.providers.ollama.urlopen", fake_urlopen)
    with pytest.raises(ProposalGenerationError):
        ProposalOllamaAdapter().generate(_prompt())

    provider = ProposalLLMFactory().create("ollama", model="configured-model")
    assert isinstance(provider, ProposalOllamaAdapter)
    assert provider.model_name == "configured-model"


def test_adapter_uses_runtime_configuration_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("BDIL_PROPOSAL_OLLAMA_TIMEOUT", raising=False)
    monkeypatch.delenv("BDIL_PROPOSAL_NUM_PREDICT", raising=False)
    assert ProposalOllamaAdapter().timeout == 300
    assert ProposalOllamaAdapter().num_predict == 1200

    monkeypatch.setenv("BDIL_PROPOSAL_OLLAMA_TIMEOUT", "450")
    monkeypatch.setenv("BDIL_PROPOSAL_NUM_PREDICT", "800")
    assert ProposalOllamaAdapter().timeout == 450
    assert ProposalOllamaAdapter().num_predict == 800
