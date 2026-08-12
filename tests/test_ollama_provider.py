"""Unit tests for the OllamaProvider adapter.

Tests mock urllib.request to avoid real HTTP calls.
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
from urllib.error import HTTPError, URLError

import pytest

from unittest.mock import Mock

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online import QueryEngine
from proposal_ai_agent.online.contracts.synthesis import PromptPackage
from proposal_ai_agent.online.contracts.retrieval import ProcessedCandidate, RetrievedContext
from proposal_ai_agent.online.providers.llm.ollama import OllamaProvider
from proposal_ai_agent.online.contracts.response import GeneratedResponse
from proposal_ai_agent.online.providers.llm.factory import LLMProviderFactory
from proposal_ai_agent.online.providers.llm.provider import ProviderUnavailableError, LLMGenerationError


def _request():
    query_engine = QueryEngine(
        embedding_provider=MockEmbeddingProvider(dimensions=3),
        embedding_dimension=3,
        embedding_model_id="shared-offline-model",
    )
    user_query = query_engine.receive_query("What applies? document: handbook")
    return query_engine.plan_retrieval(
        query_engine.embed_query(query_engine.process_query(query_engine.qualify_query(user_query)))
    )


def _processed_candidate(*, chunk_id: str, text: str) -> ProcessedCandidate:
    return ProcessedCandidate(
        chunk_id=chunk_id,
        document_id="document-1",
        text=text,
        score=0.5,
        metadata={"page": 1},
        header_path=("Policies",),
        chunk_index=0,
        processing_flags={"metadata_validated": True, "merged_chunk_count": 1},
    )


def _retrieved_context(request, *candidates):
    return RetrievedContext(
        retrieval_request=request,
        candidates=candidates,
        final_context="\n\n".join(candidate.text for candidate in candidates),
        total_candidates=len(candidates),
        returned_candidates=len(candidates),
        reranking_applied=False,
        reranking_time_ms=0.0,
        retrieval_summary={
            "processing_summary": {},
            "candidate_budget": request.candidate_budget,
            "top_k": request.top_k,
            "reranking_pool_size": 0,
        },
    )


def _prompt_package(request):
    from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine

    assembled = SynthesisEngine().assemble(
        _retrieved_context(request, _processed_candidate(chunk_id="chunk-1", text="Content"))
    )
    return PromptPackage(
        system_prompt="Enterprise assistant.",
        user_prompt=f"Question: {request.query_embedding.processed_query.normalized_query}",
        conversation_history=("User: Hello",),
        assembled_context=assembled,
        prompt_template="{system_prompt}\n{context_block}\n{user_prompt}",
        output_format="markdown",
        prompt_statistics={"system_tokens": 1, "user_tokens": 2, "history_tokens": 3, "context_tokens": 4, "total_tokens": 10},
        generation_metadata={"benchmark_id": request.query_embedding.processed_query.qualified_query.benchmark_id},
        validation_result=request.query_embedding.processed_query.qualified_query.validation_result,
    )


class DummyResponse:
    def __init__(self, status_code: int, body: bytes):
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


def test_ollama_provider_success(monkeypatch) -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider(base_url="http://localhost:11434/api", model="test-model", timeout=5)

    def fake_urlopen(request_obj, timeout):
        body = json.dumps(
            {
                "response": "generated answer",
                "done_reason": "completed",
                "model": "test-model",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 1.23,
            }
        ).encode("utf-8")
        return DummyResponse(200, body)

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    response = provider.generate(prompt)

    assert isinstance(response, GeneratedResponse)
    assert response.provider_name == "ollama"
    assert response.model_name == "test-model"
    assert response.generated_text == "generated answer"
    assert response.token_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert response.generation_metadata["total_duration"] == 1.23


def test_ollama_provider_normalizes_host_url_for_health_and_generation(monkeypatch) -> None:
    provider = OllamaProvider(base_url="http://localhost:11434", model="test-model")
    seen_urls: list[str] = []

    def fake_urlopen(request_obj, timeout):
        seen_urls.append(request_obj.full_url)
        if request_obj.get_method() == "GET":
            return DummyResponse(200, b'{}')
        return DummyResponse(
            200,
            json.dumps(
                {
                    "response": "generated answer",
                    "done_reason": "stop",
                    "model": "test-model",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
            ).encode("utf-8"),
        )

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    assert provider.health_check() is True
    provider.generate(_prompt_package(_request()))

    assert seen_urls == ["http://localhost:11434/api/tags", "http://localhost:11434/api/generate"]


def test_ollama_provider_invalid_json(monkeypatch) -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider()

    def fake_urlopen(request_obj, timeout):
        return DummyResponse(200, b"not-json")

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    with pytest.raises(LLMGenerationError):
        provider.generate(prompt)


def test_ollama_provider_missing_field(monkeypatch) -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider()

    def fake_urlopen(request_obj, timeout):
        body = json.dumps(
            {
                "done_reason": "completed",
                "model": "test-model",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        ).encode("utf-8")
        return DummyResponse(200, body)

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    with pytest.raises(LLMGenerationError):
        provider.generate(prompt)


def test_ollama_provider_timeout(monkeypatch) -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider(timeout=0.01)

    def fake_urlopen(request_obj, timeout):
        raise URLError("timed out")

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    with pytest.raises(ProviderUnavailableError):
        provider.generate(prompt)


def test_ollama_provider_connection_failure(monkeypatch) -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider()

    def fake_urlopen(request_obj, timeout):
        raise URLError("connection failed")

    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)

    with pytest.raises(ProviderUnavailableError):
        provider.generate(prompt)


def test_ollama_provider_factory_registration() -> None:
    factory = LLMProviderFactory()
    assert "ollama" in factory.supported_providers()
    provider = factory.create("ollama", base_url="http://localhost:11434/api")
    assert provider.provider_name == "ollama"


def test_ollama_generated_response_immutable() -> None:
    request = _request()
    prompt = _prompt_package(request)
    provider = OllamaProvider()

    def fake_urlopen(request_obj, timeout):
        body = json.dumps(
            {
                "response": "generated answer",
                "done_reason": "completed",
                "model": "test-model",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 0.5,
            }
        ).encode("utf-8")
        return DummyResponse(200, body)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("proposal_ai_agent.online.providers.llm.ollama.urlopen", fake_urlopen)
    try:
        response = provider.generate(prompt)
    finally:
        monkeypatch.undo()

    assert response == replace(response)
    with pytest.raises(AttributeError):
        response.generated_text = "changed"
