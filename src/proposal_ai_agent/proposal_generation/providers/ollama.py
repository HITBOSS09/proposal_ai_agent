"""Ollama adapter for structured proposal transport responses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..prompt_composer import SectionPromptPackage
from ..transport_contract import ProposalResponse
from .provider import ProposalGenerationError, ProposalProviderUnavailableError


def _environment_positive_int(name: str, default: int) -> int:
    """Read a positive integer runtime setting without changing adapter callers."""
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass
class ProposalOllamaAdapter:
    """Map a complete proposal prompt to Ollama's structured JSON response."""

    base_url: str = "http://localhost:11434/api"
    model: str = field(default_factory=lambda: os.getenv("BDIL_PROPOSAL_OLLAMA_MODEL", os.getenv("BDIL_OLLAMA_MODEL", "qwen2.5:3b")))
    timeout: int = field(default_factory=lambda: _environment_positive_int("BDIL_PROPOSAL_OLLAMA_TIMEOUT", 300))
    temperature: float = 0.1
    num_predict: int = field(default_factory=lambda: _environment_positive_int("BDIL_PROPOSAL_NUM_PREDICT", 1200))
    _provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.num_predict <= 0:
            raise ValueError("num_predict must be positive")
        normalized_base_url = self.base_url.rstrip("/")
        self.base_url = normalized_base_url if normalized_base_url.endswith("/api") else f"{normalized_base_url}/api"

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        return dict(self._provider_metadata or {})

    def generate(self, prompt: SectionPromptPackage) -> ProposalResponse:
        """Request and parse exactly one ``ProposalResponse`` transport object."""
        if not isinstance(prompt, SectionPromptPackage):
            raise TypeError("prompt must be a SectionPromptPackage")
        payload = {
            "model": self.model,
            "system": prompt.system_prompt,
            "prompt": prompt.user_prompt,
            "format": self._planned_response_schema(prompt),
            "stream": False,
            "options": {"temperature": float(self.temperature), "num_predict": int(self.num_predict)},
        }
        request = Request(
            f"{self.base_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
        except HTTPError as error:
            raise ProposalProviderUnavailableError(f"ollama returned HTTP {error.code}") from error
        except URLError as error:
            raise ProposalProviderUnavailableError("ollama connection failed") from error
        except OSError as error:
            raise ProposalProviderUnavailableError("ollama request failed") from error
        if status_code != 200:
            raise ProposalProviderUnavailableError(f"ollama returned HTTP {status_code}")
        try:
            envelope = json.loads(raw_body)
        except ValueError as error:
            raise ProposalGenerationError("ollama returned invalid JSON") from error
        if not isinstance(envelope, dict) or not isinstance(envelope.get("response"), str):
            raise ProposalGenerationError("ollama response missing structured proposal payload")
        try:
            structured_payload = json.loads(envelope["response"])
            if not isinstance(structured_payload, dict):
                raise ValueError("structured proposal payload must be an object")
            # proposal_id is application-owned correlation metadata derived from
            # ProjectModel.source_request_id.  The model authors proposal content;
            # it never owns, invents, or overrides this identifier.
            structured_payload["proposal_id"] = prompt.proposal_id
            return ProposalResponse.model_validate(structured_payload)
        except ValueError as error:
            raise ProposalGenerationError("ollama returned an invalid ProposalResponse") from error

    @staticmethod
    def _planned_response_schema(prompt: SectionPromptPackage) -> dict[str, Any]:
        """Constrain the existing transport schema to the canonical flat section plan."""
        schema = ProposalResponse.model_json_schema()
        sections = schema["properties"]["sections"]
        section_reference = {"$ref": "#/$defs/SectionResponse"}
        sections.pop("default", None)
        sections["minItems"] = len(prompt.planned_section_ids)
        sections["maxItems"] = len(prompt.planned_section_ids)
        sections["items"] = section_reference
        required = schema.setdefault("required", [])
        if "sections" not in required:
            required.append("sections")
        schema["properties"]["proposal_id"]["const"] = prompt.proposal_id
        schema["$defs"]["SectionResponse"]["properties"]["section_id"]["enum"] = list(
            prompt.planned_section_ids
        )
        section_children = schema["$defs"]["SectionResponse"]["properties"]["children"]
        section_children.pop("default", None)
        section_children["maxItems"] = 0
        return schema
