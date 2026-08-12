"""Ollama provider adapter implementing the LLMProvider protocol.

This adapter calls the local Ollama REST API at /api/generate and maps the
official sync response schema into the `GeneratedResponse` DTO.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ...contracts.synthesis import PromptPackage
from ...contracts.response import GeneratedResponse
from .provider import LLMProvider, ProviderUnavailableError, LLMGenerationError


@dataclass
class OllamaProvider:
    """Adapter for the Ollama local REST API."""

    base_url: str = "http://localhost:11434/api"
    model: str = field(default_factory=lambda: os.getenv("BDIL_OLLAMA_MODEL", "qwen2.5:3b"))
    timeout: int = 60
    temperature: float = 0.1
    _provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_base_url = self.base_url.rstrip("/")
        self.base_url = (
            normalized_base_url
            if normalized_base_url.endswith("/api")
            else f"{normalized_base_url}/api"
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        return dict(self._provider_metadata or {})

    def health_check(self) -> bool:
        url = f"{self.base_url}/tags"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=1.0) as response:
                return response.getcode() == 200
        except (HTTPError, URLError, OSError):
            return False

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        if not isinstance(prompt_package, PromptPackage):
            raise TypeError("prompt_package must be a PromptPackage")

        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model,
            "system": prompt_package.system_prompt,
            "prompt": self._flatten_prompt(prompt_package),
            "stream": False,
            "options": {"temperature": float(self.temperature)},
        }

        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
        except HTTPError as error:
            raise ProviderUnavailableError(f"ollama returned HTTP {error.code}") from error
        except URLError as error:
            raise ProviderUnavailableError("ollama connection failed") from error
        except OSError as error:
            raise ProviderUnavailableError("ollama request failed") from error
        latency_ms = (perf_counter() - start) * 1000.0

        if status_code != 200:
            raise ProviderUnavailableError(f"ollama returned HTTP {status_code}")

        try:
            data = json.loads(raw_body)
        except ValueError as error:
            raise LLMGenerationError("ollama returned invalid JSON") from error

        if not isinstance(data, dict):
            raise LLMGenerationError("ollama returned malformed response")

        required_keys = (
            "response",
            "done_reason",
            "model",
            "prompt_eval_count",
            "eval_count",
        )
        if not all(key in data for key in required_keys):
            raise LLMGenerationError("ollama response missing required fields")

        generated_text = data["response"]
        finish_reason = data["done_reason"]
        model_name = data["model"]
        prompt_tokens = int(data["prompt_eval_count"])
        completion_tokens = int(data["eval_count"])
        total_duration = data.get("total_duration")

        return GeneratedResponse(
            generated_text=str(generated_text),
            provider_name=self.provider_name,
            model_name=str(model_name),
            finish_reason=str(finish_reason),
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            latency_ms=latency_ms,
            generation_timestamp=datetime.now(timezone.utc),
            generation_metadata={
                "provider_metadata": self.provider_metadata,
                "total_duration": total_duration,
            },
        )

    @staticmethod
    def _flatten_prompt(prompt_package: PromptPackage) -> str:
        """Render the structured prompt contract as Ollama's single prompt string."""
        sections = [
            "Context\n"
            "----------------------------------------\n"
            f"{prompt_package.assembled_context.assembled_context}",
        ]
        history = "\n".join(item for item in prompt_package.conversation_history if item)
        if history:
            sections.append(
                "Conversation History\n"
                "----------------------------------------\n"
                f"{history}"
            )
        sections.append(
            "User Question\n"
            "----------------------------------------\n"
            f"{prompt_package.user_prompt}"
        )
        return "\n\n".join(sections)
