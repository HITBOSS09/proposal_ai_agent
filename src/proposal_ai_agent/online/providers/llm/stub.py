"""Deterministic in-process stub LLM provider for testing the response pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Mapping

from ...contracts.synthesis import PromptPackage
from ...contracts.response import GeneratedResponse
from .provider import LLMProvider, ProviderUnavailableError


class StubLLMProvider:
    """A deterministic, side-effect-free LLM provider used for testing.

    - Implements the `LLMProvider` protocol.
    - Never performs network I/O or invokes external SDKs.
    - Returns a deterministic `GeneratedResponse` for any valid `PromptPackage`.
    """

    def __init__(self, available: bool = True, provider_metadata: Mapping[str, Any] | None = None) -> None:
        self._available = bool(available)
        self._provider_metadata = dict(provider_metadata or {})
        self.generated_calls = 0

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        return dict(self._provider_metadata)

    def health_check(self) -> bool:
        return self._available

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        if not isinstance(prompt_package, PromptPackage):
            raise TypeError("prompt_package must be a PromptPackage")
        if not self._available:
            raise ProviderUnavailableError("stub provider unavailable")

        self.generated_calls += 1
        start = perf_counter()

        # Deterministic response text derived from the prompt package metadata
        benchmark = prompt_package.generation_metadata.get("benchmark_id", "unknown")
        user_part = prompt_package.user_prompt or ""
        generated_text = f"[stub] benchmark={benchmark} | {user_part}".strip()

        # Use provided prompt statistics when available to compute token usage
        prompt_tokens = int(prompt_package.prompt_statistics.get("total_tokens", 0))
        completion_tokens = 5
        total_tokens = prompt_tokens + completion_tokens

        latency_ms = max(0.0, (perf_counter() - start) * 1000.0) or 1.0

        generation_metadata = {"provider_metadata": self.provider_metadata}

        return GeneratedResponse(
            generated_text=generated_text,
            provider_name=self.provider_name,
            model_name=self.model_name,
            finish_reason="completed",
            token_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
            latency_ms=latency_ms,
            generation_timestamp=datetime.now(timezone.utc),
            generation_metadata=generation_metadata,
        )
