"""Provider-neutral contract for LLM generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ...contracts.response import GeneratedResponse
from ...contracts.synthesis import PromptPackage


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider cannot execute generation due to availability."""


class LLMGenerationError(RuntimeError):
    """Raised when a provider fails during generation."""


@runtime_checkable
class LLMProvider(Protocol):
    """Provider abstraction for one selected LLM model."""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        ...

    def health_check(self) -> bool:
        ...

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        ...
