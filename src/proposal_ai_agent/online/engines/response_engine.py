"""LLM generation engine for the online pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.response import GeneratedResponse
from ..contracts.synthesis import PromptPackage
from ..providers.llm.router import LLMRouter


@dataclass(frozen=True, slots=True)
class ResponseEngine:
    """Delegate one prompt package to a single selected LLM provider."""

    router: LLMRouter

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        if not isinstance(prompt_package, PromptPackage):
            raise TypeError("prompt_package must be a PromptPackage")
        return self.router.generate(prompt_package)
