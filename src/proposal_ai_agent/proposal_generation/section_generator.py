"""Provider-agnostic generation of one proposal section."""

from __future__ import annotations

from .prompt_composer import SectionPromptPackage
from .providers.provider import ProposalLLMProvider
from .transport_contract import ProposalResponse


class SectionGenerator:
    """Generate one section through the proposal-domain provider port."""

    def __init__(self, provider: ProposalLLMProvider) -> None:
        if not isinstance(provider, ProposalLLMProvider):
            raise TypeError("provider must implement ProposalLLMProvider")
        self._provider = provider

    def generate(self, prompt: SectionPromptPackage) -> ProposalResponse:
        """Delegate one complete proposal prompt to the configured provider."""
        if not isinstance(prompt, SectionPromptPackage):
            raise TypeError("prompt must be a SectionPromptPackage")
        response = self._provider.generate(prompt)
        if not isinstance(response, ProposalResponse):
            raise TypeError("provider.generate must return a ProposalResponse")
        return response
