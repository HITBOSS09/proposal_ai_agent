"""Provider port for proposal-section generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..transport_contract import ProposalResponse

if TYPE_CHECKING:
    from ..prompt_composer import SectionPromptPackage


class ProposalProviderUnavailableError(RuntimeError):
    """Raised when a proposal-generation provider cannot be reached."""


class ProposalGenerationError(RuntimeError):
    """Raised when a proposal-generation provider returns an invalid result."""


@runtime_checkable
class ProposalLLMProvider(Protocol):
    """Provider-neutral port for generating one proposal section."""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        ...

    def generate(self, prompt: SectionPromptPackage) -> ProposalResponse:
        ...
