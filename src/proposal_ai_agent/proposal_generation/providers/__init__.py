"""Proposal-domain LLM provider adapters and their public port."""

from typing import TYPE_CHECKING

from .provider import ProposalGenerationError, ProposalLLMProvider, ProposalProviderUnavailableError

if TYPE_CHECKING:
    from .factory import ProposalLLMFactory
    from .ollama import ProposalOllamaAdapter

__all__ = [
    "ProposalGenerationError",
    "ProposalLLMFactory",
    "ProposalLLMProvider",
    "ProposalOllamaAdapter",
    "ProposalProviderUnavailableError",
]


def __getattr__(name: str):
    """Load concrete adapters lazily to keep the provider port acyclic."""
    if name == "ProposalLLMFactory":
        from .factory import ProposalLLMFactory

        return ProposalLLMFactory
    if name == "ProposalOllamaAdapter":
        from .ollama import ProposalOllamaAdapter

        return ProposalOllamaAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
