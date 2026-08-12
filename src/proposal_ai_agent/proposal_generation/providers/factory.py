"""Factory for proposal-domain providers returning structured transport DTOs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .ollama import ProposalOllamaAdapter
from .provider import ProposalLLMProvider


_DEFAULT_PROVIDERS: dict[str, Callable[..., ProposalLLMProvider]] = {
    "ollama": ProposalOllamaAdapter,
}


class ProposalLLMFactory:
    """Resolve providers without exposing provider details or Proposal IR to callers."""

    def __init__(self, provider_registry: Mapping[str, Callable[..., ProposalLLMProvider]] | None = None) -> None:
        registry = dict(_DEFAULT_PROVIDERS)
        registry.update(provider_registry or {})
        self._provider_registry = registry

    def register_provider(self, name: str, constructor: Callable[..., ProposalLLMProvider]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name must be a non-empty string")
        if name in self._provider_registry:
            raise ValueError(f"provider already registered: {name}")
        self._provider_registry[name] = constructor

    def create(self, name: str, **configuration: Any) -> ProposalLLMProvider:
        if name not in self._provider_registry:
            raise ValueError(f"proposal provider not registered: {name}")
        provider = self._provider_registry[name](**configuration)
        if not isinstance(provider, ProposalLLMProvider):
            raise TypeError("created provider must implement ProposalLLMProvider")
        return provider

    def supported_providers(self) -> tuple[str, ...]:
        return tuple(self._provider_registry)
