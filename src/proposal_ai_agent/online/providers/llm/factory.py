"""Factory for creating LLM provider instances by name."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .provider import LLMProvider
from .ollama import OllamaProvider
from .stub import StubLLMProvider


_DEFAULT_PROVIDERS: dict[str, Callable[..., LLMProvider]] = {
    "stub": StubLLMProvider,
    "ollama": OllamaProvider,
}


class LLMProviderFactory:
    """Resolve provider implementations by stable name."""

    def __init__(self, provider_registry: Mapping[str, Callable[..., LLMProvider]] | None = None) -> None:
        registry = dict(_DEFAULT_PROVIDERS)
        registry.update(provider_registry or {})
        self._provider_registry = registry

    def register_provider(self, provider_name: str, provider_constructor: Callable[..., LLMProvider]) -> None:
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        if provider_name in self._provider_registry:
            raise ValueError(f"provider already registered: {provider_name}")
        self._provider_registry[provider_name] = provider_constructor

    def create(self, name: str, **kwargs: Any) -> LLMProvider:
        if name not in self._provider_registry:
            raise ValueError(f"provider not registered: {name}")
        provider_constructor = self._provider_registry[name]
        provider = provider_constructor(**kwargs)
        if not isinstance(provider, LLMProvider):
            raise TypeError("created provider must implement LLMProvider")
        return provider

    def supported_providers(self) -> tuple[str, ...]:
        return tuple(self._provider_registry)
