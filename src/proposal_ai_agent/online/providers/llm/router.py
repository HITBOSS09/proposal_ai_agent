"""Deterministic provider routing for one selected LLM per request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...contracts.response import GeneratedResponse
from ...contracts.synthesis import PromptPackage
from .provider import LLMProvider, ProviderUnavailableError, LLMGenerationError


class NoAvailableProviderError(RuntimeError):
    """Raised when no provider is healthy or suitable for the request."""


@dataclass(frozen=True, slots=True)
class LLMRouter:
    """Select one provider for a prompt package using deterministic policy."""

    providers: tuple[LLMProvider, ...]

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("LLMRouter requires at least one provider")
        if any(not isinstance(provider, LLMProvider) for provider in self.providers):
            raise TypeError("providers must implement LLMProvider")

    def generate(self, prompt_package: PromptPackage) -> GeneratedResponse:
        provider = self.route(prompt_package)
        try:
            return provider.generate(prompt_package)
        except (ProviderUnavailableError, LLMGenerationError, TimeoutError) as error:
            fallback_providers = [
                p
                for p in self._ordered_providers(prompt_package)
                if p is not provider and self.provider_suits_security(prompt_package, p)
            ]
            for fallback in fallback_providers:
                if not fallback.health_check():
                    continue
                try:
                    return fallback.generate(prompt_package)
                except (ProviderUnavailableError, LLMGenerationError, TimeoutError):
                    continue
            raise NoAvailableProviderError(
                f"no available provider could generate a response: {error}"
            ) from error

    def route(self, prompt_package: PromptPackage) -> LLMProvider:
        if not isinstance(prompt_package, PromptPackage):
            raise TypeError("prompt_package must be a PromptPackage")

        ordered = self._ordered_providers(prompt_package)
        for provider in ordered:
            if provider.health_check() and self.provider_suits_security(prompt_package, provider):
                return provider
        raise NoAvailableProviderError("no healthy providers are available")

    def _ordered_providers(self, prompt_package: PromptPackage) -> tuple[LLMProvider, ...]:
        preferred = self._preferred_provider_name(prompt_package)
        providers = tuple(self.providers)
        return tuple(sorted(providers, key=lambda provider: self._routing_key(provider, preferred)))

    @staticmethod
    def _preferred_provider_name(prompt_package: PromptPackage) -> str | None:
        metadata = prompt_package.generation_metadata
        preferred = metadata.get("preferred_provider")
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
        return None

    @staticmethod
    def _routing_key(provider: LLMProvider, preferred_provider: str | None) -> tuple[int, int, int, str, str]:
        metadata = provider.provider_metadata
        priority = int(metadata.get("priority", 100))
        cost_tier = LLMRouter._tier_value(metadata.get("cost_tier"))
        latency_tier = LLMRouter._tier_value(metadata.get("latency_tier"))
        preferred_order = 0 if preferred_provider and provider.provider_name == preferred_provider else 1
        return (preferred_order, priority, cost_tier, latency_tier, provider.provider_name or provider.model_name)

    @staticmethod
    def _tier_value(value: Any) -> int:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "low":
                return 0
            if normalized == "medium":
                return 1
            if normalized == "high":
                return 2
        if isinstance(value, int):
            return value
        return 1

    @staticmethod
    def provider_suits_security(prompt_package: PromptPackage, provider: LLMProvider) -> bool:
        requested = prompt_package.generation_metadata.get("security_policy")
        provider_policy = provider.provider_metadata.get("security_policy")
        if requested is None or provider_policy is None:
            return True
        return requested == provider_policy
