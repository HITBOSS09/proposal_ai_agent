"""Provider-neutral contracts for bounded neural candidate reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ...contracts.retrieval import ProcessedCandidate


@dataclass(frozen=True, slots=True)
class RerankingScore:
    """One relevance score referring to a candidate's input position."""

    candidate_index: int
    relevance_score: float

    def __post_init__(self) -> None:
        """Reject invalid provider output before it reaches the engine."""
        if isinstance(self.candidate_index, bool) or not isinstance(self.candidate_index, int):
            raise TypeError("candidate_index must be an integer")
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        if isinstance(self.relevance_score, bool) or not isinstance(
            self.relevance_score, (int, float)
        ):
            raise TypeError("relevance_score must be numeric")
        if not isfinite(self.relevance_score):
            raise ValueError("relevance_score must be finite")
        object.__setattr__(self, "relevance_score", float(self.relevance_score))


@runtime_checkable
class RerankingProvider(Protocol):
    """Swappable neural scorer that never retrieves or alters candidates."""

    def rerank(
        self, query: str, candidates: Sequence[ProcessedCandidate]
    ) -> Sequence[RerankingScore]:
        """Return one relevance score for every supplied candidate position."""
