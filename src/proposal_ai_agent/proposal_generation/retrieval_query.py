"""Provider-neutral retrieval-query contracts for proposal sections."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReferenceType(str, Enum):
    """Reference categories used by proposal-section retrieval policies."""

    AUTHORING = "authoring"
    TECHNICAL = "technical"
    BLUEPRINT = "blueprint"


class RetrievalStrategy(str, Enum):
    """Provider-neutral retrieval strategy requested by a section query."""

    DENSE = "dense"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class SectionRetrievalQuery(BaseModel):
    """Immutable, deterministic retrieval instruction for one section reference type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_id: str = Field(min_length=1)
    reference_type: ReferenceType
    query_text: str = Field(min_length=1)
    metadata_filters: Mapping[str, str | int | float | bool | None] = Field(default_factory=dict)
    max_results: int = Field(gt=0)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    rerank_enabled: bool = False

    @field_validator("metadata_filters")
    @classmethod
    def freeze_metadata_filters(
        cls, value: Mapping[str, str | int | float | bool | None]
    ) -> Mapping[str, str | int | float | bool | None]:
        return MappingProxyType(dict(value))
