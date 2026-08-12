"""Immutable proposal-domain results of deterministic retrieval execution."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .prompt_composer import RetrievedReference
from .retrieval_query import SectionRetrievalQuery


class RetrievedContext(BaseModel):
    """Grouped typed references retrieved for one proposal section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_id: str = Field(min_length=1)
    queries: tuple[SectionRetrievalQuery, ...]
    style_references: tuple[RetrievedReference, ...] = ()
    technical_references: tuple[RetrievedReference, ...] = ()
    blueprint_references: tuple[RetrievedReference, ...] = ()
    metadata: Mapping[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(
        cls, value: Mapping[str, str | int | float | bool | None]
    ) -> Mapping[str, str | int | float | bool | None]:
        """Prevent mutation through the context's metadata mapping."""
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def validate_groups(self) -> "RetrievedContext":
        if not self.queries:
            raise ValueError("queries must not be empty")
        if any(query.section_id != self.section_id for query in self.queries):
            raise ValueError("queries must belong to the context section_id")
        expected_types = {
            "authoring": self.style_references,
            "technical": self.technical_references,
            "blueprint": self.blueprint_references,
        }
        for reference_type, references in expected_types.items():
            if any(reference.reference_type != reference_type for reference in references):
                raise ValueError(f"{reference_type} reference group contains an incompatible reference")
        return self
