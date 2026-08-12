"""Explicit collection configuration consumed by the Qdrant writer."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field


class IndexingConfig(BaseModel):
    """Collection settings supplied by application configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_name: str = Field(min_length=1)
    vector_dimensions: int = Field(gt=0)
    distance: Literal["cosine", "euclid", "dot"]
    payload_indexes: Tuple[str, ...]
