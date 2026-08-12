"""Immutable schemas for metadata-enriched chunk payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict


class _FrozenPayload(BaseModel):
    """Base schema that prevents mutation and rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentMetadataPayload(_FrozenPayload):
    """Document-level metadata available on a Phase 1 chunk."""

    document_id: str
    document_type: Optional[str]
    language: str
    source_file: str
    source_path: str
    document_title: Optional[str] = None


class SectionMetadataPayload(_FrozenPayload):
    """Section context retained by the structure-aware chunker."""

    section_id: Optional[str]
    section_path: Tuple[str, ...]
    heading: str
    parent_section_id: Optional[str] = None


class ChunkMetadataPayload(_FrozenPayload):
    """Chunk sequencing and upstream-version metadata."""

    chunk_index: int
    token_count: int
    total_chunks: int
    is_first_chunk: bool
    is_last_chunk: bool
    section_chunk_index: int
    total_chunks_in_section: int
    parser_version: Optional[str]
    element_type: str


class LocationMetadataPayload(_FrozenPayload):
    """Source-order and character-offset metadata for a chunk."""

    order_start: int
    order_end: int
    char_start: int
    char_end: int


class EnrichedChunkPayload(_FrozenPayload):
    """Stable, serializable metadata envelope for a Phase 1 ``Chunk``."""

    schema_version: str
    pipeline_version: str
    processing_timestamp: datetime
    content_hash: str
    chunk_id: str
    point_uuid: str
    document_role: Optional[str] = None
    document: DocumentMetadataPayload
    section: SectionMetadataPayload
    chunk: ChunkMetadataPayload
    location: LocationMetadataPayload
    original_text: str
    embedding_text: str
