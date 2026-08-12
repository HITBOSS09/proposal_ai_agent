"""Pure enrichment functions for converting Phase 1 chunks into frozen payloads."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Iterable, Optional, Union
from uuid import UUID

from ..chunk_models import Chunk
from .schemas import (
    ChunkMetadataPayload,
    DocumentMetadataPayload,
    EnrichedChunkPayload,
    LocationMetadataPayload,
    SectionMetadataPayload,
)

METADATA_PIPELINE_VERSION = "2.1"


def normalize_text(text: str) -> str:
    """Trim and collapse whitespace without changing the order of text content."""
    return " ".join(text.split())


def compute_content_hash(text: str) -> str:
    """Return the SHA-256 hash of normalized text."""
    return sha256(normalize_text(text).encode("utf-8")).hexdigest()


def compute_chunk_id(
    document_id: Union[UUID, str],
    section_id: Optional[Union[UUID, str]],
    chunk_index: int,
    normalized_text: str,
) -> str:
    """Return the deterministic SHA-256 identifier for an enriched chunk.

    Text is normalized here so direct callers and ``enrich_chunk`` share the
    same identifier contract.
    """
    section_value = "" if section_id is None else str(section_id)
    identifier_input = (
        f"{document_id}{section_value}{chunk_index}{normalize_text(normalized_text)}"
    )
    return sha256(identifier_input.encode("utf-8")).hexdigest()


def build_embedding_text(chunk: Chunk) -> str:
    """Add concise structural context without modifying canonical chunk text."""
    context: list[str] = []
    if chunk.document_title and chunk.document_title.strip():
        context.append(f"Document: {chunk.document_title.strip()}")
    section_path = tuple(item.strip() for item in chunk.section_path if item and item.strip())
    if section_path:
        context.append(f"Section Path: {' > '.join(section_path)}")
    if chunk.element_type and chunk.element_type.strip():
        context.append(f"Element Type: {chunk.element_type.strip()}")

    body = chunk.text
    rendered_path = "\n".join(chunk.section_path)
    if rendered_path and body.startswith(f"{rendered_path}\n\n"):
        body = body[len(rendered_path) + 2 :]
    context_text = "\n".join(context)
    return f"{context_text}\n\n{body}" if context else body


def enrich_chunk(
    chunk: Chunk,
    processing_timestamp: datetime,
    pipeline_version: str = METADATA_PIPELINE_VERSION,
    document_role: str | None = None,
) -> EnrichedChunkPayload:
    """Create an immutable, serializable payload from one Phase 1 chunk.

    The caller provides the timestamp, keeping enrichment deterministic except
    for that explicit provenance value. The timestamp is never used in a hash.
    """
    normalized_text = normalize_text(chunk.text)
    return EnrichedChunkPayload(
        schema_version=chunk.schema_version,
        pipeline_version=pipeline_version,
        processing_timestamp=processing_timestamp,
        content_hash=compute_content_hash(chunk.text),
        chunk_id=compute_chunk_id(
            chunk.document_id,
            chunk.section_id,
            chunk.chunk_index,
            normalized_text,
        ),
        point_uuid=str(chunk.chunk_id),
        document_role=document_role,
        document=DocumentMetadataPayload(
            document_id=str(chunk.document_id),
            document_type=chunk.document_type,
            language=chunk.language,
            source_file=chunk.source_file,
            source_path=chunk.source_path,
            document_title=chunk.document_title,
        ),
        section=SectionMetadataPayload(
            section_id=str(chunk.section_id) if chunk.section_id is not None else None,
            section_path=tuple(chunk.section_path),
            heading=chunk.heading,
            parent_section_id=str(chunk.parent_section_id) if chunk.parent_section_id is not None else None,
        ),
        chunk=ChunkMetadataPayload(
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
            total_chunks=chunk.total_chunks,
            is_first_chunk=chunk.is_first_chunk,
            is_last_chunk=chunk.is_last_chunk,
            section_chunk_index=chunk.section_chunk_index,
            total_chunks_in_section=chunk.total_chunks_in_section,
            parser_version=chunk.parser_version,
            element_type=chunk.element_type,
        ),
        location=LocationMetadataPayload(
            order_start=chunk.order_start,
            order_end=chunk.order_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        ),
        original_text=chunk.text,
        embedding_text=build_embedding_text(chunk),
    )


def enrich_chunks(
    chunks: Iterable[Chunk],
    processing_timestamp: datetime,
    pipeline_version: str = METADATA_PIPELINE_VERSION,
    document_role: str | None = None,
) -> tuple[EnrichedChunkPayload, ...]:
    """Enrich chunks in input order without mutating the input collection."""
    return tuple(
        enrich_chunk(chunk, processing_timestamp, pipeline_version, document_role)
        for chunk in chunks
    )


__all__ = [
    "METADATA_PIPELINE_VERSION",
    "compute_chunk_id",
    "compute_content_hash",
    "build_embedding_text",
    "enrich_chunk",
    "enrich_chunks",
    "normalize_text",
]
