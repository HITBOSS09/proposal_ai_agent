"""Metadata enrichment for immutable Phase 1 chunk payloads."""

from .enricher import (
    METADATA_PIPELINE_VERSION,
    compute_chunk_id,
    compute_content_hash,
    build_embedding_text,
    enrich_chunk,
    enrich_chunks,
    normalize_text,
)
from .schemas import EnrichedChunkPayload

__all__ = [
    "EnrichedChunkPayload",
    "METADATA_PIPELINE_VERSION",
    "compute_chunk_id",
    "compute_content_hash",
    "build_embedding_text",
    "enrich_chunk",
    "enrich_chunks",
    "normalize_text",
]
