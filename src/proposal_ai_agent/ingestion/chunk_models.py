"""
Chunk model definitions for the structure-aware chunking engine.

These models define chunk configuration and the chunk data structure used by the
chunking engine to preserve document structure, section context, and reading order.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass(slots=True)
class ChunkConfig:
    """Configures how chunks are created from document sections."""
    chunk_size: int = 500
    overlap_ratio: float = 0.1


@dataclass(slots=True)
class Chunk:
    """Represents a single chunk of document content."""
    # Core identifiers (required first)
    document_id: UUID
    section_id: Optional[UUID]
    section_path: List[str]
    heading: str

    # Document-level metadata
    document_type: Optional[str]
    language: str
    source_file: str
    source_path: str

    # Global and ordering metadata
    chunk_index: int
    order_start: int
    order_end: int
    text: str
    token_count: int

    # Deterministic structural context used independently from raw chunk text.
    document_title: Optional[str] = None
    parent_section_id: Optional[UUID] = None
    element_type: str = "mixed"

    # Optional fields with defaults
    chunk_id: UUID = field(default_factory=uuid4)
    total_chunks: int = 0
    is_first_chunk: bool = False
    is_last_chunk: bool = False

    # Section sequence metadata
    section_chunk_index: int = 0
    total_chunks_in_section: int = 0

    # Offsets into normalized section text
    char_start: int = 0
    char_end: int = 0

    # Versioning
    schema_version: str = "1.0"
    parser_version: Optional[str] = "1.0"

    # Optional created timestamp (not set by chunker by default)
    created_at: Optional[str] = None
