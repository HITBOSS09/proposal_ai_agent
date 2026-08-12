"""Tests for the immutable metadata enrichment engine."""

from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

import pytest
from pydantic import ValidationError

from proposal_ai_agent.ingestion.chunk_models import Chunk
from proposal_ai_agent.ingestion.metadata import (
    EnrichedChunkPayload,
    compute_chunk_id,
    compute_content_hash,
    enrich_chunk,
    enrich_chunks,
    normalize_text,
)


def make_chunk() -> Chunk:
    return Chunk(
        document_id=uuid4(),
        section_id=uuid4(),
        section_path=["Introduction", "Scope"],
        heading="Scope",
        document_type="proposal",
        language="en",
        source_file="proposal.docx",
        source_path="/documents/proposal.docx",
        chunk_index=3,
        order_start=10,
        order_end=12,
        text="  Original\n\nchunk\ttext.  ",
        token_count=3,
        total_chunks=8,
        section_chunk_index=1,
        total_chunks_in_section=2,
        char_start=50,
        char_end=72,
        schema_version="1.0",
        parser_version="1.0",
    )


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  A\n\nB\t C   ") == "A B C"


def test_content_hash_uses_normalized_text() -> None:
    expected = sha256(b"A B C").hexdigest()
    assert compute_content_hash(" A\nB\t C ") == expected


def test_chunk_id_is_deterministic() -> None:
    chunk = make_chunk()
    normalized_text = normalize_text(chunk.text)
    assert compute_chunk_id(chunk.document_id, chunk.section_id, 3, normalized_text) == compute_chunk_id(
        chunk.document_id, chunk.section_id, 3, normalized_text
    )
    assert compute_chunk_id(chunk.document_id, chunk.section_id, 3, " Original\nchunk text. ") == compute_chunk_id(
        chunk.document_id, chunk.section_id, 3, normalized_text
    )


def test_timestamp_does_not_affect_hashes_or_identifier() -> None:
    chunk = make_chunk()
    first = enrich_chunk(chunk, datetime(2026, 1, 1, tzinfo=timezone.utc))
    second = enrich_chunk(chunk, datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert first.processing_timestamp != second.processing_timestamp
    assert first.content_hash == second.content_hash
    assert first.chunk_id == second.chunk_id


def test_enrich_chunk_attaches_chunk_metadata_and_is_immutable() -> None:
    chunk = make_chunk()
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payload = enrich_chunk(chunk, timestamp)

    assert payload.original_text == chunk.text
    assert payload.point_uuid == str(chunk.chunk_id)
    assert payload.document.document_id == str(chunk.document_id)
    assert payload.section.section_path == ("Introduction", "Scope")
    assert payload.chunk.chunk_index == 3
    assert payload.location.order_start == 10
    assert payload.processing_timestamp == timestamp
    with pytest.raises(ValidationError):
        payload.original_text = "mutated"  # type: ignore[misc]


def test_enrich_chunks_preserves_input_order() -> None:
    first = make_chunk()
    second = make_chunk()
    second.chunk_index = 4
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payloads = enrich_chunks([first, second], timestamp)

    assert isinstance(payloads, tuple)
    assert [payload.chunk.chunk_index for payload in payloads] == [3, 4]


def test_payload_serializes_to_json_and_validates() -> None:
    payload = enrich_chunk(make_chunk(), datetime(2026, 1, 1, tzinfo=timezone.utc))
    serialized = payload.model_dump_json()
    restored = EnrichedChunkPayload.model_validate_json(serialized)

    assert restored == payload
    with pytest.raises(ValidationError):
        EnrichedChunkPayload.model_validate({"schema_version": "1.0"})
