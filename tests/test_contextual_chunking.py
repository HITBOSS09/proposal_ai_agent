"""Phase 2 structural-context preservation and embedding separation tests."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from proposal_ai_agent.embeddings import EmbeddingEngine, EmbeddingEngineConfig, MemoryCache
from proposal_ai_agent.embeddings.providers.base import EmbeddingProvider
from proposal_ai_agent.indexing import IndexBuilder
from proposal_ai_agent.ingestion.chunk_models import Chunk
from proposal_ai_agent.ingestion.chunker import ChunkConfig, chunk_document
from proposal_ai_agent.ingestion.metadata import enrich_chunk
from proposal_ai_agent.ingestion.models import (
    Document, DocumentMetadata, ElementMetadata, Paragraph, Section, Table, TableCell, TableRow,
)


def _metadata(document_id, section_id, order):
    return ElementMetadata(
        document_id=document_id,
        section_id=section_id,
        order_index=order,
        source_file="arbitrary.docx",
        source_path="/documents/arbitrary.docx",
    )


def _nested_document() -> Document:
    document_id = uuid4()
    parent = Section(heading="Parent Area", section_level=1)
    child = Section(heading="Component Inventory", section_level=2)
    child.elements = [
        Paragraph(metadata=_metadata(document_id, child.id, 0), content="Inventory introduction."),
        Table(
            metadata=_metadata(document_id, child.id, 1),
            rows=(
                TableRow(cells=(TableCell(content="Item"), TableCell(content="Quantity"))),
                TableRow(cells=(TableCell(content="Controller"), TableCell(content="2"))),
            ),
        ),
    ]
    parent.subsections = [child]
    return Document(
        title="Arbitrary Technical Document",
        metadata=DocumentMetadata(source="/documents/arbitrary.docx", document_type="technical"),
        document_id=document_id,
        sections=[parent],
    )


def test_nested_table_is_atomic_and_inherits_authoritative_parent_path() -> None:
    chunks = chunk_document(_nested_document(), ChunkConfig(chunk_size=100, overlap_ratio=0.0))

    assert len(chunks) == 2
    paragraph, table = chunks
    assert paragraph.element_type == "paragraph"
    assert table.element_type == "table"
    assert table.order_start == table.order_end == 1
    assert table.section_path == ["Parent Area", "Component Inventory"]
    assert table.heading == "Component Inventory"
    assert table.parent_section_id is not None
    assert "Item | Quantity" in table.text
    assert "Inventory introduction" not in table.text


def test_contextual_embedding_is_concise_and_raw_chunk_remains_unchanged() -> None:
    table = chunk_document(_nested_document(), ChunkConfig(chunk_size=100, overlap_ratio=0.0))[1]
    raw_before = table.text

    payload = enrich_chunk(table, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert payload.original_text == raw_before == table.text
    assert payload.embedding_text.startswith(
        "Document: Arbitrary Technical Document\n"
        "Section Path: Parent Area > Component Inventory\n"
        "Element Type: table\n\n"
    )
    assert payload.embedding_text.count("Component Inventory") == 1
    assert "None" not in payload.embedding_text
    assert payload.section.parent_section_id == str(table.parent_section_id)
    assert payload.chunk.element_type == "table"


def test_context_changes_embedding_only_and_preserves_raw_identity_semantics() -> None:
    chunk = Chunk(
        document_id=uuid4(), section_id=uuid4(), section_path=["Inventory"], heading="Inventory",
        document_type="technical", language="en", source_file="source.docx", source_path="/source.docx",
        chunk_index=1, order_start=2, order_end=2, text="Inventory\n\nItem | Quantity\nUnit | 1",
        token_count=6, element_type="table", document_title="First Document",
    )
    first = enrich_chunk(chunk, datetime(2026, 1, 1, tzinfo=timezone.utc))
    chunk.document_title = "Second Document"
    second = enrich_chunk(chunk, datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert first.original_text == second.original_text
    assert first.content_hash == second.content_hash
    assert first.chunk_id == second.chunk_id
    assert first.point_uuid == second.point_uuid
    assert first.embedding_text != second.embedding_text


class _CapturingProvider(EmbeddingProvider):
    def __init__(self):
        self.texts: tuple[str, ...] = ()

    def embed(self, text: str):
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]):
        self.texts = tuple(texts)
        return tuple((1.0, 2.0, 3.0) for _ in texts)


def test_embedding_uses_context_but_index_payload_keeps_original_evidence() -> None:
    table = chunk_document(_nested_document(), ChunkConfig(chunk_size=100, overlap_ratio=0.0))[1]
    payload = enrich_chunk(table, datetime(2026, 1, 1, tzinfo=timezone.utc))
    provider = _CapturingProvider()

    result = EmbeddingEngine(
        EmbeddingEngineConfig(provider=provider, dimensions=3), MemoryCache()
    ).embed(payload)
    point = IndexBuilder(3).build(payload, result.vector)

    assert provider.texts == (payload.embedding_text,)
    assert point.payload["original_text"] == table.text
    assert point.payload["embedding_text"] == payload.embedding_text
    assert point.payload["document"]["document_title"] == "Arbitrary Technical Document"
    assert point.payload["section_path"] == ["Parent Area", "Component Inventory"]
    assert point.payload["section_title"] == "Component Inventory"
    assert point.payload["parent_section_id"] == str(table.parent_section_id)
    assert point.payload["element_type"] == "table"


def test_contextual_chunking_has_no_source_specific_names() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "src/proposal_ai_agent/ingestion/chunker.py",
            "src/proposal_ai_agent/ingestion/metadata/enricher.py",
            "src/proposal_ai_agent/embeddings/engine.py",
        )
    )
    assert "PRU_T72" not in sources
    assert "Bill of Materials" not in sources
    assert "Module:" not in sources
