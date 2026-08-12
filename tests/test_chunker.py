"""
Tests for the structure-aware chunking engine.

Covers:
- section-aware chunk creation
- table atomicity
- chunk overlap semantics
- heading and order preservation
"""

from uuid import uuid4

from proposal_ai_agent.ingestion.chunker import ChunkConfig, chunk_document
from proposal_ai_agent.ingestion.models import (
    Document,
    DocumentMetadata,
    ElementMetadata,
    Section,
    Paragraph,
    Table,
    TableRow,
    TableCell,
)


def make_element_metadata(doc_id, section_id, order_index):
    return ElementMetadata(
        id=uuid4(),
        document_id=doc_id,
        section_id=section_id,
        order_index=order_index,
        source_file="test.docx",
    )


class TestChunkingEngine:
    def test_chunk_document_preserves_section_path_and_order(self):
        doc_id = uuid4()
        section = Section(heading="Introduction", section_level=1)
        para1 = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 0),
            content="First paragraph",
        )
        para2 = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 1),
            content="Second paragraph",
        )
        section.elements = [para1, para2]

        document = Document(
            title="Chunk Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section],
        )

        chunks = chunk_document(document, ChunkConfig(chunk_size=50, overlap_ratio=0.0))

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.section_path == ["Introduction"]
        assert chunk.heading == "Introduction"
        assert chunk.order_start == 0
        assert chunk.order_end == 1
        assert "Introduction" in chunk.text
        assert "First paragraph" in chunk.text
        assert "Second paragraph" in chunk.text

    def test_table_is_always_a_standalone_chunk(self):
        doc_id = uuid4()
        section = Section(heading="Data", section_level=1)

        para = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 0),
            content="Before the table",
        )
        table = Table(
            metadata=make_element_metadata(doc_id, section.id, 1),
            rows=[
                TableRow(cells=[TableCell(content="A"), TableCell(content="B")], row_index=0),
                TableRow(cells=[TableCell(content="C"), TableCell(content="D")], row_index=1),
            ],
            caption="Sample table",
        )
        after_para = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 2),
            content="After the table",
        )
        section.elements = [para, table, after_para]

        document = Document(
            title="Chunk Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section],
        )

        chunks = chunk_document(document, ChunkConfig(chunk_size=100, overlap_ratio=0.0))

        assert len(chunks) == 3
        assert chunks[0].order_start == 0
        assert chunks[0].order_end == 0
        assert "Before the table" in chunks[0].text

        assert chunks[1].order_start == 1
        assert chunks[1].order_end == 1
        assert "A | B" in chunks[1].text
        assert "C | D" in chunks[1].text

        assert chunks[2].order_start == 2
        assert chunks[2].order_end == 2
        assert "After the table" in chunks[2].text

    def test_chunk_overlap_repeats_last_item_into_next_chunk(self):
        doc_id = uuid4()
        section = Section(heading="Overlap", section_level=1)

        para1 = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 0),
            content="Alpha Beta",
        )
        para2 = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 1),
            content="Gamma Delta",
        )
        para3 = Paragraph(
            metadata=make_element_metadata(doc_id, section.id, 2),
            content="Epsilon Zeta",
        )
        section.elements = [para1, para2, para3]

        document = Document(
            title="Chunk Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section],
        )

        chunks = chunk_document(document, ChunkConfig(chunk_size=5, overlap_ratio=0.2))

        assert len(chunks) == 2
        assert chunks[0].order_start == 0
        assert chunks[0].order_end == 1
        assert "Alpha Beta" in chunks[0].text
        assert "Gamma Delta" in chunks[0].text

        assert chunks[1].order_start == 1
        assert chunks[1].order_end == 2
        assert "Gamma Delta" in chunks[1].text
        assert "Epsilon Zeta" in chunks[1].text
