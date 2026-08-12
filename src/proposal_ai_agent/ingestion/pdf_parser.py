"""Native PDF parser that converts text-based PDFs into internal document models."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from uuid import UUID, uuid4

import fitz

from .models import (
    Document,
    DocumentMetadata,
    ElementMetadata,
    Paragraph,
    Section,
    Table,
    TableCell,
    TableRow,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
PARSER_VERSION = "1.0"


class PDFParserError(Exception):
    """Raised when a PDF cannot be converted into the internal document model."""


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """Structured text extracted from a PDF page."""

    bbox: Tuple[float, float, float, float]
    text: str
    max_font_size: float
    is_bold: bool


@dataclass(frozen=True, slots=True)
class _TableBlock:
    """Structured table extracted from a PDF page."""

    bbox: Tuple[float, float, float, float]
    rows: List[List[str]]


_PageBlock = Union[_TextBlock, _TableBlock]
_SECTION_PATTERN = re.compile(r"^(section|chapter|appendix|form)\b", re.IGNORECASE)


def _normalise_text(text: str) -> str:
    """Trim PDF extraction noise while retaining paragraph line breaks."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _is_bold_font(font_name: str) -> bool:
    """Recognise common bold font-name markers emitted by PDF producers."""
    font_name = font_name.lower()
    return "bold" in font_name or "black" in font_name or "demi" in font_name


def _extract_text_blocks(page: fitz.Page) -> List[_TextBlock]:
    """Extract sorted text blocks with the formatting needed for heading heuristics."""
    text_blocks: List[_TextBlock] = []
    page_data = page.get_text("dict", sort=True)

    for block in page_data["blocks"]:
        if block["type"] != 0:
            continue

        lines = block.get("lines", [])
        spans = [span for line in lines for span in line.get("spans", [])]
        text = _normalise_text("".join(span["text"] for span in spans))
        if not text:
            continue

        text_blocks.append(
            _TextBlock(
                bbox=tuple(block["bbox"]),
                text=text,
                max_font_size=max((span["size"] for span in spans), default=0.0),
                is_bold=any(_is_bold_font(span["font"]) for span in spans),
            )
        )

    return text_blocks


def _extract_table_blocks(page: fitz.Page) -> List[_TableBlock]:
    """Extract tables recognised by PyMuPDF, preserving row and column positions."""
    tables: List[_TableBlock] = []
    try:
        table_finder = page.find_tables()
    except Exception as error:
        logger.warning("Table detection failed on PDF page %d: %s", page.number + 1, error)
        return tables

    for detected_table in table_finder.tables:
        rows = [
            [cell.strip() if cell else "" for cell in row]
            for row in detected_table.extract()
        ]
        if rows and any(any(cell for cell in row) for row in rows):
            tables.append(_TableBlock(bbox=tuple(detected_table.bbox), rows=rows))

    return tables


def _intersects_table(text_block: _TextBlock, table_blocks: Sequence[_TableBlock]) -> bool:
    """Avoid duplicating table cell text as paragraph content."""
    text_rect = fitz.Rect(text_block.bbox)
    return any(text_rect.intersects(fitz.Rect(table_block.bbox)) for table_block in table_blocks)


def _page_blocks(page: fitz.Page) -> List[_PageBlock]:
    """Return paragraph and table blocks in page reading order."""
    table_blocks = _extract_table_blocks(page)
    text_blocks = [
        block for block in _extract_text_blocks(page)
        if not _intersects_table(block, table_blocks)
    ]
    return sorted(
        [*text_blocks, *table_blocks],
        key=lambda block: (block.bbox[1], block.bbox[0]),
    )


def _body_font_size(blocks: Iterable[_PageBlock]) -> float:
    """Calculate a robust body-font baseline for heading detection."""
    font_sizes = [
        block.max_font_size
        for block in blocks
        if isinstance(block, _TextBlock) and block.max_font_size > 0
    ]
    return median(font_sizes) if font_sizes else 0.0


def _is_uppercase_heading(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and len(text) <= 160 and sum(character.isupper() for character in letters) / len(letters) >= 0.8


def _heading_level(block: _TextBlock, body_font_size: float) -> Optional[int]:
    """Infer a section level from PDF typography and common proposal heading forms."""
    single_line = " ".join(block.text.splitlines())
    if len(single_line) > 160:
        return None

    if _SECTION_PATTERN.match(single_line) or _is_uppercase_heading(single_line):
        return 1

    if body_font_size and block.max_font_size >= body_font_size * 1.45:
        return 1
    if body_font_size and block.is_bold and block.max_font_size >= body_font_size * 1.15:
        return 2
    return None


class PDFParser:
    """Parse text-based PDFs into the common internal document model."""

    def __init__(self) -> None:
        self.order_index = 0
        self.sections_stack: List[Section] = []

    def _push_section(
        self,
        heading: str,
        heading_level: int,
        top_level_sections: List[Section],
    ) -> Section:
        while self.sections_stack and self.sections_stack[-1].section_level >= heading_level:
            self.sections_stack.pop()

        section = Section(heading=heading, section_level=heading_level)
        if self.sections_stack:
            self.sections_stack[-1].subsections.append(section)
        else:
            top_level_sections.append(section)
        self.sections_stack.append(section)
        return section

    def _add_element(
        self,
        element: Union[Paragraph, Table],
        root_elements: List[Union[Paragraph, Table]],
    ) -> None:
        if self.sections_stack:
            element.metadata.section_id = self.sections_stack[-1].id
            self.sections_stack[-1].elements.append(element)
        else:
            root_elements.append(element)

    def _element_metadata(
        self,
        document_id: UUID,
        page_number: int,
        source_file: str,
        source_path: str,
        source_document: Optional[str],
    ) -> ElementMetadata:
        return ElementMetadata(
            document_id=document_id,
            order_index=self.order_index,
            section_id=self.sections_stack[-1].id if self.sections_stack else None,
            page_number=page_number,
            source_file=source_file,
            source_path=source_path,
            source_document=source_document,
        )

    def _paragraph_from_block(
        self,
        block: _TextBlock,
        document_id: UUID,
        page_number: int,
        source_file: str,
        source_path: str,
        source_document: Optional[str],
    ) -> Paragraph:
        return Paragraph(
            metadata=self._element_metadata(
                document_id,
                page_number,
                source_file,
                source_path,
                source_document,
            ),
            content=block.text,
        )

    def _table_from_block(
        self,
        block: _TableBlock,
        document_id: UUID,
        page_number: int,
        source_file: str,
        source_path: str,
        source_document: Optional[str],
    ) -> Table:
        return Table(
            metadata=self._element_metadata(
                document_id,
                page_number,
                source_file,
                source_path,
                source_document,
            ),
            rows=[
                TableRow(
                    row_index=row_index,
                    cells=[
                        TableCell(content=content, column_index=column_index)
                        for column_index, content in enumerate(row)
                    ],
                )
                for row_index, row in enumerate(block.rows)
            ],
        )

    def parse(
        self,
        pdf_document: fitz.Document,
        source_path: Path,
        source_document: Optional[str] = None,
        document_title: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Document:
        """Convert an open PyMuPDF document into the common internal model."""
        if pdf_document.is_closed:
            raise PDFParserError("Cannot parse a closed PDF document")
        if pdf_document.page_count == 0:
            raise PDFParserError("Cannot parse a PDF document with no pages")

        self.order_index = 0
        self.sections_stack = []
        source_path = Path(source_path).resolve()
        source_file = source_path.name
        document_id = uuid4()
        root_elements: List[Union[Paragraph, Table]] = []
        top_level_sections: List[Section] = []
        extracted_elements = 0

        metadata = pdf_document.metadata or {}
        title = document_title or metadata.get("title") or source_path.stem

        for page in pdf_document:
            blocks = _page_blocks(page)
            body_font_size = _body_font_size(blocks)
            for block in blocks:
                if isinstance(block, _TextBlock):
                    heading_level = _heading_level(block, body_font_size)
                    if heading_level is not None:
                        self._push_section(block.text, heading_level, top_level_sections)
                        continue
                    element = self._paragraph_from_block(
                        block,
                        document_id,
                        page.number + 1,
                        source_file,
                        str(source_path),
                        source_document,
                    )
                else:
                    element = self._table_from_block(
                        block,
                        document_id,
                        page.number + 1,
                        source_file,
                        str(source_path),
                        source_document,
                    )

                self._add_element(element, root_elements)
                self.order_index += 1
                extracted_elements += 1

        if extracted_elements == 0:
            raise PDFParserError(
                "PDF contains no extractable text or tables; scanned PDFs are not supported"
            )

        dm = DocumentMetadata(
            source=str(source_path),
            author=author or metadata.get("author") or None,
            creation_date=metadata.get("creationDate") or None,
            modification_date=metadata.get("modDate") or None,
            version=metadata.get("format") or None,
            document_type=source_document,
        )
        # attach parser version
        dm.custom_metadata["parser_version"] = PARSER_VERSION

        return Document(
            title=title,
            metadata=dm,
            document_id=document_id,
            sections=top_level_sections,
            elements=root_elements,
        )


def parse_pdf_document(
    pdf_path: Path,
    source_document: Optional[str] = None,
    document_title: Optional[str] = None,
    author: Optional[str] = None,
) -> Document:
    """Open and parse a PDF file, closing the source document afterwards."""
    pdf_path = Path(pdf_path)
    try:
        with fitz.open(pdf_path) as pdf_document:
            return PDFParser().parse(
                pdf_document,
                pdf_path,
                source_document=source_document,
                document_title=document_title,
                author=author,
            )
    except PDFParserError:
        raise
    except Exception as error:
        raise PDFParserError(f"Failed to parse PDF file {pdf_path}: {error}") from error


__all__ = ["PDFParser", "PDFParserError", "parse_pdf_document"]
