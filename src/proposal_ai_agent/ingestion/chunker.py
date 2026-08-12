"""
Structure-aware chunking engine for validated Documents.

This chunker preserves semantic structure and reading order while splitting
content into chunks suitable for defence proposal retrieval.

Key behaviors:
- Chunk by section; never merge content from different sections
- Preserve section heading and path in every chunk
- Never split tables; tables are always complete chunks
- Merge paragraphs until token limits are reached
- Never split inside paragraphs
- Support configurable chunk size and overlap
"""

import logging
from typing import List, Optional, Tuple

from .models import (
    Document,
    Section,
    DocumentElement,
    Paragraph,
    Table,
    Image,
    ListElement,
    ListType,
)
from .chunk_models import Chunk, ChunkConfig
from typing import Dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Tokenization helpers
# ============================================================================

def count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    if not text:
        return 0
    return len(text.split())


def render_table_text(table: Table) -> str:
    """Render table rows into plain text for chunking."""
    if not table.rows:
        return ""

    rendered_rows = []
    for row in table.rows:
        cells = [cell.content.strip() for cell in row.cells]
        rendered_rows.append(" | ".join(cells))
    return "\n".join(rendered_rows)


def render_image_text(image: Image) -> str:
    """Render image metadata into plain text."""
    if image.caption and image.caption.strip():
        return image.caption.strip()
    if image.alt_text and image.alt_text.strip():
        return image.alt_text.strip()
    return "[Image]"


def render_list_text(list_element: ListElement) -> str:
    """Render list items into plain text."""
    lines = []
    for index, item in enumerate(list_element.items, start=1):
        prefix = "-"
        if list_element.list_type == ListType.ORDERED:
            prefix = f"{index}."
        elif list_element.list_type == ListType.CHECKLIST:
            prefix = "- [ ]"

        lines.append(f"{prefix} {item.content.strip()}")
    return "\n".join(lines)


def render_element_text(element: DocumentElement) -> str:
    """Render any document element into a textual representation."""
    if isinstance(element, Paragraph):
        return element.content
    if isinstance(element, Table):
        return render_table_text(element)
    if isinstance(element, Image):
        return render_image_text(element)
    if isinstance(element, ListElement):
        return render_list_text(element)
    return str(element)


def render_heading_text(section_path: List[str]) -> str:
    """Render section path into a heading block that belongs to every chunk."""
    if not section_path:
        return ""
    return "\n".join(section_path)


def _chunk_text(heading_text: str, element_texts: List[str]) -> str:
    """Combine heading text and element texts into the final chunk body."""
    if heading_text:
        if element_texts:
            return f"{heading_text}\n\n" + "\n\n".join(element_texts)
        return heading_text
    return "\n\n".join(element_texts)


def _normalize_text(text: str) -> str:
    """Normalize text for character offset calculations.

    Keep it simple: strip trailing/leading whitespace but preserve spacing
    between elements as chunker does ("\n\n").
    """
    if text is None:
        return ""
    return text.strip()


def _select_overlap_elements(
    items: List[Tuple[DocumentElement, str, int]],
    overlap_budget: int,
) -> List[Tuple[DocumentElement, str, int]]:
    """
    Select trailing paragraph-like items to overlap into the next chunk.

    Overlap is implemented only using paragraph-like content. Table chunks
    are not partially reused.
    """
    if overlap_budget <= 0:
        return []

    selected: List[Tuple[DocumentElement, str, int]] = []
    running_tokens = 0

    for element, text, token_count in reversed(items):
        if isinstance(element, Table):
            # Do not overlap tables; they must remain complete.
            break

        selected.insert(0, (element, text, token_count))
        running_tokens += token_count
        if running_tokens >= overlap_budget:
            break

    return selected


# ============================================================================
# Chunk builder
# ============================================================================

def _create_chunk(
    document_id,
    section_id,
    chunk_index: int,
    elements: List[Tuple[DocumentElement, str, int]],
    heading: str,
    section_path: List[str],
    document_type: Optional[str],
    language: str,
    source_file: str,
    source_path: str,
    char_start: int,
    char_end: int,
    parser_version: Optional[str] = "1.0",
    document_title: Optional[str] = None,
    parent_section_id=None,
) -> Chunk:
    """Create a Chunk from a list of rendered elements."""
    if not elements:
        raise ValueError("Cannot create a chunk with no elements")
    text_parts = [text for _, text, _ in elements if text.strip()]
    heading_text = render_heading_text(section_path)
    text = _chunk_text(heading_text, text_parts)
    token_count = count_tokens(text)
    element_types = {
        "paragraph" if isinstance(element, Paragraph)
        else "table" if isinstance(element, Table)
        else "image" if isinstance(element, Image)
        else "list" if isinstance(element, ListElement)
        else "unknown"
        for element, _, _ in elements
    }
    element_type = next(iter(element_types)) if len(element_types) == 1 else "mixed"

    # derive source info
    source_file_local = source_file or (elements[0][0].metadata.source_file if elements else "")
    source_path_local = source_path or (elements[0][0].metadata.source_path or "")

    return Chunk(
        document_id=document_id,
        section_id=section_id,
        section_path=section_path,
        heading=heading,
        document_type=document_type,
        language=language or "en",
        source_file=source_file_local,
        source_path=source_path_local,
        chunk_index=chunk_index,
        order_start=elements[0][0].metadata.order_index,
        order_end=elements[-1][0].metadata.order_index,
        text=text,
        token_count=token_count,
        document_title=document_title,
        parent_section_id=parent_section_id,
        element_type=element_type,
        char_start=char_start,
        char_end=char_end,
        parser_version=parser_version,
    )


def _should_start_new_chunk(
    current_token_count: int,
    element_token_count: int,
    chunk_size: int,
    has_content: bool,
) -> bool:
    """Decide whether to start a new chunk before adding the next element."""
    if not has_content:
        return False
    return current_token_count + element_token_count > chunk_size


def _add_elements_to_chunk(
    elements: List[Tuple[DocumentElement, str, int]],
    new_element: Tuple[DocumentElement, str, int],
) -> Tuple[List[Tuple[DocumentElement, str, int]], int]:
    """Add a rendered element to the current chunk and return updated token count."""
    elements.append(new_element)
    return sum(token_count for _, _, token_count in elements)


def _chunk_elements(
    document_id,
    section_id: Optional[str],
    section_path: List[str],
    heading: str,
    content_elements: List[DocumentElement],
    config: ChunkConfig,
    starting_chunk_index: int,
    document_type: Optional[str],
    language: str,
    source_file: str,
    source_path: str,
    parser_version: Optional[str],
    document_title: Optional[str] = None,
    parent_section_id=None,
) -> Tuple[List[Chunk], int]:
    """Chunk a list of section-specific content elements."""
    chunks: List[Chunk] = []
    chunk_index = starting_chunk_index
    heading_text = render_heading_text(section_path)
    heading_token_count = count_tokens(heading_text)

    current_block: List[Tuple[DocumentElement, str, int]] = []
    current_token_count = heading_token_count
    pending_overlap: List[Tuple[DocumentElement, str, int]] = []

    # Precompute normalized section text and element char spans
    element_texts: List[str] = [render_element_text(e) for e in content_elements]
    heading_text_norm = _normalize_text(heading_text)

    # Build section_text and element char spans
    section_text_parts: List[str] = []
    element_spans: List[Tuple[int, int]] = []
    cursor = 0
    if heading_text_norm:
        section_text_parts.append(heading_text_norm)
        cursor += len(heading_text_norm)
        if element_texts:
            cursor += 2  # for the separator between heading and first element

    for idx, etext in enumerate(element_texts):
        start = cursor
        section_text_parts.append(etext)
        cursor = start + len(etext)
        end = cursor
        element_spans.append((start, end))
        # add separator except after last
        if idx != len(element_texts) - 1:
            cursor += 2

    section_text = "\n\n".join(section_text_parts) if section_text_parts else ""

    def finalize_current_block() -> None:
        nonlocal chunks, chunk_index, current_block, current_token_count, pending_overlap
        if not current_block:
            return

        # determine char offsets for this block
        if current_block:
            # find indices of first and last element in content_elements
            first_elem = current_block[0][0]
            last_elem = current_block[-1][0]
            try:
                first_index = content_elements.index(first_elem)
                last_index = content_elements.index(last_elem)
                # char_start is start of first element; if no elements, 0
                char_start = element_spans[first_index][0] if element_spans else 0
                char_end = element_spans[last_index][1] if element_spans else len(section_text)
            except ValueError:
                char_start = 0
                char_end = len(section_text)
        else:
            char_start = 0
            char_end = len(section_text)

        chunk = _create_chunk(
            document_id=document_id,
            section_id=section_id,
            chunk_index=chunk_index,
            elements=current_block,
            heading=heading,
            section_path=section_path,
            document_type=document_type,
            language=language,
            source_file=source_file,
            source_path=source_path,
            char_start=char_start,
            char_end=char_end,
            parser_version=parser_version,
            document_title=document_title,
            parent_section_id=parent_section_id,
        )
        chunks.append(chunk)
        chunk_index += 1

        overlap_budget = max(1, int(config.chunk_size * config.overlap_ratio))
        pending_overlap = _select_overlap_elements(current_block, overlap_budget)

        current_block = []
        current_token_count = heading_token_count

    def start_new_block_with_overlap():
        nonlocal current_block, current_token_count
        current_block = []
        current_token_count = heading_token_count
        for item in pending_overlap:
            current_block.append(item)
            current_token_count += item[2]

    def add_element(element: DocumentElement) -> None:
        nonlocal chunk_index, current_block, current_token_count, pending_overlap
        text = render_element_text(element)
        token_count = count_tokens(text)

        if isinstance(element, Table):
            # Force table chunks to remain intact.
            if current_block:
                finalize_current_block()

            current_block = [(element, text, token_count)]
            current_token_count = heading_token_count + token_count
            # table is single element block; compute its span
            if content_elements:
                try:
                    tbl_index = content_elements.index(element)
                    char_start = element_spans[tbl_index][0]
                    char_end = element_spans[tbl_index][1]
                except ValueError:
                    char_start = 0
                    char_end = len(section_text)
            else:
                char_start = 0
                char_end = len(section_text)

            chunks.append(_create_chunk(
                document_id=document_id,
                section_id=section_id,
                chunk_index=chunk_index,
                elements=current_block,
                heading=heading,
                section_path=section_path,
                document_type=document_type,
                language=language,
                source_file=source_file,
                source_path=source_path,
                char_start=char_start,
                char_end=char_end,
                parser_version=parser_version,
                document_title=document_title,
                parent_section_id=parent_section_id,
            ))
            chunk_index += 1
            current_block = []
            current_token_count = heading_token_count
            pending_overlap = []
            return

        needs_new_chunk = _should_start_new_chunk(
            current_token_count,
            token_count,
            config.chunk_size,
            has_content=bool(current_block),
        )

        if needs_new_chunk:
            finalize_current_block()
            start_new_block_with_overlap()

        current_block.append((element, text, token_count))
        current_token_count += token_count

    for element in content_elements:
        add_element(element)

    finalize_current_block()
    return chunks, chunk_index


# ============================================================================
# Document chunking
# ============================================================================

def _chunk_section(
    document_id,
    section: Section,
    parent_path: List[str],
    config: ChunkConfig,
    starting_chunk_index: int,
    document_title: Optional[str] = None,
    parent_section_id=None,
) -> Tuple[List[Chunk], int]:
    """Chunk a section and its subsections recursively."""
    chunks: List[Chunk] = []
    section_path = parent_path + [section.heading] if section.heading else parent_path
    heading = section.heading

    # The caller (chunk_document) will monkeypatch in document metadata via
    # outer variables; to keep changes minimal we fetch them from section.properties
    # if present, else use defaults.
    document_type = section.properties.get("document_type") if section.properties else None
    language = section.properties.get("language") if section.properties else "en"
    source_file = section.properties.get("source_file") if section.properties else ""
    source_path = section.properties.get("source_path") if section.properties else ""
    parser_version = section.properties.get("parser_version") if section.properties else "1.0"

    section_chunks, next_index = _chunk_elements(
        document_id=document_id,
        section_id=section.id,
        section_path=section_path,
        heading=heading,
        content_elements=section.elements,
        config=config,
        starting_chunk_index=starting_chunk_index,
        document_type=document_type,
        language=language,
        source_file=source_file,
        source_path=source_path,
        parser_version=parser_version,
        document_title=document_title,
        parent_section_id=parent_section_id,
    )
    chunks.extend(section_chunks)

    current_index = next_index
    for subsection in section.subsections:
        subsection_chunks, current_index = _chunk_section(
            document_id=document_id,
            section=subsection,
            parent_path=section_path,
            config=config,
            starting_chunk_index=current_index,
            document_title=document_title,
            parent_section_id=section.id,
        )
        chunks.extend(subsection_chunks)

    return chunks, current_index


def chunk_document(document: Document, config: Optional[ChunkConfig] = None) -> List[Chunk]:
    """Create structure-aware chunks from a validated Document."""
    if config is None:
        config = ChunkConfig()

    chunks: List[Chunk] = []
    chunk_index = 0

    # Propagate document-level metadata into section.properties for access during chunking
    def _propagate_doc_metadata(sec: Section):
        if not hasattr(sec, "properties"):
            sec.properties = {}
        sec.properties["document_type"] = document.metadata.document_type if document.metadata else None
        sec.properties["language"] = document.metadata.language if document.metadata else "en"
        sec.properties["source_file"] = document.metadata.source if document.metadata else ""
        sec.properties["source_path"] = document.metadata.source if document.metadata else ""
        sec.properties["parser_version"] = (document.metadata.custom_metadata.get("parser_version")
                                             if document.metadata and document.metadata.custom_metadata else "1.0")
        for s in sec.subsections:
            _propagate_doc_metadata(s)

    for s in document.sections:
        _propagate_doc_metadata(s)

    # Chunk root-level elements if any
    if document.elements:
        root_chunks, chunk_index = _chunk_elements(
            document_id=document.document_id,
            section_id=None,
            section_path=[],
            heading="",
            content_elements=document.elements,
            config=config,
            starting_chunk_index=chunk_index,
            document_type=document.metadata.document_type if document.metadata else None,
            language=document.metadata.language if document.metadata else "en",
            source_file=document.metadata.source if document.metadata else "",
            source_path=document.metadata.source if document.metadata else "",
            parser_version=(document.metadata.custom_metadata.get("parser_version")
                            if document.metadata and document.metadata.custom_metadata else "1.0"),
            document_title=document.title,
            parent_section_id=None,
        )
        chunks.extend(root_chunks)

    for section in document.sections:
        section_chunks, chunk_index = _chunk_section(
            document_id=document.document_id,
            section=section,
            parent_path=[],
            config=config,
            starting_chunk_index=chunk_index,
            document_title=document.title,
            parent_section_id=None,
        )
        chunks.extend(section_chunks)

    # Populate global and per-section sequence metadata
    total = len(chunks)
    # map section_id -> list of chunk positions
    section_map: Dict[Optional[str], List[int]] = {}
    for idx, ch in enumerate(chunks):
        # set total_chunks and global flags
        ch.total_chunks = total
        ch.is_first_chunk = (idx == 0)
        ch.is_last_chunk = (idx == total - 1)
        # group by section
        sid = ch.section_id
        section_map.setdefault(sid, []).append(idx)

    # set per-section indices
    for sid, indices in section_map.items():
        total_in_section = len(indices)
        for pos, global_idx in enumerate(indices):
            ch = chunks[global_idx]
            ch.total_chunks_in_section = total_in_section
            ch.section_chunk_index = pos

    return chunks


__all__ = [
    "ChunkConfig",
    "Chunk",
    "chunk_document",
]
