"""
Internal document models for the proposal AI agent - Composition-Based RAG Pipeline.

Design Principles (Composition over Inheritance):
---------------------------------------------------
1. NO INHERITANCE: Each element type is a standalone dataclass with a composed
   ElementMetadata object. This avoids Python dataclass field ordering issues.

2. UNIFORM METADATA: ElementMetadata centralizes all per-element tracking:
   - UUID for vector DB indexing
   - order_index for reading order preservation
   - section_id for hierarchical context
   - source tracking for traceability

3. TYPE SAFETY: Using slots=True on all dataclasses for memory efficiency and
   attribute access safety. No runtime attribute assignment.

4. COMPOSITION OVER EXTENSION: Each element type composes ElementMetadata rather
   than inheriting from a base class. This provides:
   - Clean field ordering in dataclasses
   - Explicit metadata access (element.metadata.*)
   - No method resolution order complexity
   - Easy serialization and deserialization

5. ELEMENT TYPE UNION: Multiple element types exist (Paragraph, Table, Image,
   ListElement); code using elements works with Union[Paragraph, Table, ...].

Architecture:
--------------
    Document
    ├── metadata (DocumentMetadata)
    ├── sections (Section[])
    │   ├── heading (string)
    │   ├── elements (Paragraph | Table | Image | ListElement)[])
    │   └── subsections (Section[])
    └── elements (Paragraph | Table | Image | ListElement)[])

Each element has explicit metadata: element.metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from uuid import UUID, uuid4


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ParagraphStyle(str, Enum):
    """Enumeration of paragraph styling/semantic types."""
    NORMAL = "normal"
    EMPHASIS = "emphasis"
    QUOTE = "quote"
    CODE = "code"
    CAPTION = "caption"


class ListType(str, Enum):
    """Enumeration of list types."""
    UNORDERED = "unordered"
    ORDERED = "ordered"
    CHECKLIST = "checklist"


# ============================================================================
# DOCUMENT METADATA
# ============================================================================


@dataclass(slots=True)
class DocumentMetadata:
    """
    Document-level metadata with structured fields.
    
    Attributes:
        source: The file path or URI of the source document
        author: Document author or creator
        creation_date: When the document was created
        modification_date: Last modification timestamp
        version: Document version string
        language: ISO 639-1 language code (e.g., 'en', 'fr')
        document_type: Classification (e.g., 'proposal', 'report', 'contract')
        custom_metadata: Extension point for additional metadata
    """
    source: str
    author: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    version: Optional[str] = None
    language: str = "en"
    document_type: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ElementMetadata:
    """
    Per-element metadata shared across all content types.
    
    Design Decision: Composition-based approach. Every element has an ElementMetadata
    instance rather than inheriting metadata fields. This avoids dataclass field
    ordering issues and provides explicit metadata access patterns.
    
    Attributes:
        id: Unique UUID for this element (vector DB indexing, citations)
        document_id: UUID of parent Document (no tree traversal needed)
        section_id: Optional UUID of parent Section (None for root-level)
        order_index: Position in document traversal (0-based, monotonically increasing)
                     Critical for RAG: preserves reading order across chunking
        page_number: Page number if source is paginated (None for digital)
        source_file: Filename of source document (e.g., 'proposal.docx')
        source_path: Full path to source file for reprocessing/tracking
        source_document: Logical document name/type for filtering/grounding
    """
    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    order_index: int = 0
    section_id: Optional[UUID] = None
    parent_element_id: Optional[UUID] = None
    page_number: Optional[int] = None
    source_file: str = ""
    source_path: Optional[str] = None
    source_document: Optional[str] = None


# ============================================================================
# TABLE COMPONENTS
# ============================================================================


@dataclass(slots=True)
class TableCell:
    """
    Represents a single cell within a table row.
    
    Attributes:
        content: Text content of the cell
        cell_type: 'header', 'data', or custom type
        column_index: Position in the row (0-based)
        properties: Optional formatting metadata (span, alignment, etc.)
    """
    content: str
    cell_type: str = "data"
    column_index: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TableRow:
    """
    Represents a single row within a table.
    
    Attributes:
        cells: List of TableCell objects in row
        row_index: Position in the parent table (0-based)
        properties: Optional row-level metadata (height, styling, etc.)
    """
    cells: List[TableCell] = field(default_factory=list)
    row_index: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LIST COMPONENTS
# ============================================================================


@dataclass(slots=True)
class ListItem:
    """
    Represents a single item in a list.
    
    Attributes:
        content: Text content of the list item
        level: Nesting level (0 for top-level, increases with sublists)
        properties: Optional metadata
    """
    content: str
    level: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DOCUMENT ELEMENTS (COMPOSITION-BASED)
# ============================================================================


@dataclass(slots=True)
class Paragraph:
    """
    Represents a paragraph or block of prose text.
    
    Attributes:
        metadata: ElementMetadata (id, order_index, source tracking)
        content: The paragraph text
        style: Semantic style (normal, quote, code, etc.)
        properties: Optional formatting metadata
    """
    metadata: ElementMetadata
    content: str
    style: ParagraphStyle = ParagraphStyle.NORMAL
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Table:
    """
    Represents a tabular data structure.
    
    Attributes:
        metadata: ElementMetadata (id, order_index, source tracking)
        rows: List of TableRow objects
        caption: Descriptive title for the table
        table_type: Classification (e.g., 'data', 'comparison', 'financial')
        properties: Optional table-level metadata (width, borders, etc.)
    """
    metadata: ElementMetadata
    rows: List[TableRow] = field(default_factory=list)
    caption: Optional[str] = None
    table_type: str = "data"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Image:
    """
    Represents an embedded image or figure.
    
    Attributes:
        metadata: ElementMetadata (id, order_index, source tracking)
        alt_text: Alternative text description
        caption: Optional figure caption or title
        properties: Metadata (width, height, format, etc.)
    """
    metadata: ElementMetadata
    alt_text: str
    caption: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ListElement:
    """
    Represents a list structure (unordered, ordered, or checklist).
    
    Note: Class name is ListElement to avoid conflict with Python built-in list
    and typing.List.
    
    Attributes:
        metadata: ElementMetadata (id, order_index, source tracking)
        items: List of ListItem objects
        list_type: Type of list (unordered, ordered, checklist)
        properties: Optional list-level metadata
    """
    metadata: ElementMetadata
    items: List[ListItem] = field(default_factory=list)
    list_type: ListType = ListType.UNORDERED
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TYPE ALIASES FOR ELEMENT UNIONS
# ============================================================================

# Union type for all element types
DocumentElement = Union[Paragraph, Table, Image, ListElement]


# ============================================================================
# SECTION & DOCUMENT
# ============================================================================


@dataclass(slots=True)
class Section:
    """
    Represents a logical section or subsection of a document.
    
    Design Decision: Section is NOT a DocumentElement. Sections are structural
    containers. Section.heading is a plain string (semantic structure), and
    Section contains a list of mixed element types via composition.
    
    Attributes:
        heading: Title of this section
        id: Unique UUID for this section (for parent references in elements)
        elements: Content blocks in this section (Paragraph, Table, Image, ListElement)
        subsections: Nested subsections
        section_level: Depth in hierarchy (1 = top, increases with nesting)
        properties: Optional section-level metadata
    """
    heading: str
    id: UUID = field(default_factory=uuid4)
    elements: List[DocumentElement] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)
    section_level: int = 1
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """
    Represents a complete document.
    
    Attributes:
        title: Document title
        metadata: Structured document metadata
        document_id: Unique UUID identifier for retrieval and tracking
        sections: Top-level sections with hierarchy
        elements: Root-level content blocks (not in any section)
    """
    title: str
    metadata: DocumentMetadata
    document_id: UUID = field(default_factory=uuid4)
    sections: List[Section] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)


__all__ = [
    "DocumentMetadata",
    "ElementMetadata",
    "ParagraphStyle",
    "ListType",
    "TableCell",
    "TableRow",
    "ListItem",
    "Paragraph",
    "Table",
    "Image",
    "ListElement",
    "DocumentElement",
    "Section",
    "Document",
]
