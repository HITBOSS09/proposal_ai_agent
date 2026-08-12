"""
Document Validator - Inspects internal Document model for structural integrity.

The validator performs non-destructive validation, checking for:
- Logical consistency (no duplicate IDs or order_index)
- Structural integrity (no orphan elements, broken hierarchy)
- Metadata completeness (required fields populated)
- Semantic correctness (valid heading levels, parent references)
- Data quality (empty sections, redundant elements)

Validation approach:
1. Traverse entire document tree structure
2. Collect all elements and sections
3. Check for violations of constraints
4. Return detailed results with errors, warnings, and statistics

The validator does NOT:
- Modify the document in any way
- Clean or normalize text
- Repair issues
- Perform content validation (grammar, spelling, etc.)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from uuid import UUID
from enum import Enum
import re

from .models import (
    Document,
    Section,
    DocumentElement,
    DocumentMetadata,
    ElementMetadata,
    Paragraph,
    Table,
    Image,
    ListElement,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_element_type_name(element) -> str:
    """Get human-readable element type name from element instance."""
    if isinstance(element, Paragraph):
        return "paragraph"
    elif isinstance(element, Table):
        return "table"
    elif isinstance(element, Image):
        return "image"
    elif isinstance(element, ListElement):
        return "list"
    else:
        return "unknown"


def is_structural_container_section(heading: Optional[str]) -> bool:
    """
    Determine if a section heading indicates a structural/container section.
    
    Container sections are organizational by nature and may legitimately have
    no direct content or subsections. Examples include Table of Contents, 
    Appendices, and Form sections which serve as organizational markers.
    
    Args:
        heading: Section heading text
        
    Returns:
        True if the heading indicates a container section, False otherwise
    """
    import re

    if not heading:
        return False

    h = heading.strip().lower()

    patterns = [
        re.compile(r"^form\s*\d+[a-z]?", re.I),
        re.compile(r"^form\b", re.I),
        re.compile(r"^appendi", re.I),
        re.compile(r"^annex", re.I),
        re.compile(r"^annexure", re.I),
        re.compile(r"^table\s+of\s+contents", re.I),
        re.compile(r"^contents?\b", re.I),
        re.compile(r"^section.*(appendi|annex)", re.I),
        re.compile(r"^section\s+[a-z0-9]+\b", re.I),
        re.compile(r"^chapter\b", re.I),
        re.compile(r"^part\b", re.I),
    ]

    for p in patterns:
        if p.search(h):
            return True

    return False


class SectionType(str, Enum):
    COVER_PAGE = "COVER_PAGE"
    TABLE_OF_CONTENTS = "TABLE_OF_CONTENTS"
    TEMPLATE_FORM = "TEMPLATE_FORM"
    CONTAINER_SECTION = "CONTAINER_SECTION"
    CERTIFICATE = "CERTIFICATE"
    SIGNATURE_PAGE = "SIGNATURE_PAGE"
    PLACEHOLDER_SECTION = "PLACEHOLDER_SECTION"
    CONTENT_SECTION = "CONTENT_SECTION"


def classify_section(section: Section, is_top_level_first: bool = False) -> SectionType:
    """Classify a section using heuristics into the defined SectionType set.

    Heuristics use heading patterns and element composition.
    """
    heading = (section.heading or "").strip()
    h = heading.lower()

    # Table of contents detection
    if re.search(r"table\s*of\s*contents|^contents$", h, re.I):
        return SectionType.TABLE_OF_CONTENTS

    # Cover page heuristic (first top-level section and short content)
    if is_top_level_first:
        if not h or "title" in h or "cover" in h:
            # ensure little body content
            total_chars = 0
            for elem in section.elements:
                if isinstance(elem, Paragraph):
                    total_chars += len(elem.content or "")
            if total_chars < 300 and not section.subsections:
                return SectionType.COVER_PAGE

    # Template form detection (heading 'form' or many tables)
    if re.match(r"^form\b|^form\s*\d+[a-z]?", h, re.I):
        return SectionType.TEMPLATE_FORM

    # Certificate / signature detection
    if re.search(r"signature|signed by|signatory|certificate", h, re.I):
        return SectionType.SIGNATURE_PAGE if "signature" in h else SectionType.CERTIFICATE

    # Placeholder detection
    if any(tok in h for tok in ["left blank", "to be completed", "tbd", "placeholder"]):
        return SectionType.PLACEHOLDER_SECTION

    # Annex/Appendix/Annexure => container
    if re.match(r"^appendi|^annex", h, re.I):
        return SectionType.CONTAINER_SECTION

    # If this section has subsections, treat as container
    if section.subsections:
        return SectionType.CONTAINER_SECTION

    # If section contains tables but no paragraphs, may be template form
    has_tables = any(isinstance(e, Table) for e in section.elements)
    has_paragraphs = any(isinstance(e, Paragraph) and (e.content or "").strip() for e in section.elements)
    if has_tables and not has_paragraphs:
        return SectionType.TEMPLATE_FORM

    # Default to content
    return SectionType.CONTENT_SECTION


# ============================================================================
# VALIDATION RESULT TYPES
# ============================================================================

@dataclass
class ValidationError:
    """Represents a validation error."""
    code: str
    message: str
    element_id: Optional[UUID] = None
    element_type: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class ValidationWarning:
    """Represents a validation warning (non-blocking issue)."""
    code: str
    message: str
    element_id: Optional[UUID] = None
    element_type: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class ValidationStatistics:
    """Document structure statistics."""
    total_elements: int = 0
    total_sections: int = 0
    element_breakdown: Dict[str, int] = field(default_factory=dict)
    max_nesting_depth: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    unique_uuids: int = 0
    unique_order_indices: int = 0


@dataclass
class ValidationResult:
    """
    Complete validation result for a document.
    
    Contains all errors, warnings, and statistics from the validation pass.
    A valid document has errors = [] and appropriate statistics.
    
    Attributes:
        document_id: UUID of validated document
        is_valid: True if no errors (warnings allowed)
        errors: List of validation errors
        warnings: List of validation warnings
        statistics: Document structure statistics
    """
    document_id: UUID
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    statistics: ValidationStatistics = field(default_factory=ValidationStatistics)


# ============================================================================
# DOCUMENT VALIDATOR
# ============================================================================

class DocumentValidator:
    """
    Validates internal Document model without modifying it.
    
    Performs comprehensive validation of:
    - Duplicate UUID detection
    - Order index monotonicity
    - Section hierarchy integrity
    - Orphan element detection
    - Metadata completeness
    - Parent reference validity
    - Empty section detection
    - Heading level correctness
    
    Usage:
        validator = DocumentValidator()
        result = validator.validate(document)
        if not result.is_valid:
            for error in result.errors:
                print(f"ERROR: {error.message}")
    """
    
    def __init__(self):
        """Initialize validator."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationWarning] = []
        self.statistics: ValidationStatistics = ValidationStatistics()
        
        # Tracking sets/dicts for duplicate detection
        self.seen_uuids: Set[UUID] = set()
        self.uuid_locations: Dict[UUID, List[str]] = defaultdict(list)
        self.seen_order_indices: Set[int] = set()
        self.order_index_locations: Dict[int, List[Tuple[str, UUID]]] = defaultdict(list)
        
        # Tracking for element validation
        self.all_element_ids: List[UUID] = []
        self.element_id_to_type: Dict[UUID, str] = {}
        self.section_ids: Set[UUID] = set()
    
    def validate(self, document: Document) -> ValidationResult:
        """
        Validate a complete Document.
        
        Args:
            document: The Document to validate
            
        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        # Reset state for new validation
        self._reset()
        
        # Run validation checks
        self._validate_document_metadata(document)
        self._validate_elements_tree(document)
        self._validate_sections_tree(document)
        self._check_duplicate_uuids()
        self._check_duplicate_order_indices()
        self._check_parent_references(document)
        self._check_section_hierarchy(document)
        self._check_empty_sections(document)
        
        # Build statistics
        self._compute_statistics()
        
        # Determine validity
        is_valid = len(self.errors) == 0
        
        return ValidationResult(
            document_id=document.document_id,
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            statistics=self.statistics,
        )
    
    def _reset(self):
        """Reset all tracking state."""
        self.errors = []
        self.warnings = []
        self.statistics = ValidationStatistics()
        self.seen_uuids = set()
        self.uuid_locations = defaultdict(list)
        self.seen_order_indices = set()
        self.order_index_locations = defaultdict(list)
        self.all_element_ids = []
        self.element_id_to_type = {}
        self.section_ids = set()
    
    # ========================================================================
    # VALIDATION CHECKS
    # ========================================================================
    
    def _validate_document_metadata(self, document: Document):
        """Validate document-level metadata."""
        if not document.title or not document.title.strip():
            self.errors.append(ValidationError(
                code="MISSING_DOCUMENT_TITLE",
                message="Document title is empty or whitespace-only",
                element_id=document.document_id
            ))
        
        if not document.metadata:
            self.errors.append(ValidationError(
                code="MISSING_METADATA",
                message="Document metadata is None",
                element_id=document.document_id
            ))
        else:
            # Check required metadata fields
            if not document.metadata.source:
                self.warnings.append(ValidationWarning(
                    code="MISSING_SOURCE",
                    message="Document metadata source is empty",
                    element_id=document.document_id
                ))
            
            if not document.metadata.language:
                self.warnings.append(ValidationWarning(
                    code="MISSING_LANGUAGE",
                    message="Document metadata language is empty",
                    element_id=document.document_id
                ))
    
    def _validate_elements_tree(self, document: Document):
        """Validate all elements (root-level and sectioned)."""
        # Validate root elements
        for elem in document.elements:
            self._validate_element(elem, location="root", document_id=document.document_id)
        
        # Validate elements in sections
        for section in document.sections:
            self._validate_section_contents(section, document.document_id)
    
    def _validate_section_contents(self, section: Section, document_id: UUID, depth: int = 1):
        """Recursively validate elements within a section."""
        # Validate elements in this section
        for elem in section.elements:
            self._validate_element(
                elem,
                location=f"section '{section.heading}'",
                document_id=document_id,
                section_id=section.id
            )
        
        # Recursively validate subsections
        for subsection in section.subsections:
            self._validate_section_contents(subsection, document_id, depth + 1)
    
    def _validate_element(
        self,
        element: DocumentElement,
        location: str,
        document_id: UUID,
        section_id: Optional[UUID] = None,
    ):
        """Validate a single element."""
        # Check element ID is UUID
        if not isinstance(element.metadata.id, UUID):
            self.errors.append(ValidationError(
                code="INVALID_ELEMENT_ID",
                message=f"Element ID is not a UUID in {location}",
                element_id=element.metadata.id,
                element_type=get_element_type_name(element),
            ))
            return  # Can't continue without valid ID
        
        # Track element ID and type
        self.all_element_ids.append(element.metadata.id)
        self.element_id_to_type[element.metadata.id] = get_element_type_name(element)
        self.uuid_locations[element.metadata.id].append(location)
        
        # Check document_id matches
        if element.metadata.document_id != document_id:
            self.errors.append(ValidationError(
                code="MISMATCHED_DOCUMENT_ID",
                message=f"Element has different document_id than parent in {location}",
                element_id=element.metadata.id,
                element_type=get_element_type_name(element),
                details={"expected": str(document_id), "actual": str(element.metadata.document_id)}
            ))
        
        # Check order_index is non-negative integer
        if not isinstance(element.metadata.order_index, int) or element.metadata.order_index < 0:
            self.errors.append(ValidationError(
                code="INVALID_ORDER_INDEX",
                message=f"Element has invalid order_index: {element.metadata.order_index} in {location}",
                element_id=element.metadata.id,
                element_type=get_element_type_name(element),
            ))
        else:
            self.seen_order_indices.add(element.metadata.order_index)
            self.order_index_locations[element.metadata.order_index].append(
                (location, element.metadata.id)
            )
        
        # Check source_file is populated
        if not element.metadata.source_file or not element.metadata.source_file.strip():
            self.warnings.append(ValidationWarning(
                code="MISSING_SOURCE_FILE",
                message=f"Element has empty source_file in {location}",
                element_id=element.metadata.id,
                element_type=get_element_type_name(element),
            ))
        
        # Check section_id matches if element is inside a section
        if section_id is not None:
            if element.metadata.section_id is None:
                self.errors.append(ValidationError(
                    code="MISSING_SECTION_ID",
                    message=f"Element in section has no section_id in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
            elif element.metadata.section_id != section_id:
                self.errors.append(ValidationError(
                    code="MISMATCHED_SECTION_ID",
                    message=f"Element section_id doesn't match parent in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                    details={"expected": str(section_id), "actual": str(element.metadata.section_id)}
                ))
        else:
            # Root-level element should have section_id = None
            if element.metadata.section_id is not None:
                self.errors.append(ValidationError(
                    code="UNEXPECTED_SECTION_ID",
                    message=f"Root-level element has non-None section_id in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
        
        # Type-specific validation
        if isinstance(element, Paragraph):
            if not element.content or not element.content.strip():
                self.warnings.append(ValidationWarning(
                    code="EMPTY_PARAGRAPH",
                    message=f"Paragraph has empty content in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
        
        elif isinstance(element, Table):
            if not element.rows:
                self.errors.append(ValidationError(
                    code="EMPTY_TABLE",
                    message=f"Table has no rows in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
            else:
                # Check table structure consistency
                first_row_cell_count = len(element.rows[0].cells) if element.rows else 0
                for row_idx, row in enumerate(element.rows[1:], start=1):
                    if len(row.cells) != first_row_cell_count:
                        self.warnings.append(ValidationWarning(
                            code="TABLE_INCONSISTENT_COLUMNS",
                            message=f"Table row {row_idx} has {len(row.cells)} cells, expected {first_row_cell_count} in {location}",
                            element_id=element.metadata.id,
                            element_type=get_element_type_name(element),
                            details={"row_index": row_idx}
                        ))
        
        elif isinstance(element, Image):
            if not element.alt_text or not element.alt_text.strip():
                self.warnings.append(ValidationWarning(
                    code="MISSING_ALT_TEXT",
                    message=f"Image has empty alt_text in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
        
        elif isinstance(element, ListElement):
            if not element.items:
                self.errors.append(ValidationError(
                    code="EMPTY_LIST",
                    message=f"List has no items in {location}",
                    element_id=element.metadata.id,
                    element_type=get_element_type_name(element),
                ))
    
    def _validate_sections_tree(self, document: Document):
        """Recursively validate section structure."""
        for section in document.sections:
            self._validate_section(section)
    
    def _validate_section(self, section: Section, parent_level: int = 0):
        """Validate a single section and its subsections."""
        # Track section ID
        if not isinstance(section.id, UUID):
            self.errors.append(ValidationError(
                code="INVALID_SECTION_ID",
                message=f"Section '{section.heading}' has invalid ID",
            ))
            return  # Can't continue without valid ID
        
        self.section_ids.add(section.id)
        
        # Check heading is not empty
        if not section.heading or not section.heading.strip():
            self.errors.append(ValidationError(
                code="EMPTY_SECTION_HEADING",
                message=f"Section has empty heading",
                element_id=section.id,
            ))
        
        # Check section_level is positive and reasonable
        if section.section_level < 1 or section.section_level > 6:
            self.errors.append(ValidationError(
                code="INVALID_SECTION_LEVEL",
                message=f"Section '{section.heading}' has invalid level: {section.section_level}",
                element_id=section.id,
                details={"level": section.section_level}
            ))
        
        # Check subsection hierarchy
        for subsection in section.subsections:
            # Subsection level should be > parent level
            if subsection.section_level <= section.section_level:
                self.errors.append(ValidationError(
                    code="BROKEN_SECTION_HIERARCHY",
                    message=f"Subsection '{subsection.heading}' level {subsection.section_level} <= parent level {section.section_level}",
                    element_id=subsection.id,
                    details={"parent_level": section.section_level, "subsection_level": subsection.section_level}
                ))
            
            # Recursively validate subsections
            self._validate_section(subsection, section.section_level)
    
    def _check_duplicate_uuids(self):
        """Check for duplicate UUIDs across all elements and sections."""
        # Check element UUIDs
        for elem_id, locations in self.uuid_locations.items():
            if len(locations) > 1:
                self.errors.append(ValidationError(
                    code="DUPLICATE_ELEMENT_UUID",
                    message=f"Duplicate element UUID found at: {locations}",
                    element_id=elem_id,
                ))
        
        # Check section UUIDs don't conflict with element UUIDs
        for section_id in self.section_ids:
            if section_id in self.all_element_ids:
                self.errors.append(ValidationError(
                    code="CONFLICTING_UUID",
                    message=f"UUID {section_id} used for both section and element",
                    element_id=section_id,
                ))
    
    def _check_duplicate_order_indices(self):
        """Check for duplicate order_index values."""
        duplicates = {
            idx: locations
            for idx, locations in self.order_index_locations.items()
            if len(locations) > 1
        }
        
        for order_index, locations in duplicates.items():
            location_strs = [f"{loc[0]} (id={loc[1]})" for loc in locations]
            self.errors.append(ValidationError(
                code="DUPLICATE_ORDER_INDEX",
                message=f"Duplicate order_index {order_index} at: {location_strs}",
                details={"order_index": order_index, "count": len(locations)}
            ))
    
    def _check_parent_references(self, document: Document):
        """Check that all parent references are valid."""
        # Collect all valid document IDs (just the one document)
        valid_document_ids = {document.document_id}
        
        # Collect all valid section IDs during element traversal
        # This is done implicitly during _validate_elements_tree
        
        # Check section_id references
        for elem in document.elements:
            if elem.metadata.section_id is not None:
                if elem.metadata.section_id not in self.section_ids:
                    self.errors.append(ValidationError(
                        code="ORPHAN_SECTION_REFERENCE",
                        message=f"Element references non-existent section {elem.metadata.section_id}",
                        element_id=elem.metadata.id,
                        element_type=get_element_type_name(elem),
                    ))
        
        # Recursively check elements in sections
        def check_section_elements(section: Section):
            for elem in section.elements:
                if elem.metadata.section_id != section.id:
                    self.errors.append(ValidationError(
                        code="ORPHAN_SECTION_REFERENCE",
                        message=f"Element in section '{section.heading}' has mismatched section_id",
                        element_id=elem.metadata.id,
                        element_type=get_element_type_name(elem),
                    ))
            for subsection in section.subsections:
                check_section_elements(subsection)
        
        for section in document.sections:
            check_section_elements(section)
    
    def _check_section_hierarchy(self, document: Document):
        """Check that section hierarchy is structurally sound."""
        # Get all top-level sections
        top_levels = [s.section_level for s in document.sections]
        
        # Top-level sections should all be level 1
        for section in document.sections:
            if section.section_level != 1:
                self.warnings.append(ValidationWarning(
                    code="TOP_LEVEL_SECTION_NOT_LEVEL_1",
                    message=f"Top-level section '{section.heading}' has level {section.section_level}, expected 1",
                    element_id=section.id,
                ))
        
        # Check for gaps in hierarchy (e.g., H1 -> H3 skipping H2)
        def check_subsection_gaps(section: Section):
            for subsection in section.subsections:
                expected_min_level = section.section_level + 1
                if subsection.section_level > expected_min_level + 1:
                    self.warnings.append(ValidationWarning(
                        code="SECTION_LEVEL_GAP",
                        message=f"Gap in section hierarchy: parent level {section.section_level}, child level {subsection.section_level}",
                        element_id=subsection.id,
                        details={"parent_level": section.section_level, "child_level": subsection.section_level}
                    ))
                check_subsection_gaps(subsection)
        
        for section in document.sections:
            check_subsection_gaps(section)
    
    def _check_empty_sections(self, document: Document):
        """
        Check for sections that have no elements and no subsections.
        
        Structural container sections (Table of Contents, Appendix, Forms, etc.)
        are excluded from this check as they legitimately serve an organizational
        purpose without direct content.
        """
        def check_section(section: Section):
            # Only warn if section is empty AND is NOT a structural container
            if not section.elements and not section.subsections:
                # Skip container sections identified by regex patterns
                if is_structural_container_section(section.heading):
                    return

                # Cover page heuristic: skip first/top-level section that looks like a title/cover
                # This will be handled by the outer loop which provides position context.
                # Default: emit warning
                self.warnings.append(ValidationWarning(
                    code="EMPTY_SECTION",
                    message=f"Section '{section.heading}' has no elements or subsections",
                    element_id=section.id,
                ))
            
            for subsection in section.subsections:
                check_section(subsection)
        
        # Determine cover-page candidate (top-level first section)
        for idx, section in enumerate(document.sections):
            # simple cover-page heuristic: first top-level section that clearly looks like a title/cover
            is_cover = False
            if idx == 0:
                # conservative: only treat as cover if heading is empty or contains 'title' or 'cover'
                h = (section.heading or "").strip().lower()
                if (not h) or ("title" in h) or ("cover" in h):
                    # ensure very little body content
                    total_chars = 0
                    for elem in section.elements:
                        if isinstance(elem, Paragraph):
                            total_chars += len(elem.content or "")
                        elif isinstance(elem, ListElement):
                            for it in elem.items:
                                total_chars += len(it.content or "")
                    if (not section.subsections) and (len(section.elements) <= 1) and (total_chars < 300):
                        is_cover = True

            # If cover, do not treat as empty-section
            if is_cover:
                continue

            check_section(section)
    
    def _compute_statistics(self):
        """Compute document structure statistics."""
        # Count unique UUIDs and order indices
        self.statistics.unique_uuids = len(set(self.all_element_ids)) + len(self.section_ids)
        self.statistics.unique_order_indices = len(self.seen_order_indices)
        
        # Count total elements (should be computed during validation)
        self.statistics.total_elements = len(self.all_element_ids)
        self.statistics.total_sections = len(self.section_ids)
        
        # Element breakdown by type
        self.statistics.element_breakdown = defaultdict(int)
        for elem_type in self.element_id_to_type.values():
            self.statistics.element_breakdown[elem_type] += 1
        
        # Error and warning counts
        self.statistics.total_errors = len(self.errors)
        self.statistics.total_warnings = len(self.warnings)


__all__ = [
    "DocumentValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    "ValidationStatistics",
    "is_structural_container_section",
]
