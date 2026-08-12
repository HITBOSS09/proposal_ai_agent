"""
Tests for the Document Validator.

Comprehensive test suite covering:
- Validation of document metadata
- Duplicate UUID detection
- Order index validation
- Section hierarchy validation
- Parent reference validation
- Element validation (paragraphs, tables, images, lists)
- Statistics computation
- Warning detection
"""

import pytest
from uuid import UUID, uuid4

from proposal_ai_agent.ingestion.models import (
    Document,
    DocumentMetadata,
    ElementMetadata,
    Section,
    Paragraph,
    Table,
    TableRow,
    TableCell,
    Image,
    ListElement,
    ListItem,
    ParagraphStyle,
    ListType,
)
from proposal_ai_agent.ingestion.validator import (
    DocumentValidator,
    ValidationError,
    ValidationWarning,
    is_structural_container_section,
)


def make_element_metadata(document_id, order_index, source_file="test.docx", section_id=None, source_path=None, source_document=None, element_id=None):
    return ElementMetadata(
        id=element_id or uuid4(),
        document_id=document_id,
        order_index=order_index,
        section_id=section_id,
        parent_element_id=None,
        page_number=None,
        source_file=source_file,
        source_path=source_path,
        source_document=source_document,
    )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def validator():
    """Create a DocumentValidator instance."""
    return DocumentValidator()


@pytest.fixture
def simple_document():
    """Create a simple valid document for testing."""
    doc_id = uuid4()
    
    para = Paragraph(
        metadata=make_element_metadata(
            document_id=doc_id,
            order_index=0,
            source_file="test.docx",
        ),
        content="Test paragraph"
    )
    
    return Document(
        title="Test Document",
        metadata=DocumentMetadata(
            source="/path/to/test.docx",
            author="Test Author",
            language="en"
        ),
        document_id=doc_id,
        elements=[para]
    )


# ============================================================================
# METADATA VALIDATION TESTS
# ============================================================================

class TestMetadataValidation:
    """Tests for document metadata validation."""
    
    def test_validate_valid_metadata(self, validator, simple_document):
        """Test validation of valid metadata."""
        result = validator.validate(simple_document)
        
        # No metadata errors
        metadata_errors = [e for e in result.errors if "METADATA" in e.code]
        assert len(metadata_errors) == 0
    
    def test_validate_missing_title(self, validator, simple_document):
        """Test validation catches missing document title."""
        simple_document.title = ""
        
        result = validator.validate(simple_document)
        
        errors = [e for e in result.errors if e.code == "MISSING_DOCUMENT_TITLE"]
        assert len(errors) == 1
        assert not result.is_valid
    
    def test_validate_missing_metadata(self, validator, simple_document):
        """Test validation catches missing metadata object."""
        simple_document.metadata = None
        
        result = validator.validate(simple_document)
        
        errors = [e for e in result.errors if e.code == "MISSING_METADATA"]
        assert len(errors) == 1
        assert not result.is_valid
    
    def test_validate_missing_source(self, validator, simple_document):
        """Test validation warns about missing source."""
        simple_document.metadata.source = ""
        
        result = validator.validate(simple_document)
        
        warnings = [w for w in result.warnings if w.code == "MISSING_SOURCE"]
        assert len(warnings) == 1


# ============================================================================
# UUID VALIDATION TESTS
# ============================================================================

class TestUUIDValidation:
    """Tests for UUID validation and duplicate detection."""
    
    def test_validate_duplicate_element_uuids(self, validator):
        """Test detection of duplicate element UUIDs."""
        doc_id = uuid4()
        shared_id = uuid4()
        
        elem1 = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
                element_id=shared_id,
            ),
            content="First"
        )
        
        elem2 = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=1,
                source_file="test.docx",
                element_id=shared_id,
            ),
            content="Second"
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem1, elem2]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "DUPLICATE_ELEMENT_UUID"]
        assert len(errors) >= 1
        assert not result.is_valid
    
    def test_validate_duplicate_section_and_element_uuid(self, validator):
        """Test detection when section and element share UUID."""
        doc_id = uuid4()
        shared_id = uuid4()
        
        section = Section(
            heading="Test Section",
            id=shared_id,  # This ID is also
            section_level=1
        )
        
        elem = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
                section_id=None,
                element_id=shared_id,
            ),
            content="Test"
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section],
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if "UUID" in e.code]
        assert len(errors) >= 1
        assert not result.is_valid


# ============================================================================
# ORDER INDEX VALIDATION TESTS
# ============================================================================

class TestOrderIndexValidation:
    """Tests for order_index validation."""
    
    def test_validate_duplicate_order_indices(self, validator):
        """Test detection of duplicate order_index values."""
        doc_id = uuid4()
        
        elem1 = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=5,
                source_file="test.docx",
            ),
            content="First"
        )
        
        elem2 = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=5,
                source_file="test.docx",
            ),
            content="Second"
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem1, elem2]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "DUPLICATE_ORDER_INDEX"]
        assert len(errors) >= 1
        assert not result.is_valid
    
    def test_validate_negative_order_index(self, validator):
        """Test validation catches negative order_index."""
        doc_id = uuid4()
        
        elem = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=-1,
                source_file="test.docx",
            ),
            content="Test"
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "INVALID_ORDER_INDEX"]
        assert len(errors) == 1
        assert not result.is_valid


# ============================================================================
# SECTION HIERARCHY TESTS
# ============================================================================

class TestSectionHierarchy:
    """Tests for section hierarchy validation."""
    
    def test_validate_valid_hierarchy(self, validator):
        """Test validation of valid section hierarchy."""
        doc_id = uuid4()
        
        section1 = Section(
            heading="Section 1",
            id=uuid4(),
            section_level=1,
            elements=[
                Paragraph(
                    metadata=make_element_metadata(
                        document_id=doc_id,
                        order_index=0,
                        source_file="test.docx",
                        section_id=None,
                    ),
                    content="Text"
                )
            ]
        )
        
        section1_1 = Section(
            heading="Section 1.1",
            id=uuid4(),
            section_level=2,
            elements=[]
        )
        
        section1.subsections.append(section1_1)
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section1]
        )
        
        result = validator.validate(doc)
        
        hierarchy_errors = [e for e in result.errors if "HIERARCHY" in e.code]
        assert len(hierarchy_errors) == 0
    
    def test_validate_broken_hierarchy_subsection_not_deeper(self, validator):
        """Test detection of subsection not being deeper than parent."""
        doc_id = uuid4()
        
        section1 = Section(
            heading="Section 1",
            id=uuid4(),
            section_level=1
        )
        
        section1_broken = Section(
            heading="Broken Subsection",
            id=uuid4(),
            section_level=1,  # Same as parent!
        )
        
        section1.subsections.append(section1_broken)
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section1]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "BROKEN_SECTION_HIERARCHY"]
        assert len(errors) >= 1
        assert not result.is_valid
    
    def test_validate_invalid_section_level(self, validator):
        """Test detection of invalid section level."""
        doc_id = uuid4()
        
        section = Section(
            heading="Invalid Section",
            id=uuid4(),
            section_level=7  # Invalid! Max is 6
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "INVALID_SECTION_LEVEL"]
        assert len(errors) == 1
        assert not result.is_valid
    
    def test_validate_top_level_section_not_level_1(self, validator):
        """Test warning for top-level section that isn't level 1."""
        doc_id = uuid4()
        
        section = Section(
            heading="Top Level But Not Level 1",
            id=uuid4(),
            section_level=2  # Should be 1!
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        warnings = [w for w in result.warnings if "TOP_LEVEL_SECTION" in w.code]
        assert len(warnings) >= 1


# ============================================================================
# PARENT REFERENCE TESTS
# ============================================================================

class TestParentReferences:
    """Tests for parent reference validation."""
    
    def test_validate_mismatched_document_id(self, validator):
        """Test detection of element with wrong document_id."""
        doc_id = uuid4()
        wrong_doc_id = uuid4()
        
        elem = Paragraph(
            metadata=make_element_metadata(
                document_id=wrong_doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            content="Test"
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "MISMATCHED_DOCUMENT_ID"]
        assert len(errors) == 1
        assert not result.is_valid
    
    def test_validate_missing_section_id_in_section(self, validator):
        """Test detection of element in section without section_id."""
        doc_id = uuid4()
        section_id = uuid4()
        
        section = Section(
            heading="Test Section",
            id=section_id,
            section_level=1,
            elements=[
                Paragraph(
                    metadata=make_element_metadata(
                        document_id=doc_id,
                        order_index=0,
                        source_file="test.docx",
                        section_id=None,
                    ),
                    content="Test"
                )
            ]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if "SECTION_ID" in e.code]
        assert len(errors) >= 1
        assert not result.is_valid
    
    def test_validate_mismatched_section_id(self, validator):
        """Test detection of element with wrong section_id."""
        doc_id = uuid4()
        section_id = uuid4()
        wrong_section_id = uuid4()
        
        section = Section(
            heading="Test Section",
            id=section_id,
            section_level=1,
            elements=[
                Paragraph(
                    metadata=make_element_metadata(
                        document_id=doc_id,
                        order_index=0,
                        source_file="test.docx",
                        section_id=wrong_section_id,
                    ),
                    content="Test"
                )
            ]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "MISMATCHED_SECTION_ID"]
        assert len(errors) == 1
        assert not result.is_valid


# ============================================================================
# ELEMENT-SPECIFIC VALIDATION TESTS
# ============================================================================

class TestElementValidation:
    """Tests for element-specific validation."""
    
    def test_validate_empty_paragraph_warning(self, validator):
        """Test warning for empty paragraph."""
        doc_id = uuid4()
        
        elem = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            content="   "  # Whitespace only
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        warnings = [w for w in result.warnings if w.code == "EMPTY_PARAGRAPH"]
        assert len(warnings) >= 1
    
    def test_validate_empty_table_error(self, validator):
        """Test error for table with no rows."""
        doc_id = uuid4()
        
        elem = Table(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            rows=[]  # Empty!
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "EMPTY_TABLE"]
        assert len(errors) >= 1
        assert not result.is_valid
    
    def test_validate_table_inconsistent_columns(self, validator):
        """Test warning for table with inconsistent column counts."""
        doc_id = uuid4()
        
        elem = Table(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            rows=[
                TableRow(
                    row_index=0,
                    cells=[TableCell(content="A1"), TableCell(content="A2")]
                ),
                TableRow(
                    row_index=1,
                    cells=[TableCell(content="B1")]  # Wrong count!
                )
            ]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        warnings = [w for w in result.warnings if "COLUMN" in w.code]
        assert len(warnings) >= 1
    
    def test_validate_image_missing_alt_text(self, validator):
        """Test warning for image with missing alt_text."""
        doc_id = uuid4()
        
        elem = Image(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            alt_text=""  # Missing!
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        warnings = [w for w in result.warnings if w.code == "MISSING_ALT_TEXT"]
        assert len(warnings) >= 1
    
    def test_validate_empty_list_error(self, validator):
        """Test error for list with no items."""
        doc_id = uuid4()
        
        elem = ListElement(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                source_file="test.docx",
            ),
            items=[]  # Empty!
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "EMPTY_LIST"]
        assert len(errors) >= 1
        assert not result.is_valid


# ============================================================================
# SECTION CONTENT TESTS
# ============================================================================

class TestSectionContent:
    """Tests for section-specific validation."""
    
    def test_validate_empty_section_warning(self, validator):
        """Test warning for section with no content."""
        doc_id = uuid4()
        
        section = Section(
            heading="Empty Section",
            id=uuid4(),
            section_level=1,
            elements=[],  # No elements
            subsections=[]  # No subsections
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(warnings) >= 1
    
    def test_validate_empty_section_heading_error(self, validator):
        """Test error for section with empty heading."""
        doc_id = uuid4()
        
        section = Section(
            heading="",  # Empty heading!
            id=uuid4(),
            section_level=1
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        errors = [e for e in result.errors if e.code == "EMPTY_SECTION_HEADING"]
        assert len(errors) >= 1
        assert not result.is_valid


# ============================================================================
# STATISTICS TESTS
# ============================================================================

class TestStatistics:
    """Tests for statistics computation."""
    
    def test_compute_statistics(self, validator):
        """Test that statistics are computed correctly."""
        doc_id = uuid4()
        sect_id = uuid4()
        
        elem1 = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=0,
                section_id=sect_id,
                source_file="test.docx",
            ),
            content="Text"
        )
        
        elem2 = Table(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=1,
                section_id=sect_id,
                source_file="test.docx",
            ),
            rows=[TableRow(cells=[TableCell(content="A")])]
        )
        
        section = Section(
            heading="Test",
            id=sect_id,
            section_level=1,
            elements=[elem1, elem2]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        assert result.statistics.total_elements == 2
        assert result.statistics.total_sections == 1
        assert result.statistics.element_breakdown["paragraph"] == 1
        assert result.statistics.element_breakdown["table"] == 1
        assert result.statistics.unique_uuids == 3  # 2 elements + 1 section
        assert result.statistics.total_errors == 0
        assert result.statistics.total_warnings >= 0
    
    def test_statistics_error_and_warning_counts(self, validator):
        """Test that error and warning counts are correct."""
        doc_id = uuid4()
        
        # Create document with error and warning
        elem = Paragraph(
            metadata=make_element_metadata(
                document_id=doc_id,
                order_index=-1,
                section_id=None,
                source_file="",
            ),
            content="   "  # Warning: empty
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/test.docx"),
            document_id=doc_id,
            elements=[elem]
        )
        
        result = validator.validate(doc)
        
        assert result.statistics.total_errors >= 1
        assert result.statistics.total_warnings >= 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestValidationIntegration:
    """Integration tests with complete documents."""
    
    def test_validate_complex_document_valid(self, validator):
        """Test validation of a complex but valid document."""
        doc_id = uuid4()
        sect1_id = uuid4()
        sect1_1_id = uuid4()
        sect2_id = uuid4()
        
        section1_1 = Section(
            heading="Subsection",
            id=sect1_1_id,
            section_level=2,
            elements=[
                Paragraph(
                    metadata=ElementMetadata(
                        id=uuid4(),
                        document_id=doc_id,
                        order_index=2,
                        section_id=sect1_1_id,
                        parent_element_id=None,
                        page_number=None,
                        source_file="test.docx",
                        source_path=None,
                        source_document=None,
                    ),
                    content="Subsection content"
                )
            ]
        )
        
        section1 = Section(
            heading="Section 1",
            id=sect1_id,
            section_level=1,
            elements=[
                Paragraph(
                    metadata=ElementMetadata(
                        id=uuid4(),
                        document_id=doc_id,
                        order_index=1,
                        section_id=sect1_id,
                        parent_element_id=None,
                        page_number=None,
                        source_file="test.docx",
                        source_path=None,
                        source_document=None,
                    ),
                    content="Section 1 content"
                )
            ],
            subsections=[section1_1]
        )
        
        section2 = Section(
            heading="Section 2",
            id=sect2_id,
            section_level=1,
            elements=[
                Table(
                    metadata=ElementMetadata(
                        id=uuid4(),
                        document_id=doc_id,
                        order_index=3,
                        section_id=sect2_id,
                        parent_element_id=None,
                        page_number=None,
                        source_file="test.docx",
                        source_path=None,
                        source_document=None,
                    ),
                    rows=[TableRow(cells=[TableCell(content="Data")])]
                )
            ]
        )
        
        doc = Document(
            title="Complete Test Document",
            metadata=DocumentMetadata(
                source="/path/to/test.docx",
                author="Test Author",
                language="en"
            ),
            document_id=doc_id,
            elements=[
                Paragraph(
                    metadata=ElementMetadata(
                        id=uuid4(),
                        document_id=doc_id,
                        order_index=0,
                        section_id=None,
                        parent_element_id=None,
                        page_number=None,
                        source_file="test.docx",
                        source_path=None,
                        source_document=None,
                    ),
                    content="Root level content"
                )
            ],
            sections=[section1, section2]
        )
        
        result = validator.validate(doc)
        
        assert result.is_valid
        assert len(result.errors) == 0
        # Warnings allowed
        assert result.statistics.total_elements == 4
        assert result.statistics.total_sections == 3


# ============================================================================
# EMPTY SECTION AND STRUCTURAL CONTAINER TESTS
# ============================================================================

class TestStructuralContainerDetection:
    """Tests for identifying structural container sections."""
    
    def test_table_of_contents_is_container(self):
        """Test that 'Table of Contents' is recognized as container."""
        assert is_structural_container_section("Table of Contents") is True
    
    def test_contents_is_container(self):
        """Test that 'Contents' is recognized as container."""
        assert is_structural_container_section("Contents") is True
    
    def test_appendix_is_container(self):
        """Test that 'Appendix' is recognized as container."""
        assert is_structural_container_section("Appendix") is True
    
    def test_appendix_with_colon_is_container(self):
        """Test that 'Appendix:' is recognized as container."""
        assert is_structural_container_section("Appendix:") is True
    
    def test_appendix_1_is_container(self):
        """Test that 'Appendix 1' is recognized as container."""
        assert is_structural_container_section("Appendix 1") is True
    
    def test_appendices_is_container(self):
        """Test that 'Appendices' is recognized as container."""
        assert is_structural_container_section("Appendices") is True
    
    def test_annexure_is_container(self):
        """Test that 'Annexure' is recognized as container."""
        assert is_structural_container_section("Annexure") is True
    
    def test_annexures_is_container(self):
        """Test that 'Annexures' is recognized as container."""
        assert is_structural_container_section("Annexures") is True
    
    def test_section_d_is_container(self):
        """Test that 'Section D' is recognized as container."""
        assert is_structural_container_section("Section D") is True
    
    def test_section_appendix_is_container(self):
        """Test that 'Section: Appendix Details' is recognized as container."""
        assert is_structural_container_section("Section: Appendix Details") is True
    
    def test_chapter_is_container(self):
        """Test that 'Chapter' is recognized as container."""
        assert is_structural_container_section("Chapter") is True
    
    def test_chapter_1_is_container(self):
        """Test that 'Chapter 1' is recognized as container."""
        assert is_structural_container_section("Chapter 1") is True
    
    def test_part_is_container(self):
        """Test that 'Part' is recognized as container."""
        assert is_structural_container_section("Part") is True
    
    def test_form_is_container(self):
        """Test that 'Form' is recognized as container."""
        assert is_structural_container_section("Form") is True
    
    def test_form_1_is_container(self):
        """Test that 'Form 1' is recognized as container."""
        assert is_structural_container_section("Form 1") is True
    
    def test_form_5a_is_container(self):
        """Test that 'Form 5A' is recognized as container."""
        assert is_structural_container_section("Form 5A") is True
    
    def test_form_5b_is_container(self):
        """Test that 'Form 5B' is recognized as container."""
        assert is_structural_container_section("Form 5B") is True
    
    def test_form_6a_is_container(self):
        """Test that 'Form 6A' is recognized as container."""
        assert is_structural_container_section("Form 6A") is True
    
    def test_form_7f_is_container(self):
        """Test that 'Form 7F' is recognized as container."""
        assert is_structural_container_section("Form 7F") is True
    
    def test_form_colon_title_is_container(self):
        """Test that 'Form: Financial Details' is recognized as container."""
        assert is_structural_container_section("Form: Financial Details") is True
    
    def test_case_insensitive_container_detection(self):
        """Test that container detection is case-insensitive."""
        assert is_structural_container_section("TABLE OF CONTENTS") is True
        assert is_structural_container_section("Appendix") is True
        assert is_structural_container_section("FORM 5A") is True
    
    def test_intro_is_not_container(self):
        """Test that 'Introduction' is not a container."""
        assert is_structural_container_section("Introduction") is False
    
    def test_conclusion_is_not_container(self):
        """Test that 'Conclusion' is not a container."""
        assert is_structural_container_section("Conclusion") is False
    
    def test_empty_heading_is_not_container(self):
        """Test that empty heading is not a container."""
        assert is_structural_container_section("") is False
    
    def test_none_heading_is_not_container(self):
        """Test that None heading is not a container."""
        assert is_structural_container_section(None) is False
    
    def test_whitespace_only_heading_is_not_container(self):
        """Test that whitespace-only heading is not a container."""
        assert is_structural_container_section("   ") is False


class TestEmptySectionValidation:
    """Tests for EMPTY_SECTION warning validation."""
    
    def test_empty_genuine_section_produces_warning(self, validator):
        """Test that a genuine empty body section produces EMPTY_SECTION warning."""
        doc_id = uuid4()
        section = Section(
            heading="Empty Content Section",
            section_level=1,
            elements=[],
            subsections=[]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(empty_warnings) == 1
        assert "Empty Content Section" in empty_warnings[0].message
    
    def test_empty_section_with_structure_does_not_warn(self, validator):
        """Test that an empty section with subsections does not produce EMPTY_SECTION warning."""
        doc_id = uuid4()
        subsection = Section(
            heading="Subsection",
            section_level=2,
            elements=[],
            subsections=[]
        )
        
        section = Section(
            heading="Parent Section",
            section_level=1,
            elements=[],
            subsections=[subsection]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_section_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION" and "Parent Section" in w.message]
        assert len(empty_section_warnings) == 0
    
    def test_empty_section_with_elements_does_not_warn(self, validator):
        """Test that a section with elements does not produce EMPTY_SECTION warning."""
        doc_id = uuid4()
        section = Section(
            heading="Section with Content",
            section_level=1,
            elements=[
                Paragraph(
                    metadata=make_element_metadata(doc_id, 0),
                    content="Some content"
                )
            ]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_section_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(empty_section_warnings) == 0
    
    def test_container_section_does_not_warn_when_empty(self, validator):
        """Test that an empty container section does NOT produce EMPTY_SECTION warning."""
        doc_id = uuid4()
        section = Section(
            heading="Appendix A",
            section_level=1,
            elements=[],
            subsections=[]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_section_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(empty_section_warnings) == 0
    
    def test_toc_container_does_not_warn_when_empty(self, validator):
        """Test that empty Table of Contents does NOT produce EMPTY_SECTION warning."""
        doc_id = uuid4()
        section = Section(
            heading="Table of Contents",
            section_level=1,
            elements=[],
            subsections=[]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_section_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(empty_section_warnings) == 0
    
    def test_form_container_does_not_warn_when_empty(self, validator):
        """Test that empty Form section does NOT produce EMPTY_SECTION warning."""
        doc_id = uuid4()
        section = Section(
            heading="Form 5A: Financial Details",
            section_level=1,
            elements=[],
            subsections=[]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        empty_section_warnings = [w for w in result.warnings if w.code == "EMPTY_SECTION"]
        assert len(empty_section_warnings) == 0
    
    def test_empty_section_heading_still_produces_error(self, validator):
        """Test that EMPTY_SECTION_HEADING error is still produced for blank headings."""
        doc_id = uuid4()
        section = Section(
            heading="",
            section_level=1,
            elements=[],
            subsections=[]
        )
        
        doc = Document(
            title="Test",
            metadata=DocumentMetadata(source="/path/to/test.docx"),
            document_id=doc_id,
            sections=[section]
        )
        
        result = validator.validate(doc)
        
        heading_errors = [e for e in result.errors if e.code == "EMPTY_SECTION_HEADING"]
        assert len(heading_errors) == 1
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
