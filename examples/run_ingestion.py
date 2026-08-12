#!/usr/bin/env python3
"""
Complete ingestion pipeline runner for the Defence Proposal AI Agent.

Runs the full pipeline on the real proposal corpus:
  Loader → Parser → Validator → Semantic Chunker

Produces a human-readable validation report with per-document and corpus-wide statistics.
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from statistics import mean, stdev

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proposal_ai_agent.ingestion.loader import (
    discover_document_files,
    load_document,
)
from proposal_ai_agent.ingestion.parser import DocumentParser
from proposal_ai_agent.ingestion.pdf_parser import parse_pdf_document
from proposal_ai_agent.ingestion.validator import DocumentValidator, ValidationResult
from proposal_ai_agent.ingestion.validator import classify_section, SectionType
from proposal_ai_agent.ingestion.chunker import chunk_document, ChunkConfig
from proposal_ai_agent.ingestion.models import Document, Paragraph, Table

# Configure logging to suppress debug output
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# CORPUS STATISTICS TRACKING
# ============================================================================


class CorpusStatistics:
    """Tracks aggregate statistics across all processed documents."""
    
    def __init__(self):
        self.total_documents = 0
        self.total_sections = 0
        self.total_paragraphs = 0
        self.total_tables = 0
        self.total_chunks = 0
        self.total_warnings = 0
        self.total_errors = 0
        self.validation_rules_applied = 0
        self.chunk_sizes: List[int] = []
        
    def add_document_stats(self, doc: Document, validation_result: ValidationResult, chunks: List):
        """Aggregate statistics from a single document."""
        self.total_documents += 1
        
        # Count sections
        self._count_sections(doc)
        
        # Count elements
        self._count_elements(doc)
        
        # Count chunks and sizes
        self.total_chunks += len(chunks)
        if chunks:
            for chunk in chunks:
                self.chunk_sizes.append(chunk.token_count)
        
        # Count validation results
        self.total_warnings += validation_result.statistics.total_warnings
        self.total_errors += validation_result.statistics.total_errors
        
    def _count_sections(self, doc: Document, max_depth: int = 10):
        """Recursively count all sections in the document."""
        def count_sections_recursive(section_list, depth=0):
            if depth > max_depth:
                return 0
            count = len(section_list)
            for section in section_list:
                count += count_sections_recursive(section.subsections, depth + 1)
            return count
        
        self.total_sections += count_sections_recursive(doc.sections)
    
    def _count_elements(self, doc: Document):
        """Count all paragraphs and tables in the document."""
        def count_in_section(section_list):
            para_count = 0
            table_count = 0
            for section in section_list:
                # Count elements in this section
                for element in section.elements:
                    if isinstance(element, Paragraph):
                        para_count += 1
                    elif isinstance(element, Table):
                        table_count += 1
                # Recursively count in subsections
                sub_para, sub_table = count_in_section(section.subsections)
                para_count += sub_para
                table_count += sub_table
            return para_count, table_count
        
        # Count in sections
        sec_para, sec_table = count_in_section(doc.sections)
        self.total_paragraphs += sec_para
        self.total_tables += sec_table
        
        # Count at root level
        for element in doc.elements:
            if isinstance(element, Paragraph):
                self.total_paragraphs += 1
            elif isinstance(element, Table):
                self.total_tables += 1
    
    def get_average_chunk_size(self) -> float:
        """Calculate average chunk size in tokens."""
        if not self.chunk_sizes:
            return 0.0
        return mean(self.chunk_sizes)
    
    def get_chunk_size_stdev(self) -> Optional[float]:
        """Calculate chunk size standard deviation."""
        if len(self.chunk_sizes) < 2:
            return None
        return stdev(self.chunk_sizes)


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================


def process_document(
    document_path: Path,
    parser: DocumentParser,
    validator: DocumentValidator,
    chunk_config: ChunkConfig,
) -> Tuple[Optional[Document], Optional[ValidationResult], Optional[List]]:
    """
    Process a single document through the complete pipeline.
    
    Returns:
        (document, validation_result, chunks) or (None, None, None) on error
    """
    try:
        # Determine document type
        doc_type = "DOCX" if document_path.suffix.lower() == ".docx" else "PDF"
        
        # Load document
        if doc_type == "DOCX":
            loaded_doc = load_document(document_path)
            parsed_doc = parser.parse(
                loaded_doc,
                document_path,
                source_document=document_path.stem,
                document_title=document_path.stem,
            )
        else:  # PDF
            parsed_doc = parse_pdf_document(
                document_path,
                source_document=document_path.stem,
                document_title=document_path.stem,
            )
        
        # Validate document
        validation_result = validator.validate(parsed_doc)
        
        # Chunk document
        chunks = chunk_document(parsed_doc, chunk_config)
        
        return parsed_doc, validation_result, chunks
    
    except Exception as e:
        print(f"ERROR processing {document_path.name}: {e}")
        return None, None, None


def count_sections_in_doc(doc: Document) -> int:
    """Count all sections (including subsections) in a document."""
    def count_recursive(section_list):
        count = len(section_list)
        for section in section_list:
            count += count_recursive(section.subsections)
        return count
    
    return count_recursive(doc.sections)


def format_document_report(
    document_path: Path,
    document: Document,
    validation_result: ValidationResult,
    chunks: List,
) -> str:
    """Format a single document report."""
    lines = []
    lines.append("-" * 48)
    lines.append(f"Document:")
    lines.append(f"{document_path.name}")
    lines.append("")
    
    # Type
    doc_type = "DOCX" if document_path.suffix.lower() == ".docx" else "PDF"
    lines.append(f"Type:")
    lines.append(doc_type)
    lines.append("")
    
    # Statistics
    section_count = count_sections_in_doc(document)
    
    # Count paragraphs and tables
    def count_elements(section_list):
        para_count = 0
        table_count = 0
        for section in section_list:
            for element in section.elements:
                if isinstance(element, Paragraph):
                    para_count += 1
                elif isinstance(element, Table):
                    table_count += 1
            sub_para, sub_table = count_elements(section.subsections)
            para_count += sub_para
            table_count += sub_table
        return para_count, table_count
    
    sec_para, sec_table = count_elements(document.sections)
    root_para = sum(1 for e in document.elements if isinstance(e, Paragraph))
    root_table = sum(1 for e in document.elements if isinstance(e, Table))
    
    total_paragraphs = sec_para + root_para
    total_tables = sec_table + root_table
    
    lines.append(f"Sections:")
    lines.append(f"{section_count}")
    lines.append("")
    
    lines.append(f"Paragraphs:")
    lines.append(f"{total_paragraphs}")
    lines.append("")
    
    lines.append(f"Tables:")
    lines.append(f"{total_tables}")
    lines.append("")
    
    # Validation status
    if validation_result.is_valid and validation_result.statistics.total_warnings == 0:
        validation_status = "PASS"
    elif validation_result.is_valid:
        validation_status = "WARNINGS"
    else:
        validation_status = "FAIL"
    
    lines.append(f"Validation:")
    lines.append(validation_status)
    lines.append("")
    
    # Detailed validation issues
    if validation_result.errors:
        lines.append(f"Errors ({len(validation_result.errors)}):")
        for error in validation_result.errors:
            lines.append(f"  - {error.code}: {error.message}")
        lines.append("")
    
    if validation_result.warnings:
        lines.append(f"Warnings ({len(validation_result.warnings)}):")
        for warning in validation_result.warnings:
            lines.append(f"  - {warning.code}: {warning.message}")
        lines.append("")
    
    # Chunks
    lines.append(f"Chunks:")
    lines.append(f"{len(chunks)}")
    lines.append("")
    
    # Average chunk size
    if chunks:
        avg_chunk_size = mean([chunk.token_count for chunk in chunks])
        lines.append(f"Average chunk size:")
        lines.append(f"{avg_chunk_size:.1f} tokens")
    else:
        lines.append(f"Average chunk size:")
        lines.append(f"N/A (no chunks)")
    
    lines.append("-" * 48)
    lines.append("")
    # Classify sections and tally EXPECTED vs WARNING
    expected = []
    warnings_list = []
    for idx, section in enumerate(document.sections):
        stype = classify_section(section, is_top_level_first=(idx == 0))
        if stype in (SectionType.COVER_PAGE, SectionType.TABLE_OF_CONTENTS):
            # never report
            continue
        if stype in (SectionType.TEMPLATE_FORM, SectionType.PLACEHOLDER_SECTION, SectionType.CERTIFICATE, SectionType.SIGNATURE_PAGE):
            # expected empty
            if not section.elements and not section.subsections:
                expected.append((section.heading, stype))
        elif stype == SectionType.CONTAINER_SECTION:
            # valid if empty but has children
            if not section.elements and section.subsections:
                # valid container; no warning
                pass
            elif not section.elements and not section.subsections:
                warnings_list.append((section.heading, stype))
        else:
            # CONTENT_SECTION
            if not section.elements and not section.subsections:
                warnings_list.append((section.heading, stype))

    lines.append(f"Validation:")
    lines.append(validation_status)
    lines.append("")

    # Report EXPECTED vs WARNING
    if expected:
        lines.append(f"Expected empty sections ({len(expected)}):")
        for h, t in expected:
            lines.append(f"  - {t}: {h}")
        lines.append("")

    if warnings_list:
        lines.append(f"Warnings ({len(warnings_list)}):")
        for h, t in warnings_list:
            lines.append(f"  - {t}: {h}")
        lines.append("")

    # Include any validation_result.warnings that are not EMPTY_SECTION for completeness
    other_warnings = [w for w in validation_result.warnings if w.code != "EMPTY_SECTION"]
    if other_warnings:
        lines.append(f"Other warnings ({len(other_warnings)}):")
        for w in other_warnings:
            lines.append(f"  - {w.code}: {w.message}")
        lines.append("")

    return "\n".join(lines)


def format_corpus_summary(stats: CorpusStatistics) -> str:
    """Format the final corpus-wide summary."""
    lines = []
    lines.append("\n" + "=" * 48)
    lines.append("CORPUS SUMMARY")
    lines.append("=" * 48)
    lines.append("")
    
    lines.append(f"Documents: {stats.total_documents}")
    lines.append(f"Sections: {stats.total_sections}")
    lines.append(f"Paragraphs: {stats.total_paragraphs}")
    lines.append(f"Tables: {stats.total_tables}")
    lines.append(f"Chunks: {stats.total_chunks}")
    lines.append(f"Warnings: {stats.total_warnings}")
    lines.append(f"Errors: {stats.total_errors}")
    
    if stats.chunk_sizes:
        avg_size = stats.get_average_chunk_size()
        lines.append(f"Average chunk size: {avg_size:.1f} tokens")
        stdev_size = stats.get_chunk_size_stdev()
        if stdev_size is not None:
            lines.append(f"Chunk size stdev: {stdev_size:.1f} tokens")
    
    lines.append("=" * 48)
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Run the complete ingestion pipeline on the real proposal corpus."""
    # Initialize components
    parser = DocumentParser()
    validator = DocumentValidator()
    chunk_config = ChunkConfig(chunk_size=500, overlap_ratio=0.1)
    corpus_stats = CorpusStatistics()
    
    # Discover documents
    raw_directory = Path(__file__).resolve().parents[1] / "data" / "raw" / "proposals"
    
    if not raw_directory.exists():
        print(f"ERROR: Directory not found: {raw_directory}")
        return 1
    
    try:
        document_paths = discover_document_files(raw_directory)
    except Exception as e:
        print(f"ERROR discovering documents: {e}")
        return 1
    
    if not document_paths:
        print(f"WARNING: No documents found in {raw_directory}")
        return 0
    
    print(f"Found {len(document_paths)} document(s) to process:")
    for path in document_paths:
        print(f"  - {path.relative_to(raw_directory)}")
    print()
    
    # Process each document
    for document_path in document_paths:
        doc, val_result, chunks = process_document(
            document_path, parser, validator, chunk_config
        )
        
        if doc is None:
            print(f"SKIPPED: {document_path.name} (error during processing)")
            print()
            continue
        
        # Print document report
        report = format_document_report(document_path, doc, val_result, chunks)
        print(report)
        
        # Aggregate statistics
        corpus_stats.add_document_stats(doc, val_result, chunks)
    
    # Print corpus summary
    summary = format_corpus_summary(corpus_stats)
    print(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
