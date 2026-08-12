"""Document loader for the Defence Proposal AI Agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import fitz
from docx import Document as load_docx
from docx.document import Document as DocxDocument

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_RAW_DOCUMENT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".docx", ".pdf"})
LoadedDocument = Union[DocxDocument, fitz.Document]


class DocumentLoaderError(Exception):
    """Base exception for document loader failures."""


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when a required document or directory cannot be found."""


class DocumentLoadError(DocumentLoaderError):
    """Raised when a DOCX file cannot be loaded."""


def discover_docx_files(raw_directory: Path = DEFAULT_RAW_DOCUMENT_DIR) -> List[Path]:
    """Discover all .docx documents in the provided raw directory."""
    raw_directory = raw_directory.expanduser().resolve()
    logger.debug("Discovering .docx files in %s", raw_directory)

    if not raw_directory.exists():
        message = f"Raw document directory does not exist: {raw_directory}"
        logger.error(message)
        raise DocumentNotFoundError(message)

    if not raw_directory.is_dir():
        message = f"Expected a directory at: {raw_directory}"
        logger.error(message)
        raise DocumentLoaderError(message)

    docx_files = sorted(raw_directory.rglob("*.docx"))
    logger.info("Found %d .docx files in %s", len(docx_files), raw_directory)

    if not docx_files:
        message = f"No .docx files found in directory: {raw_directory}"
        logger.warning(message)
        raise DocumentNotFoundError(message)

    return docx_files


def discover_document_files(raw_directory: Path = DEFAULT_RAW_DOCUMENT_DIR) -> List[Path]:
    """Discover supported DOCX and PDF documents in the provided raw directory."""
    raw_directory = raw_directory.expanduser().resolve()
    logger.debug("Discovering supported documents in %s", raw_directory)

    if not raw_directory.exists():
        message = f"Raw document directory does not exist: {raw_directory}"
        logger.error(message)
        raise DocumentNotFoundError(message)

    if not raw_directory.is_dir():
        message = f"Expected a directory at: {raw_directory}"
        logger.error(message)
        raise DocumentLoaderError(message)

    document_files = sorted(
        path for path in raw_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )
    logger.info("Found %d supported documents in %s", len(document_files), raw_directory)

    if not document_files:
        message = f"No supported documents found in directory: {raw_directory}"
        logger.warning(message)
        raise DocumentNotFoundError(message)

    return document_files


def load_docx_document(docx_path: Path) -> DocxDocument:
    """Load a single .docx file and return a python-docx Document object."""
    docx_path = docx_path.expanduser().resolve()
    logger.debug("Loading DOCX file from %s", docx_path)

    if not docx_path.exists():
        message = f"DOCX file does not exist: {docx_path}"
        logger.error(message)
        raise DocumentNotFoundError(message)

    if not docx_path.is_file() or docx_path.suffix.lower() != ".docx":
        message = f"Unsupported file type or path is not a file: {docx_path}"
        logger.error(message)
        raise DocumentLoaderError(message)

    try:
        document = load_docx(docx_path)
    except Exception as error:
        message = f"Failed to load DOCX file {docx_path}: {error}"
        logger.exception(message)
        raise DocumentLoadError(message) from error

    logger.info("Successfully loaded document: %s", docx_path)
    return document


def load_pdf_document(pdf_path: Path) -> fitz.Document:
    """Load a text-based PDF and return its PyMuPDF document object."""
    pdf_path = pdf_path.expanduser().resolve()
    logger.debug("Loading PDF file from %s", pdf_path)

    if not pdf_path.exists():
        message = f"PDF file does not exist: {pdf_path}"
        logger.error(message)
        raise DocumentNotFoundError(message)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        message = f"Unsupported file type or path is not a file: {pdf_path}"
        logger.error(message)
        raise DocumentLoaderError(message)

    try:
        document = fitz.open(pdf_path)
    except Exception as error:
        message = f"Failed to load PDF file {pdf_path}: {error}"
        logger.exception(message)
        raise DocumentLoadError(message) from error

    logger.info("Successfully loaded PDF: %s", pdf_path)
    return document


def load_document(document_path: Path) -> LoadedDocument:
    """Load a supported document according to its file extension."""
    suffix = document_path.suffix.lower()
    if suffix == ".docx":
        return load_docx_document(document_path)
    if suffix == ".pdf":
        return load_pdf_document(document_path)

    message = f"Unsupported document type: {document_path}"
    logger.error(message)
    raise DocumentLoaderError(message)


def load_documents_from_raw(raw_directory: Path = DEFAULT_RAW_DOCUMENT_DIR) -> List[LoadedDocument]:
    """Load all supported documents found in the raw directory."""
    document_paths = discover_document_files(raw_directory)
    documents: List[LoadedDocument] = []

    for document_path in document_paths:
        documents.append(load_document(document_path))

    logger.info("Loaded %d document(s) from %s", len(documents), raw_directory)
    return documents


__all__ = [
    "DocumentLoaderError",
    "DocumentLoadError",
    "DocumentNotFoundError",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "discover_document_files",
    "discover_docx_files",
    "load_document",
    "load_docx_document",
    "load_pdf_document",
    "load_documents_from_raw",
]
