"""Manual smoke script for the document loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

from docx import Document as DocxDocument
from proposal_ai_agent.ingestion.loader import (
    discover_docx_files,
    load_docx_document,
    load_documents_from_raw,
)


ROOT = Path(__file__).resolve().parents[1]


def create_sample_document(path: Path) -> None:
    document = DocxDocument()
    document.add_paragraph("This is a test document for the loader.")
    document.save(path)


def main() -> None:
    print("Document loader test started")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sample_docx = temp_path / "sample.docx"
        create_sample_document(sample_docx)

        discovered = discover_docx_files(temp_path)
        print(f"Discovered {len(discovered)} .docx file(s): {[p.name for p in discovered]}")

        loaded_document = load_docx_document(discovered[0])
        print(f"Loaded single document object: {type(loaded_document).__name__}")

    raw_directory = ROOT / "data" / "raw"
    loaded_documents = load_documents_from_raw(raw_directory)
    print(f"Loaded {len(loaded_documents)} document(s) from {raw_directory}")

    print("Document loader test completed successfully")


if __name__ == "__main__":
    main()
