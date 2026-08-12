#!/usr/bin/env python3
"""Index a proposal corpus through the production IndexPipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proposal_ai_agent.indexing import DocumentRole, IndexPipeline, IndexRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index PDF and DOCX proposals into Qdrant.")
    parser.add_argument("--input", required=True, type=Path, help="Document file or directory")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument(
        "--document-role",
        required=True,
        choices=tuple(role.value for role in DocumentRole),
        help="Application-owned role; proposal-reference indexing accepts only REFERENCE_KNOWLEDGE",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding and upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the target collection")
    parser.add_argument("--verbose", action="store_true", help="Show each document result")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        result = IndexPipeline.from_environment(args.collection).index(
            IndexRequest(
                input_path=args.input,
                collection_name=args.collection,
                document_role=DocumentRole(args.document_role),
                batch_size=args.batch_size,
                recreate=args.recreate,
            )
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    status = "created" if result.collection_created else "reused"
    print(f"Collection {status}: {result.collection_name}")
    stats = result.statistics
    print(f"Documents indexed: {stats.documents_indexed}")
    print(f"Skipped documents: {stats.documents_skipped}")
    print(f"Chunks indexed: {stats.chunks_indexed}")
    print(f"Embeddings generated: {stats.embeddings_generated}")
    print(f"Vectors uploaded: {stats.vectors_uploaded}")
    print(f"Failures: {stats.documents_failed}")
    print(f"Elapsed time: {stats.elapsed_ms:.1f} ms")
    if args.verbose:
        for document in result.documents:
            state = "skipped" if document.skipped else f"indexed ({len(document.chunks)} chunks)"
            print(f"{state}: {document.source_path}")
    for failure in result.failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
