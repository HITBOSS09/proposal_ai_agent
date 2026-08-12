"""Production orchestration from source documents to Qdrant vectors."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from proposal_ai_agent.embeddings import EmbeddingEngine, EmbeddingEngineConfig, MemoryCache
from proposal_ai_agent.embeddings.providers.base import EmbeddingProvider
from proposal_ai_agent.ingestion.chunker import ChunkConfig, chunk_document
from proposal_ai_agent.ingestion.loader import SUPPORTED_DOCUMENT_EXTENSIONS, discover_document_files, load_document
from proposal_ai_agent.ingestion.metadata import enrich_chunks
from proposal_ai_agent.ingestion.parser import DocumentParser
from proposal_ai_agent.ingestion.pdf_parser import PDFParser
from proposal_ai_agent.ingestion.validator import DocumentValidator
from proposal_ai_agent.online.providers.embedding import OllamaEmbeddingProvider

from .builder import IndexBuilder
from .collection_manager import CollectionManager
from .config import IndexingConfig
from .exceptions import (
    DocumentRoleAuthorizationError,
    EmbeddingGenerationError,
    IndexPipelineError,
    VectorWriteError,
)
from .models import (
    DocumentRole,
    IndexPoint,
    IndexedChunk,
    IndexedDocument,
    IndexRequest,
    IndexResult,
    IndexStatistics,
)
from .writer import QdrantIndexWriter


class IndexPipeline:
    """Coordinate existing offline, embedding, and Qdrant components for a corpus."""

    _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
    _REFERENCE_KNOWLEDGE_ROOT = (
        _REPOSITORY_ROOT / "documents" / "Reference Document"
    ).resolve()

    def __init__(
        self,
        client: Any,
        embedding_provider: EmbeddingProvider,
        embedding_dimension: int,
        embedding_model: str,
        chunk_config: ChunkConfig | None = None,
        retry_attempts: int = 3,
    ) -> None:
        if retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        self._client = client
        self._embedding_provider = embedding_provider
        self._embedding_dimension = embedding_dimension
        self._embedding_model = embedding_model
        self._chunk_config = chunk_config or ChunkConfig()
        self._retry_attempts = retry_attempts

    @classmethod
    def from_environment(cls, collection_name: str) -> "IndexPipeline":
        """Construct the production runtime from the documented local-service settings."""
        qdrant_url = os.getenv("BDIL_QDRANT_URL", "http://localhost:6333")
        ollama_url = os.getenv("BDIL_OLLAMA_URL", "http://localhost:11434/api")
        embedding_model = os.getenv("BDIL_EMBEDDING_MODEL", "bge-m3")
        dimension = int(os.getenv("BDIL_EMBEDDING_DIMENSIONS", "1024"))
        timeout = int(os.getenv("BDIL_PROVIDER_TIMEOUT", "60"))
        return cls(
            client=QdrantClient(url=qdrant_url, timeout=timeout),
            embedding_provider=OllamaEmbeddingProvider(embedding_model, ollama_url, timeout),
            embedding_dimension=dimension,
            embedding_model=embedding_model,
        )

    def index(self, request: IndexRequest) -> IndexResult:
        """Index a directory or document, returning complete immutable telemetry."""
        if not isinstance(request, IndexRequest):
            raise TypeError("request must be an IndexRequest")
        started_at = perf_counter()
        paths = self._discover_paths(request.input_path)
        self._authorize_reference_paths(paths, request.document_role)
        manager = CollectionManager(
            self._client, request.collection_name, self._embedding_dimension, self._embedding_model
        )
        collection_created = manager.ensure_ready(recreate=request.recreate)
        writer = QdrantIndexWriter(
            self._client,
            IndexingConfig(
                collection_name=request.collection_name,
                vector_dimensions=self._embedding_dimension,
                distance="cosine",
                payload_indexes=(),
            ),
        )
        embedding_engine = EmbeddingEngine(
            EmbeddingEngineConfig(provider=self._embedding_provider, dimensions=self._embedding_dimension),
            MemoryCache(),
        )
        documents: list[IndexedDocument] = []
        failures: list[str] = []
        counters = {"indexed": 0, "skipped": 0, "chunks": 0, "embeddings": 0, "vectors": 0}

        for path in paths:
            fingerprint = self._fingerprint(path)
            if self._document_exists(request.collection_name, fingerprint):
                counters["skipped"] += 1
                documents.append(IndexedDocument(path, fingerprint, None, None, (), skipped=True))
                continue
            try:
                indexed, embedding_count, vector_count = self._index_document(
                    path,
                    fingerprint,
                    embedding_engine,
                    writer,
                    request.batch_size,
                    request.document_role,
                )
            except IndexPipelineError as error:
                failures.append(f"{path}: {error}")
                continue
            except Exception as error:
                failures.append(f"{path}: {error}")
                continue
            documents.append(indexed)
            counters["indexed"] += 1
            counters["chunks"] += len(indexed.chunks)
            counters["embeddings"] += embedding_count
            counters["vectors"] += vector_count

        statistics = IndexStatistics(
            documents_discovered=len(paths), documents_indexed=counters["indexed"],
            documents_skipped=counters["skipped"], documents_failed=len(failures),
            chunks_indexed=counters["chunks"], embeddings_generated=counters["embeddings"],
            vectors_uploaded=counters["vectors"], elapsed_ms=(perf_counter() - started_at) * 1000,
        )
        return IndexResult(request.collection_name, collection_created, tuple(documents), statistics, tuple(failures))

    def _index_document(
        self, path: Path, fingerprint: str, embedding_engine: EmbeddingEngine,
        writer: QdrantIndexWriter, batch_size: int, document_role: DocumentRole,
    ) -> tuple[IndexedDocument, int, int]:
        document = self._parse_document(path)
        validation = DocumentValidator().validate(document)
        if not validation.is_valid:
            raise IndexPipelineError("document validation failed: " + "; ".join(error.message for error in validation.errors))
        chunks = chunk_document(document, self._chunk_config)
        if not chunks:
            raise IndexPipelineError("document produced no indexable chunks")
        payloads = enrich_chunks(
            chunks,
            datetime.now(timezone.utc),
            document_role=document_role.value,
        )
        builder = IndexBuilder(self._embedding_dimension)
        points: list[IndexPoint] = []
        generated_embeddings = 0
        for payload_batch in self._batches(payloads, batch_size):
            try:
                embedded = embedding_engine.embed_batch(payload_batch)
            except Exception as error:
                raise EmbeddingGenerationError(f"embedding generation failed: {error}") from error
            generated_embeddings += sum(not result.cache_hit for result in embedded)
            for result in embedded:
                point = builder.build(result.payload, result.vector)
                points.append(self._with_runtime_payload(point, document.title, document.metadata.version, path, fingerprint))

        uploaded = 0
        for point_batch in self._batches(points, batch_size):
            uploaded += self._write_with_retry(writer, point_batch)
        indexed_chunks = tuple(
            IndexedChunk(chunk_id=payload.chunk_id, point_id=payload.point_uuid,
                         document_id=payload.document.document_id, token_count=payload.chunk.token_count)
            for payload in payloads
        )
        return (
            IndexedDocument(path, fingerprint, str(document.document_id), document.title, indexed_chunks),
            generated_embeddings,
            uploaded,
        )

    def _parse_document(self, path: Path) -> Any:
        loaded = load_document(path)
        if path.suffix.lower() == ".pdf":
            try:
                return PDFParser().parse(loaded, path, source_document=path.stem, document_title=path.stem)
            finally:
                loaded.close()
        return DocumentParser().parse(loaded, path, source_document=path.stem, document_title=path.stem)

    def _write_with_retry(self, writer: QdrantIndexWriter, points: tuple[IndexPoint, ...]) -> int:
        for attempt in range(self._retry_attempts):
            result = writer.upsert_batch(points)
            if result.failed == 0:
                return result.inserted + result.updated
            if attempt + 1 < self._retry_attempts:
                sleep(0.1 * (attempt + 1))
        raise VectorWriteError(f"failed to upload {len(points)} vectors after {self._retry_attempts} attempts")

    @staticmethod
    def _with_runtime_payload(point: IndexPoint, title: str, version: str | None, path: Path, fingerprint: str) -> IndexPoint:
        payload = dict(point.payload)
        payload.update({
            "document_title": title,
            "page_number": None,
            "chunk_id": payload["chunk_id"],
            "source_document": path.stem,
            "version": version,
            "document_fingerprint": fingerprint,
        })
        return IndexPoint(id=point.id, vector=point.vector, payload=payload)

    def _document_exists(self, collection_name: str, fingerprint: str) -> bool:
        records, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="document_fingerprint", match=MatchValue(value=fingerprint))]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    @staticmethod
    def _discover_paths(input_path: Path) -> list[Path]:
        path = Path(input_path).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                raise IndexPipelineError(f"unsupported document type: {path}")
            return [path]
        return [discovered.resolve() for discovered in discover_document_files(path)]

    @classmethod
    def _authorize_reference_paths(
        cls,
        paths: Iterable[Path],
        document_role: DocumentRole,
    ) -> None:
        """Authorize the complete run before any collection, embedding, or write I/O."""

        resolved_paths = tuple(Path(path).resolve() for path in paths)
        if document_role is not DocumentRole.REFERENCE_KNOWLEDGE:
            path_text = str(resolved_paths[0]) if resolved_paths else "<no discovered document>"
            role_text = getattr(document_role, "value", repr(document_role))
            raise DocumentRoleAuthorizationError(
                f"indexing refused: path={path_text}; declared_role={role_text}; "
                "expected_role=REFERENCE_KNOWLEDGE; operation=proposal-reference indexing"
            )
        for path in resolved_paths:
            if cls._REFERENCE_KNOWLEDGE_ROOT not in path.parents:
                raise DocumentRoleAuthorizationError(
                    f"indexing refused: path={path}; "
                    "declared_role=REFERENCE_KNOWLEDGE; "
                    f"authorized_root={cls._REFERENCE_KNOWLEDGE_ROOT}; "
                    "expected_role=REFERENCE_KNOWLEDGE; "
                    "operation=proposal-reference indexing"
                )

    @staticmethod
    def _fingerprint(path: Path) -> str:
        return sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _batches(items: Iterable[Any], batch_size: int) -> Iterable[tuple[Any, ...]]:
        batch: list[Any] = []
        for item in items:
            batch.append(item)
            if len(batch) == batch_size:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)
