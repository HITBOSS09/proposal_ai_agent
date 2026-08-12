"""Qdrant persistence adapter for database-neutral index points."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Iterable, List, Sequence

from qdrant_client.models import PayloadSchemaType

from .config import IndexingConfig
from .models import IndexPoint, IndexingResult
from .qdrant.client import QdrantClientProtocol
from .qdrant.collection import collection_vector_params, point_to_qdrant

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class QdrantIndexWriter:
    """Create configured Qdrant collections and batch-upsert index points."""

    def __init__(self, client: QdrantClientProtocol, config: IndexingConfig) -> None:
        self._client = client
        self._config = config
        self._process_verified = False

    def ensure_collection_exists(self) -> bool:
        """Ensure the configured collection and payload indexes are ready."""
        return self.ensure_collection_ready()

    def ensure_collection_ready(self) -> bool:
        """Reconcile the collection's configured payload indexes with its metadata."""
        if self._process_verified:
            return False

        try:
            created = False
            if not self._client.collection_exists(self._config.collection_name):
                self._client.create_collection(
                    collection_name=self._config.collection_name,
                    vectors_config=collection_vector_params(self._config),
                )
                created = True

            collection = getattr(self._client, "get_collection")(
                collection_name=self._config.collection_name
            )
            payload_schema = getattr(collection, "payload_schema", {}) or {}
            existing_indexes = set(payload_schema)
            for field_name in self._config.payload_indexes:
                if field_name not in existing_indexes:
                    self._client.create_payload_index(
                        collection_name=self._config.collection_name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
            self._process_verified = True
            return created
        except Exception:
            self._process_verified = False
            raise

    def create_payload_indexes(self) -> None:
        """Create configured keyword indexes that are absent from collection metadata."""
        collection = getattr(self._client, "get_collection")(
            collection_name=self._config.collection_name
        )
        payload_schema = getattr(collection, "payload_schema", {}) or {}
        existing_indexes = set(payload_schema)
        for field_name in self._config.payload_indexes:
            if field_name not in existing_indexes:
                self._client.create_payload_index(
                    collection_name=self._config.collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

    def health_check(self) -> bool:
        """Return whether the Qdrant client can answer a metadata request."""
        try:
            self._client.get_collections()
        except Exception:
            logger.warning(
                "Qdrant health check failed",
                extra={"collection_name": self._config.collection_name},
                exc_info=True,
            )
            return False
        return True

    def upsert_batch(self, points: Iterable[IndexPoint]) -> IndexingResult:
        """Upsert one batch and report inserted, updated, and failed counts."""
        started_at = perf_counter()
        batch = list(points)
        self._validate_batch(batch)
        if not batch:
            return self._result(total_points=0, inserted=0, updated=0, failed=0, started_at=started_at)

        try:
            self.ensure_collection_ready()
            existing_points = self._client.retrieve(
                collection_name=self._config.collection_name,
                ids=[point.id for point in batch],
                with_payload=False,
                with_vectors=False,
            )
            updated = len(existing_points)
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=[point_to_qdrant(point) for point in batch],
                wait=True,
            )
        except Exception:
            logger.exception(
                "Qdrant batch upsert failed",
                extra={
                    "collection_name": self._config.collection_name,
                    "point_count": len(batch),
                },
            )
            return self._result(
                total_points=len(batch),
                inserted=0,
                updated=0,
                failed=len(batch),
                started_at=started_at,
            )
        return self._result(
            total_points=len(batch),
            inserted=len(batch) - updated,
            updated=updated,
            failed=0,
            started_at=started_at,
        )

    def _validate_batch(self, points: Sequence[IndexPoint]) -> None:
        """Reject duplicate IDs and vectors incompatible with collection settings."""
        point_ids = [point.id for point in points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("batch contains duplicate point IDs")
        if any(len(point.vector) != self._config.vector_dimensions for point in points):
            raise ValueError("point vector dimensions do not match collection configuration")

    def _result(
        self,
        total_points: int,
        inserted: int,
        updated: int,
        failed: int,
        started_at: float,
    ) -> IndexingResult:
        """Build one timing-aware immutable indexing result."""
        return IndexingResult(
            collection_name=self._config.collection_name,
            total_points=total_points,
            inserted=inserted,
            updated=updated,
            failed=failed,
            duration_ms=(perf_counter() - started_at) * 1000,
        )
