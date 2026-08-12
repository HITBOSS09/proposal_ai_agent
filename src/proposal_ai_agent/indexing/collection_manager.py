"""Collection lifecycle management for the BDIL vector index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from .exceptions import CollectionAlreadyExists, CollectionConfigurationMismatch


@dataclass(frozen=True, slots=True)
class CollectionMetadata:
    """Immutable description of the collection contract managed by this runtime."""

    collection_name: str
    embedding_dimension: int
    embedding_model: str
    distance: str
    schema_version: str


class CollectionManager:
    """Create and validate Qdrant collections; it never parses or embeds content."""

    SCHEMA_VERSION = "1"
    _PAYLOAD_INDEXES: Mapping[str, PayloadSchemaType] = {
        "document_id": PayloadSchemaType.KEYWORD,
        "document_title": PayloadSchemaType.KEYWORD,
        "section_path": PayloadSchemaType.KEYWORD,
        "page_number": PayloadSchemaType.INTEGER,
        "chunk_id": PayloadSchemaType.KEYWORD,
        "source_document": PayloadSchemaType.KEYWORD,
        "version": PayloadSchemaType.KEYWORD,
        "document_fingerprint": PayloadSchemaType.KEYWORD,
        "document_role": PayloadSchemaType.KEYWORD,
    }

    def __init__(self, client: Any, collection_name: str, embedding_dimension: int, embedding_model: str) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        self._client = client
        self._metadata = CollectionMetadata(
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            distance="cosine",
            schema_version=self.SCHEMA_VERSION,
        )

    @property
    def metadata(self) -> CollectionMetadata:
        return self._metadata

    def ensure_ready(self, recreate: bool = False) -> bool:
        """Create or validate a collection and ensure its payload indexes exist.

        Returns ``True`` only when a collection was created. ``recreate`` is an
        explicit destructive operation intended for the CLI's ``--recreate`` flag.
        """
        exists = self._client.collection_exists(self._metadata.collection_name)
        if exists and recreate:
            self._client.delete_collection(self._metadata.collection_name)
            exists = False
        if not exists:
            self._create_collection()
            self._create_payload_indexes()
            return True
        self._validate_existing_collection()
        self._create_payload_indexes()
        self._validate_payload_schema()
        return False

    def create(self) -> None:
        """Create a new collection, refusing to overwrite an existing one."""
        if self._client.collection_exists(self._metadata.collection_name):
            raise CollectionAlreadyExists(f"collection already exists: {self._metadata.collection_name}")
        self._create_collection()
        self._create_payload_indexes()

    def collection_exists(self) -> bool:
        return bool(self._client.collection_exists(self._metadata.collection_name))

    def _create_collection(self) -> None:
        self._client.create_collection(
            collection_name=self._metadata.collection_name,
            vectors_config=VectorParams(size=self._metadata.embedding_dimension, distance=Distance.COSINE),
            metadata={
                "bdil_schema_version": self._metadata.schema_version,
                "embedding_model": self._metadata.embedding_model,
            },
        )

    def _create_payload_indexes(self) -> None:
        collection = self._client.get_collection(self._metadata.collection_name)
        existing = set(getattr(collection, "payload_schema", {}) or {})
        for field_name, field_schema in self._PAYLOAD_INDEXES.items():
            if field_name not in existing:
                self._client.create_payload_index(
                    collection_name=self._metadata.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

    def _validate_payload_schema(self) -> None:
        """Reject existing payload indexes whose type conflicts with this contract."""
        collection = self._client.get_collection(self._metadata.collection_name)
        schema = getattr(collection, "payload_schema", {}) or {}
        for field_name, expected in self._PAYLOAD_INDEXES.items():
            actual = schema.get(field_name)
            if actual is None:
                continue  # Local Qdrant does not expose payload-index schema.
            actual_type = getattr(actual, "data_type", actual)
            if str(actual_type).lower().split(".")[-1] != expected.value:
                raise CollectionConfigurationMismatch(
                    f"payload index '{field_name}' does not match the required {expected.value} schema"
                )

    def _validate_existing_collection(self) -> None:
        collection = self._client.get_collection(self._metadata.collection_name)
        vectors = collection.config.params.vectors
        actual_dimension = getattr(vectors, "size", None)
        actual_distance = getattr(vectors, "distance", None)
        if actual_dimension != self._metadata.embedding_dimension or actual_distance != Distance.COSINE:
            raise CollectionConfigurationMismatch(
                f"collection '{self._metadata.collection_name}' must use cosine vectors with "
                f"dimension {self._metadata.embedding_dimension}"
            )
        metadata = getattr(collection.config, "metadata", None) or {}
        model = metadata.get("embedding_model") if isinstance(metadata, Mapping) else None
        schema = metadata.get("bdil_schema_version") if isinstance(metadata, Mapping) else None
        if model is not None and model != self._metadata.embedding_model:
            raise CollectionConfigurationMismatch("collection embedding model does not match the configured model")
        if schema is not None and schema != self._metadata.schema_version:
            raise CollectionConfigurationMismatch("collection schema version is not supported")
