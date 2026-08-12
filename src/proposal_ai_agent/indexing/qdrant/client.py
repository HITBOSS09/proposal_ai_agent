"""Typed protocol for the Qdrant client operations used by the writer."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class QdrantClientProtocol(Protocol):
    """Narrow Qdrant client surface required for indexing."""

    def collection_exists(self, collection_name: str, **kwargs: Any) -> bool:
        """Return whether a collection exists."""

    def create_collection(self, collection_name: str, **kwargs: Any) -> Any:
        """Create a collection."""

    def create_payload_index(
        self, collection_name: str, field_name: str, **kwargs: Any
    ) -> Any:
        """Create a payload index."""

    def retrieve(self, collection_name: str, ids: Sequence[str], **kwargs: Any) -> Sequence[Any]:
        """Retrieve existing points by ID."""

    def upsert(self, collection_name: str, points: Sequence[Any], **kwargs: Any) -> Any:
        """Upsert points into a collection."""

    def get_collections(self, **kwargs: Any) -> Any:
        """Return collection metadata as a health-check operation."""

