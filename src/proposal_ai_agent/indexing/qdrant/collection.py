"""Qdrant collection and point conversion helpers."""

from __future__ import annotations

from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import IndexingConfig
from ..models import IndexPoint


_DISTANCE_BY_NAME = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def collection_vector_params(config: IndexingConfig) -> VectorParams:
    """Convert application collection settings to Qdrant vector parameters."""
    return VectorParams(
        size=config.vector_dimensions,
        distance=_DISTANCE_BY_NAME[config.distance],
    )


def point_to_qdrant(point: IndexPoint) -> PointStruct:
    """Convert a database-neutral point to Qdrant's point representation."""
    return PointStruct(id=point.id, vector=point.vector, payload=point.payload)

