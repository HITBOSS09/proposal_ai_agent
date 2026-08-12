"""Tests for database-neutral indexing and the Qdrant writer adapter."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from proposal_ai_agent.indexing import (
    IndexBuilder,
    IndexingConfig,
    IndexPoint,
    QdrantIndexWriter,
)
from proposal_ai_agent.indexing.qdrant import point_to_qdrant
from proposal_ai_agent.ingestion.chunk_models import Chunk
from proposal_ai_agent.ingestion.metadata import enrich_chunk


def make_payload(text: str = "Indexable content"):
    chunk = Chunk(
        document_id=uuid4(),
        section_id=uuid4(),
        section_path=["Technical", "Scope"],
        heading="Scope",
        document_type="proposal",
        language="en",
        source_file="proposal.docx",
        source_path="/documents/proposal.docx",
        chunk_index=2,
        order_start=5,
        order_end=6,
        text=text,
        token_count=2,
    )
    return enrich_chunk(chunk, datetime(2026, 1, 1, tzinfo=timezone.utc))


def make_config() -> IndexingConfig:
    return IndexingConfig(
        collection_name="proposal_chunks",
        vector_dimensions=3,
        distance="cosine",
        payload_indexes=("document_id", "document_type", "section_path", "language"),
    )


def make_point(text: str = "Indexable content") -> IndexPoint:
    return IndexBuilder(3).build(make_payload(text), [0.1, 0.2, 0.3])


class FailingQdrantClient:
    def collection_exists(self, collection_name: str, **kwargs: Any) -> bool:
        return True

    def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        raise AssertionError("collection should already exist")

    def create_payload_index(
        self, collection_name: str, field_name: str, **kwargs: Any
    ) -> None:
        return None

    def retrieve(
        self, collection_name: str, ids: Sequence[str], **kwargs: Any
    ) -> Sequence[Any]:
        raise RuntimeError("database unavailable")

    def upsert(self, collection_name: str, points: Sequence[Any], **kwargs: Any) -> None:
        raise AssertionError("upsert should not run after retrieve failure")

    def get_collections(self, **kwargs: Any) -> None:
        raise RuntimeError("database unavailable")


def test_index_builder_maps_payload_vector_and_preserves_existing_uuid() -> None:
    payload = make_payload()
    point = IndexBuilder(3).build(payload, [0.1, 0.2, 0.3])

    assert point.id == payload.point_uuid
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload["document_id"] == payload.document.document_id
    assert point.payload["document_type"] == "proposal"
    assert point.payload["section_path"] == ["Technical", "Scope"]
    assert point.payload["language"] == "en"


def test_index_builder_preserves_batch_order() -> None:
    first = make_payload("first")
    second = make_payload("second")
    points = IndexBuilder(3).build_batch(((first, [1, 2, 3]), (second, [4, 5, 6])))
    assert [point.id for point in points] == [first.point_uuid, second.point_uuid]


def test_index_builder_rejects_wrong_vector_dimensions() -> None:
    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        IndexBuilder(3).build(make_payload(), [0.1, 0.2])


def test_index_point_rejects_invalid_payload_and_empty_vector() -> None:
    valid_id = str(uuid4())
    with pytest.raises(ValidationError, match="vector must not be empty"):
        IndexPoint(id=valid_id, vector=[], payload={})
    with pytest.raises(ValidationError, match="JSON serializable"):
        IndexPoint(id=valid_id, vector=[1.0], payload={"invalid": {1, 2}})


def test_index_point_is_frozen_and_requires_valid_uuid() -> None:
    with pytest.raises(ValidationError):
        IndexPoint(id="not-a-uuid", vector=[1.0], payload={})
    point = make_point()
    with pytest.raises(ValidationError):
        point.id = str(uuid4())  # type: ignore[misc]
    with pytest.raises(TypeError):
        point.vector.append(0.4)
    with pytest.raises(TypeError):
        point.payload["document_id"] = "changed"
    with pytest.raises(TypeError):
        point.payload["section"]["section_path"].append("changed")


def test_point_mapping_creates_qdrant_point_struct() -> None:
    point = make_point()
    qdrant_point = point_to_qdrant(point)
    assert isinstance(qdrant_point, PointStruct)
    assert str(qdrant_point.id) == point.id
    assert qdrant_point.vector == point.vector
    assert qdrant_point.payload == point.payload


def test_writer_creates_collection_payload_indexes_and_upserts() -> None:
    client = QdrantClient(":memory:")
    writer = QdrantIndexWriter(client, make_config())
    point = make_point()

    assert writer.health_check() is True
    first = writer.upsert_batch([point])
    second = writer.upsert_batch([point])
    assert first.total_points == 1
    assert first.inserted == 1
    assert first.updated == 0
    assert first.failed == 0
    assert second.inserted == 0
    assert second.updated == 1
    assert second.duration_ms >= 0


def test_writer_automatically_creates_indexes_only_for_new_collection() -> None:
    client = Mock()
    client.collection_exists.return_value = False
    client.get_collection.return_value = SimpleNamespace(payload_schema={})
    client.retrieve.return_value = []
    writer = QdrantIndexWriter(client, make_config())

    result = writer.upsert_batch([make_point()])

    assert result.failed == 0
    assert client.create_collection.call_count == 1
    assert client.create_payload_index.call_count == 4
    assert [call.kwargs["field_name"] for call in client.create_payload_index.call_args_list] == [
        "document_id",
        "document_type",
        "section_path",
        "language",
    ]
    client.collection_exists.return_value = True
    writer.upsert_batch([make_point("second")])
    assert client.create_payload_index.call_count == 4


def test_writer_recovers_from_partial_payload_index_initialization() -> None:
    client = Mock()
    payload_schema: dict[str, object] = {}
    client.collection_exists.side_effect = [False, True]
    client.get_collection.side_effect = lambda **kwargs: SimpleNamespace(
        payload_schema=payload_schema
    )
    client.retrieve.return_value = []
    index_attempts: list[str] = []

    def create_payload_index(*, field_name: str, **kwargs: Any) -> None:
        index_attempts.append(field_name)
        if len(index_attempts) == 1:
            raise RuntimeError("first payload index creation failed")
        payload_schema[field_name] = object()

    client.create_payload_index.side_effect = create_payload_index
    writer = QdrantIndexWriter(client, make_config())

    failed = writer.upsert_batch([make_point()])

    assert failed.failed == 1
    assert client.create_collection.call_count == 1
    assert client.upsert.call_count == 0
    assert writer._process_verified is False

    recovered = writer.upsert_batch([make_point("recovered")])

    assert recovered.failed == 0
    assert client.collection_exists.call_count == 2
    assert client.get_collection.call_count == 2
    assert index_attempts[1:] == list(make_config().payload_indexes)
    assert client.upsert.call_count == 1
    assert writer._process_verified is True

    writer.upsert_batch([make_point("later")])

    assert client.collection_exists.call_count == 2
    assert client.get_collection.call_count == 2
    assert client.create_payload_index.call_count == 5


def test_writer_rejects_duplicate_ids_and_dimension_mismatch() -> None:
    writer = QdrantIndexWriter(QdrantClient(":memory:"), make_config())
    point = make_point()
    with pytest.raises(ValueError, match="duplicate"):
        writer.upsert_batch([point, point])
    wrong_dimension = IndexPoint(id=str(uuid4()), vector=[1.0], payload={})
    with pytest.raises(ValueError, match="dimensions"):
        writer.upsert_batch([wrong_dimension])


def test_writer_returns_failed_result_and_unhealthy_status(caplog: pytest.LogCaptureFixture) -> None:
    writer = QdrantIndexWriter(FailingQdrantClient(), make_config())
    with caplog.at_level("ERROR"):
        result = writer.upsert_batch([make_point()])
    assert result.total_points == 1
    assert result.inserted == 0
    assert result.updated == 0
    assert result.failed == 1
    assert "Qdrant batch upsert failed" in caplog.text
    with caplog.at_level("WARNING"):
        assert writer.health_check() is False
    assert "Qdrant health check failed" in caplog.text
