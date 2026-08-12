"""Qdrant implementation of the online vector-repository boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from ..contracts.retrieval import RetrievedCandidate, RetrievalRequest
from ..engines.retrieval_engine import RepositorySearchResult

class PayloadContractError(ValueError):
    """A Qdrant point does not satisfy the indexed retrieval payload contract."""

REQUIRED_FIELDS = frozenset({"chunk_id", "document_id", "original_text", "source_document"})

def validate_qdrant_payload(payload: Mapping[str, Any]) -> None:
    missing = sorted(name for name in REQUIRED_FIELDS if name not in payload)
    if missing:
        raise PayloadContractError("Qdrant payload missing required fields: " + ", ".join(missing))


class QdrantVectorRepository:
    """Execute dense retrieval requests against an already-indexed collection."""

    def __init__(self, client: Any, collection_name: str, timeout: int = 10) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._client = client
        self._collection_name = collection_name
        self._timeout = timeout

    def search(self, retrieval_request: RetrievalRequest) -> RepositorySearchResult:
        """Run the frozen dense-search plan and hydrate online retrieval candidates."""
        if retrieval_request.search_scope.collection not in (None, self._collection_name):
            raise ValueError("retrieval request collection does not match repository collection")
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(retrieval_request.query_embedding.vector),
            query_filter=self._payload_filter(retrieval_request.metadata_filters),
            limit=min(retrieval_request.candidate_budget, retrieval_request.retrieval_budget.max_candidates),
            with_payload=True,
            with_vectors=False,
            score_threshold=retrieval_request.score_threshold,
            timeout=self._timeout,
        )
        points = getattr(response, "points", response)
        candidates = tuple(self._candidate_from_point(point) for point in points)
        return RepositorySearchResult(
            candidates=candidates,
            repository_metadata={"repository": "qdrant", "collection": self._collection_name, "returned_points": len(candidates)},
        )

    @staticmethod
    def _payload_filter(metadata_filters: Mapping[str, Any]) -> Filter | None:
        conditions = [
            FieldCondition(
                key="source_document" if name == "document" else name,
                match=MatchValue(value=value),
            )
            for name, value in metadata_filters.items()
            if value is not None and not isinstance(value, (Mapping, list, tuple, set))
        ]
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _candidate_from_point(point: Any) -> RetrievedCandidate:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, Mapping):
            raise PayloadContractError("Qdrant result is missing its retrieval payload")
        validate_qdrant_payload(payload)
        document, section, chunk = payload.get("document"), payload.get("section"), payload.get("chunk")
        document_id = payload.get("document_id") or (document.get("document_id") if isinstance(document, Mapping) else None)
        section_path = payload.get("section_path") or (section.get("section_path") if isinstance(section, Mapping) else ())
        chunk_index = payload.get("chunk_index")
        if chunk_index is None and isinstance(chunk, Mapping):
            chunk_index = chunk.get("chunk_index")
        text = payload.get("original_text") or payload.get("text")
        chunk_id = payload["chunk_id"]
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise ValueError("Qdrant payload does not match the indexed chunk schema")
        if not isinstance(section_path, (list, tuple)) or not isinstance(chunk_index, int):
            raise ValueError("Qdrant payload is missing section or chunk metadata")
        return RetrievedCandidate(
            chunk_id=str(chunk_id), document_id=document_id, text=text,
            score=float(getattr(point, "score")), metadata=dict(payload),
            header_path=tuple(str(item) for item in section_path), chunk_index=chunk_index,
            point_id=str(getattr(point, "id", "")), page_number=payload.get("page_number"),
        )
