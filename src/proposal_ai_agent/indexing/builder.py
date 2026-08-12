"""Pure transformation from enriched chunks and vectors to index points."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from proposal_ai_agent.ingestion.metadata import EnrichedChunkPayload

from .models import IndexPoint


class IndexBuilder:
    """Build database-neutral points without IDs, hashing, or I/O."""

    def __init__(self, vector_dimensions: int) -> None:
        if vector_dimensions <= 0:
            raise ValueError("vector_dimensions must be positive")
        self._vector_dimensions = vector_dimensions

    def build(self, payload: EnrichedChunkPayload, vector: Sequence[float]) -> IndexPoint:
        """Map one payload and its validated-size vector to an index point."""
        if len(vector) != self._vector_dimensions:
            raise ValueError(
                f"Expected {self._vector_dimensions} dimensions, received {len(vector)}"
            )
        point_payload = payload.model_dump(mode="json")
        point_payload.update(
            {
                "document_id": payload.document.document_id,
                "document_type": payload.document.document_type,
                "section_path": list(payload.section.section_path),
                "section_id": payload.section.section_id,
                "section_title": payload.section.heading,
                "parent_section_id": payload.section.parent_section_id,
                "element_type": payload.chunk.element_type,
                "language": payload.document.language,
                "document_role": payload.document_role,
            }
        )
        return IndexPoint(
            id=payload.point_uuid,
            vector=[float(value) for value in vector],
            payload=point_payload,
        )

    def build_batch(
        self, items: Iterable[Tuple[EnrichedChunkPayload, Sequence[float]]]
    ) -> List[IndexPoint]:
        """Build points in input order without side effects."""
        return [self.build(payload, vector) for payload, vector in items]
