"""Proposal-domain port for reference retrieval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..prompt_composer import RetrievedReference
from ..retrieval_query import SectionRetrievalQuery


@runtime_checkable
class ProposalReferenceRetriever(Protocol):
    """Retrieve typed proposal references for one provider-neutral query."""

    def retrieve(self, query: SectionRetrievalQuery) -> tuple[RetrievedReference, ...]:
        ...
