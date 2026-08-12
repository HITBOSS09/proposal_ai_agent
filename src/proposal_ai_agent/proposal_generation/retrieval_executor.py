"""Provider-neutral execution of proposal section retrieval queries."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from .prompt_composer import RetrievedReference
from .retrieval_context import RetrievedContext
from .retrieval_query import ReferenceType, SectionRetrievalQuery
from .retrievers.provider import ProposalReferenceRetriever


class RetrievalExecutor:
    """Execute proposal retrieval queries and group results by section and reference type."""

    def __init__(self, retriever: ProposalReferenceRetriever) -> None:
        if not isinstance(retriever, ProposalReferenceRetriever):
            raise TypeError("retriever must implement ProposalReferenceRetriever")
        self._retriever = retriever

    def execute(self, queries: Sequence[SectionRetrievalQuery]) -> tuple[RetrievedContext, ...]:
        """Execute queries in order and return first-seen section contexts in order."""
        ordered_queries = tuple(queries)
        if any(not isinstance(query, SectionRetrievalQuery) for query in ordered_queries):
            raise TypeError("queries must contain SectionRetrievalQuery values")
        grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
        for query in ordered_queries:
            references = tuple(self._retriever.retrieve(query))
            self._validate_references(query, references)
            state = grouped.setdefault(
                query.section_id,
                {
                    "queries": [],
                    "authoring": [],
                    "technical": [],
                    "blueprint": [],
                    "seen": {"authoring": set(), "technical": set(), "blueprint": set()},
                },
            )
            state["queries"].append(query)  # type: ignore[index]
            bucket = state[query.reference_type.value]  # type: ignore[index]
            seen = state["seen"][query.reference_type.value]  # type: ignore[index]
            for reference in references:
                if reference.reference_id not in seen:
                    bucket.append(reference)
                    seen.add(reference.reference_id)
        return tuple(self._context(section_id, state) for section_id, state in grouped.items())

    @staticmethod
    def _validate_references(
        query: SectionRetrievalQuery, references: tuple[RetrievedReference, ...]
    ) -> None:
        if any(not isinstance(reference, RetrievedReference) for reference in references):
            raise TypeError("retriever must return RetrievedReference values")
        if any(reference.reference_type != query.reference_type.value for reference in references):
            raise ValueError("retriever returned a reference type that does not match the query")

    @staticmethod
    def _context(section_id: str, state: dict[str, object]) -> RetrievedContext:
        return RetrievedContext(
            section_id=section_id,
            queries=tuple(state["queries"]),  # type: ignore[arg-type]
            style_references=tuple(state[ReferenceType.AUTHORING.value]),  # type: ignore[arg-type]
            technical_references=tuple(state[ReferenceType.TECHNICAL.value]),  # type: ignore[arg-type]
            blueprint_references=tuple(state[ReferenceType.BLUEPRINT.value]),  # type: ignore[arg-type]
            metadata={"query_count": len(state["queries"])},  # type: ignore[arg-type]
        )
