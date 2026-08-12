"""Knowledge retrieval execution over a vendor-neutral vector repository."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from ..contracts.retrieval import (
    ProcessedCandidate,
    ProcessedCandidates,
    RetrievedCandidate,
    RetrievedCandidates,
    RetrievedContext,
    RetrievalRequest,
)
from ..contracts.retrieval import RerankingPolicy
from ..providers.reranking.provider import RerankingProvider, RerankingScore


@dataclass(frozen=True, slots=True)
class RepositorySearchResult:
    """Normalized repository response, independent of a vector-store SDK."""

    candidates: Sequence[RetrievedCandidate]
    repository_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach response containers supplied by a repository adapter."""
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "repository_metadata", dict(self.repository_metadata))


@runtime_checkable
class VectorRepository(Protocol):
    """Vendor-neutral boundary for execution of a complete retrieval request."""

    def search(self, retrieval_request: RetrievalRequest) -> RepositorySearchResult:
        """Execute every retrieval directive and return hydrated candidates."""


class RetrievalEngine:
    """Execute a frozen retrieval plan without changing its candidate set."""

    def __init__(
        self,
        repository: VectorRepository,
        reranking_provider: RerankingProvider | None = None,
    ) -> None:
        if not isinstance(repository, VectorRepository):
            raise TypeError("repository must implement VectorRepository")
        if reranking_provider is not None and not isinstance(reranking_provider, RerankingProvider):
            raise TypeError("reranking_provider must implement RerankingProvider")
        self._repository = repository
        self._reranking_provider = reranking_provider

    def retrieve(self, retrieval_request: RetrievalRequest) -> RetrievedCandidates:
        """Execute ``retrieval_request`` exactly once and preserve result order."""
        if not isinstance(retrieval_request, RetrievalRequest):
            raise TypeError("retrieval_request must be a RetrievalRequest")

        started_at = perf_counter()
        repository_result = self._repository.search(retrieval_request)
        retrieval_time_ms = (perf_counter() - started_at) * 1000
        if not isinstance(repository_result, RepositorySearchResult):
            raise TypeError("repository.search must return a RepositorySearchResult")

        candidates = tuple(repository_result.candidates)
        return RetrievedCandidates(
            retrieval_request=retrieval_request,
            candidates=candidates,
            retrieval_time_ms=retrieval_time_ms,
            candidate_count=len(candidates),
            repository_metadata=repository_result.repository_metadata,
        )

    def process(self, retrieved_candidates: RetrievedCandidates) -> ProcessedCandidates:
        """Apply deterministic, linear post-retrieval processing without I/O."""
        if not isinstance(retrieved_candidates, RetrievedCandidates):
            raise TypeError("retrieved_candidates must be a RetrievedCandidates")

        started_at = perf_counter()
        request = retrieved_candidates.retrieval_request
        summary = {
            "input_count": len(retrieved_candidates.candidates),
            "invalid_metadata_removed": 0,
            "score_threshold_removed": 0,
            "low_quality_removed": 0,
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "adjacent_merges": 0,
        }
        exact_seen: set[tuple[Any, ...]] = set()
        chunk_ids_seen: set[str] = set()
        chunk_locations_seen: set[tuple[str, int]] = set()
        metadata_keys_seen: set[tuple[str, str]] = set()
        accepted: list[RetrievedCandidate] = []

        for candidate in retrieved_candidates.candidates:
            if not self._has_valid_metadata(candidate):
                summary["invalid_metadata_removed"] += 1
                continue
            if candidate.score < request.score_threshold:
                summary["score_threshold_removed"] += 1
                continue
            if not candidate.text.strip():
                summary["low_quality_removed"] += 1
                continue

            exact_key = self._exact_key(candidate)
            if exact_key in exact_seen:
                summary["exact_duplicates_removed"] += 1
                continue
            exact_seen.add(exact_key)

            metadata_keys = self._near_duplicate_metadata_keys(candidate.metadata)
            if (
                candidate.chunk_id in chunk_ids_seen
                or (candidate.document_id, candidate.chunk_index) in chunk_locations_seen
                or any(key in metadata_keys_seen for key in metadata_keys)
            ):
                summary["near_duplicates_removed"] += 1
                continue

            chunk_ids_seen.add(candidate.chunk_id)
            chunk_locations_seen.add((candidate.document_id, candidate.chunk_index))
            metadata_keys_seen.update(metadata_keys)
            accepted.append(candidate)

        processed: list[ProcessedCandidate] = []
        for candidate in accepted:
            if self._can_merge(processed[-1], candidate) if processed else False:
                previous = processed[-1]
                merged_count = previous.processing_flags["merged_chunk_count"] + 1
                processed[-1] = ProcessedCandidate(
                    chunk_id=previous.chunk_id,
                    document_id=previous.document_id,
                    text=f"{previous.text}\n\n{candidate.text}",
                    score=previous.score,
                    metadata=previous.metadata,
                    header_path=previous.header_path,
                    chunk_index=previous.chunk_index,
                    processing_flags={
                        "metadata_validated": True,
                        "merged_chunk_count": merged_count,
                        "merged_chunk_ids": (*previous.processing_flags["merged_chunk_ids"], candidate.chunk_id),
                    },
                )
                summary["adjacent_merges"] += 1
                continue
            processed.append(self._processed_candidate(candidate))

        summary["output_count"] = len(processed)
        return ProcessedCandidates(
            retrieval_request=request,
            candidates=tuple(processed),
            processing_summary=summary,
            processing_time_ms=(perf_counter() - started_at) * 1000,
        )

    def rerank(self, processed_candidates: ProcessedCandidates) -> RetrievedContext:
        """Select final Top-K context without performing retrieval or refinement."""
        if not isinstance(processed_candidates, ProcessedCandidates):
            raise TypeError("processed_candidates must be a ProcessedCandidates")

        request = processed_candidates.retrieval_request
        candidates = processed_candidates.candidates
        reranking_applied = request.reranking_policy is not RerankingPolicy.DISABLED
        reranking_time_ms = 0.0
        if reranking_applied:
            if self._reranking_provider is None:
                raise RuntimeError("a reranking_provider is required by the retrieval request")
            pool = candidates[: request.candidate_budget]
            started_at = perf_counter()
            scores = tuple(
                self._reranking_provider.rerank(
                    request.query_embedding.processed_query.normalized_query,
                    pool,
                )
            )
            reranking_time_ms = (perf_counter() - started_at) * 1000
            ordered_pool = self._order_by_relevance(pool, scores)
            selected = ordered_pool[: request.top_k]
        else:
            selected = candidates[: request.top_k]

        return RetrievedContext(
            retrieval_request=request,
            candidates=selected,
            final_context="\n\n".join(candidate.text for candidate in selected),
            total_candidates=len(candidates),
            returned_candidates=len(selected),
            reranking_applied=reranking_applied,
            reranking_time_ms=reranking_time_ms,
            retrieval_summary={
                "processing_summary": processed_candidates.processing_summary,
                "candidate_budget": request.candidate_budget,
                "top_k": request.top_k,
                "reranking_pool_size": min(len(candidates), request.candidate_budget)
                if reranking_applied
                else 0,
            },
        )

    @staticmethod
    def _order_by_relevance(
        candidates: tuple[ProcessedCandidate, ...],
        scores: tuple[RerankingScore, ...],
    ) -> tuple[ProcessedCandidate, ...]:
        """Stably reorder a bounded pool using complete provider output."""
        if len(scores) != len(candidates):
            raise ValueError("reranking provider must score every candidate")
        indices = [score.candidate_index for score in scores]
        if len(set(indices)) != len(indices) or set(indices) != set(range(len(candidates))):
            raise ValueError("reranking provider returned invalid candidate indices")
        scores_by_index = {score.candidate_index: score.relevance_score for score in scores}
        return tuple(
            candidate
            for _, candidate in sorted(
                enumerate(candidates),
                key=lambda item: (-scores_by_index[item[0]], item[0]),
            )
        )

    @staticmethod
    def _processed_candidate(candidate: RetrievedCandidate) -> ProcessedCandidate:
        """Copy only downstream-safe candidate fields; vector payloads are excluded."""
        return ProcessedCandidate(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            text=candidate.text,
            score=candidate.score,
            metadata=candidate.metadata,
            header_path=candidate.header_path,
            chunk_index=candidate.chunk_index,
            processing_flags={
                "metadata_validated": True,
                "merged_chunk_count": 1,
                "merged_chunk_ids": (candidate.chunk_id,),
            },
        )

    @staticmethod
    def _has_valid_metadata(candidate: RetrievedCandidate) -> bool:
        """Validate available structural payload fields without reconstructing them."""
        metadata = candidate.metadata
        checks = (
            (metadata.get("chunk_id"), candidate.chunk_id),
            (metadata.get("document_id"), candidate.document_id),
            (metadata.get("chunk_index"), candidate.chunk_index),
            (metadata.get("section_path"), candidate.header_path),
        )
        for actual, expected in checks:
            if actual is not None and actual != expected:
                return False

        document = metadata.get("document")
        if isinstance(document, Mapping) and document.get("document_id") not in (None, candidate.document_id):
            return False
        chunk = metadata.get("chunk")
        if isinstance(chunk, Mapping) and chunk.get("chunk_index") not in (None, candidate.chunk_index):
            return False
        section = metadata.get("section")
        if isinstance(section, Mapping) and tuple(section.get("section_path", ())) != candidate.header_path:
            return False
        return True

    @staticmethod
    def _exact_key(candidate: RetrievedCandidate) -> tuple[Any, ...]:
        """Build an O(1) comparable representation for exact duplicate detection."""
        return (
            candidate.chunk_id,
            candidate.document_id,
            candidate.text,
            candidate.score,
            candidate.header_path,
            candidate.chunk_index,
            RetrievalEngine._freeze_for_key(candidate.metadata),
        )

    @staticmethod
    def _freeze_for_key(value: Any) -> Any:
        """Convert JSON-like metadata into a deterministic hashable key."""
        if isinstance(value, Mapping):
            return tuple(
                sorted((str(key), RetrievalEngine._freeze_for_key(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(RetrievalEngine._freeze_for_key(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(
                    (RetrievalEngine._freeze_for_key(item) for item in value),
                    key=repr,
                )
            )
        return value

    @staticmethod
    def _near_duplicate_metadata_keys(metadata: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        """Use only stable indexed identifiers; no text similarity is performed."""
        keys: list[tuple[str, str]] = []
        for name in ("point_uuid", "content_hash"):
            value = metadata.get(name)
            if value is not None:
                keys.append((name, str(value)))
        return tuple(keys)

    @staticmethod
    def _can_merge(previous: ProcessedCandidate, current: RetrievedCandidate) -> bool:
        """Merge only consecutive chunks with an identical stored hierarchy."""
        return (
            previous.document_id == current.document_id
            and previous.chunk_index + previous.processing_flags["merged_chunk_count"]
            == current.chunk_index
            and previous.header_path == current.header_path
        )
