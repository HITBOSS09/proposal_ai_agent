"""Build final answer responses from generated responses and prompt context."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..contracts.response import (
    AnswerResponse,
    AuditTrace,
    GeneratedResponse,
    ResponseMetadata,
    SourceCitation,
)
from ..contracts.synthesis import PromptPackage


class ResponseBuilder:
    """Create an immutable final answer response from existing pipeline artifacts."""

    def build_response(
        self, prompt_package: PromptPackage, generated_response: GeneratedResponse
    ) -> AnswerResponse:
        if not isinstance(prompt_package, PromptPackage):
            raise TypeError("prompt_package must be a PromptPackage")
        if not isinstance(generated_response, GeneratedResponse):
            raise TypeError("generated_response must be a GeneratedResponse")

        final_answer = generated_response.generated_text.strip()
        if not final_answer:
            raise ValueError("final_answer must not be empty")

        assembled_context = prompt_package.assembled_context
        source_citations = tuple(self._build_source_citation(citation) for citation in assembled_context.citations)
        self._validate_citations(source_citations, assembled_context.citations)

        response_metadata = ResponseMetadata(
            provider_name=generated_response.provider_name,
            model_name=generated_response.model_name,
            finish_reason=generated_response.finish_reason,
            prompt_tokens=generated_response.token_usage["prompt_tokens"],
            completion_tokens=generated_response.token_usage["completion_tokens"],
            total_tokens=generated_response.token_usage["total_tokens"],
            latency_ms=generated_response.latency_ms,
            generation_timestamp=generated_response.generation_timestamp,
        )

        query_embedding = assembled_context.retrieved_context.retrieval_request.query_embedding
        request_id = query_embedding.processed_query.qualified_query.original.request_id
        session_id = query_embedding.processed_query.qualified_query.original.session_id
        query_hash = query_embedding.processed_query.query_hash
        retrieval_summary = assembled_context.retrieved_context.retrieval_summary

        audit_trace = AuditTrace(
            request_id=request_id,
            session_id=session_id,
            query_hash=query_hash,
            retrieval_summary=retrieval_summary,
            provider=generated_response.provider_name,
            execution_timestamp=generated_response.generation_timestamp,
        )

        return AnswerResponse(
            final_answer=final_answer,
            citations=source_citations,
            response_metadata=response_metadata,
            audit_trace=audit_trace,
        )

    @staticmethod
    def _build_source_citation(citation: Any) -> SourceCitation:
        if not hasattr(citation, "citation_id"):
            raise TypeError("citation must provide citation_id")

        metadata = citation.metadata or {}
        document_name = None
        source_metadata = metadata.get("source_metadata") if isinstance(metadata, Mapping) else None
        if isinstance(source_metadata, Mapping):
            document_name = source_metadata.get("document_title") or source_metadata.get("title")

        section = None
        hierarchy_path = tuple(citation.header_path) if getattr(citation, "header_path", None) else ()
        if hierarchy_path:
            section = hierarchy_path[-1]

        return SourceCitation(
            citation_id=citation.citation_id,
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            document_name=document_name,
            section=section,
            hierarchy_path=hierarchy_path,
            page=citation.page,
            metadata=metadata,
        )

    @staticmethod
    def _validate_citations(
        citations: tuple[SourceCitation, ...], assembled_citations: tuple[Any, ...]
    ) -> None:
        if len(citations) != len(assembled_citations):
            raise ValueError("citations must preserve assembled context citations")
        assembled_ids = {citation.citation_id for citation in assembled_citations}
        if any(citation.citation_id not in assembled_ids for citation in citations):
            raise ValueError("citations reference unknown assembled context sources")
