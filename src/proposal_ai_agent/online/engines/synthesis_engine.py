"""Context assembly engine for the online pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..benchmarks import BenchmarkRegistry
from ..benchmarks.profiles import RAG_QA_PROFILE
from ..contracts.retrieval import RetrievedContext
from ..contracts.synthesis import AssembledContext, Citation, PromptPackage
from ...ingestion.chunker import count_tokens


@dataclass(frozen=True, slots=True)
class SynthesisEngine:
    """Assemble an immutable high-value context payload from retrieved sources."""

    _registry: BenchmarkRegistry = field(init=False, repr=False)

    def assemble(self, retrieved_context: RetrievedContext) -> AssembledContext:
        if not isinstance(retrieved_context, RetrievedContext):
            raise TypeError("retrieved_context must be a RetrievedContext")

        request = retrieved_context.retrieval_request
        budget = request.retrieval_budget.max_context_tokens
        if budget <= 0:
            raise ValueError("retrieval_budget.max_context_tokens must be positive")

        candidates = retrieved_context.candidates
        assembled_segments: list[str] = []
        citations: list[Citation] = []
        used_tokens = 0
        truncated = False
        last_truncated_citation_id: str | None = None

        for index, candidate in enumerate(candidates):
            available_tokens = budget - used_tokens
            if available_tokens <= 0:
                break

            text = candidate.text
            if not text.strip():
                continue

            token_count = count_tokens(text)
            truncated_candidate = False
            segment = text
            if token_count > available_tokens:
                segment = self._truncate_text(text, available_tokens)
                token_count = count_tokens(segment)
                truncated_candidate = True

            if token_count == 0:
                break

            citation_id = f"c{index + 1}"
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    header_path=candidate.header_path,
                    chunk_index=candidate.chunk_index,
                    token_count=token_count,
                    page=self._extract_page(candidate.metadata),
                    truncated=truncated_candidate,
                    metadata={
                        "source_metadata": candidate.metadata,
                        "processing_flags": candidate.processing_flags,
                        "document_id": candidate.document_id,
                        "chunk_id": candidate.chunk_id,
                        "header_path": candidate.header_path,
                        "chunk_index": candidate.chunk_index,
                    },
                )
            )
            assembled_segments.append(segment)
            used_tokens += token_count
            if truncated_candidate:
                truncated = True
                last_truncated_citation_id = citation_id
                break

        assembled_context_text = "\n\n".join(assembled_segments).strip()
        if not assembled_context_text:
            raise ValueError("assembled_context must not be empty")

        metadata = {
            "source_count": len(citations),
            "sources": [
                {
                    "citation_id": citation.citation_id,
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "header_path": citation.header_path,
                    "chunk_index": citation.chunk_index,
                    "page": citation.page,
                    "metadata": citation.metadata,
                    "truncated": citation.truncated,
                    "token_count": citation.token_count,
                }
                for citation in citations
            ],
        }

        context_statistics = {
            "total_chunks": len(candidates),
            "merged_chunks": sum(
                max(0, candidate.processing_flags.get("merged_chunk_count", 1) - 1)
                for candidate in candidates
            ),
            "total_tokens": used_tokens,
            "total_sources": len(citations),
        }

        token_usage = {
            "budget_tokens": budget,
            "used_tokens": used_tokens,
            "remaining_tokens": budget - used_tokens,
            "truncated": truncated,
            "last_truncated_citation_id": last_truncated_citation_id,
        }

        return AssembledContext(
            retrieved_context=retrieved_context,
            assembled_context=assembled_context_text,
            citations=tuple(citations),
            metadata=metadata,
            context_statistics=context_statistics,
            token_usage=token_usage,
        )

    def build_prompt(self, assembled_context: AssembledContext) -> PromptPackage:
        if not isinstance(assembled_context, AssembledContext):
            raise TypeError("assembled_context must be an AssembledContext")

        processed_query = assembled_context.retrieved_context.retrieval_request.query_embedding.processed_query
        profile = self._registry.get(processed_query.qualified_query.benchmark_id)
        prompt_template = profile.defaults.get(
            "prompt_template",
            "{system_prompt}\n\n{conversation_history}\n\n{context_block}\n\n{user_prompt}",
        )
        output_format = profile.defaults.get("output_format", "markdown")
        formatted_history = self._format_conversation_history(
            processed_query.qualified_query.original.conversation_history
        )
        system_prompt = self._build_system_prompt(output_format)
        context_block = self._build_context_block(assembled_context)
        user_prompt = self._build_user_prompt(processed_query)

        prompt_statistics = self._compute_prompt_statistics(
            system_prompt, user_prompt, formatted_history, assembled_context.assembled_context
        )
        generation_metadata = {
            "benchmark_id": profile.intent_id,
            "prompt_template": prompt_template,
            "output_format": output_format,
            "prompt_construction_version": "1.0",
        }

        return PromptPackage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            conversation_history=formatted_history,
            assembled_context=assembled_context,
            prompt_template=prompt_template,
            output_format=output_format,
            prompt_statistics=prompt_statistics,
            generation_metadata=generation_metadata,
            validation_result=processed_query.qualified_query.validation_result,
        )

    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        object.__setattr__(self, "_registry", registry or BenchmarkRegistry((RAG_QA_PROFILE,)))

    def _build_system_prompt(self, output_format: str) -> str:
        return (
            "You are an enterprise knowledge assistant. Answer only from the provided context and cite sources by citation id. "
            "If the answer is not contained in the context, say you do not know. "
            f"Respond in {output_format} format."
        )

    @staticmethod
    def _build_user_prompt(processed_query: Any) -> str:
        return f"Question: {processed_query.normalized_query}"

    @staticmethod
    def _build_context_block(assembled_context: AssembledContext) -> str:
        citation_lines = []
        for citation in assembled_context.citations:
            page = f" page {citation.page}" if citation.page is not None else ""
            section = " > ".join(citation.header_path)
            citation_lines.append(
                f"[{citation.citation_id}] {citation.document_id}{page} {section}"
            )
        citations_text = "\n".join(citation_lines)
        return (
            f"Context:\n{assembled_context.assembled_context}\n\n"
            f"Citations:\n{citations_text}"
        )

    @staticmethod
    def _format_conversation_history(conversation_history: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
        formatted: list[str] = []
        for item in conversation_history:
            if not isinstance(item, Mapping):
                raise TypeError("conversation_history items must be mappings")
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            formatted.append(f"{role.capitalize()}: {content}")
        return tuple(formatted)

    @staticmethod
    def _compute_prompt_statistics(
        system_prompt: str,
        user_prompt: str,
        history: tuple[str, ...],
        context: str,
    ) -> dict[str, int]:
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_prompt)
        history_tokens = sum(count_tokens(item) for item in history)
        context_tokens = count_tokens(context)
        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "history_tokens": history_tokens,
            "context_tokens": context_tokens,
            "total_tokens": system_tokens + user_tokens + history_tokens + context_tokens,
        }

    @staticmethod
    def _truncate_text(text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""

        tokens = list(re.finditer(r"\S+", text))
        if len(tokens) <= token_budget:
            return text

        end = tokens[token_budget - 1].end()
        return text[:end].rstrip()

    @staticmethod
    def _extract_page(metadata: Mapping[str, Any]) -> int | None:
        page = metadata.get("page")
        if isinstance(page, bool):
            return None
        if isinstance(page, int):
            return page
        return None
