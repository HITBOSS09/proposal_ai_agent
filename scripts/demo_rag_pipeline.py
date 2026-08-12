#!/usr/bin/env python3
"""Run the complete BDIL online RAG pipeline against local Qdrant and Ollama.

Environment configuration (loaded from ``.env`` when present):
``BDIL_QDRANT_URL`` (default ``http://localhost:6333``),
``BDIL_OLLAMA_URL`` (default ``http://localhost:11434/api``),
``BDIL_EMBEDDING_MODEL`` (default ``bge-m3``), and ``BDIL_OLLAMA_MODEL``
(default ``qwen2.5:3b``).  The embedding model must be the model used when
the selected Qdrant collection was indexed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proposal_ai_agent.online.engines.query_engine import QueryEngine
from proposal_ai_agent.online.engines.response_builder import ResponseBuilder
from proposal_ai_agent.online.engines.response_engine import ResponseEngine
from proposal_ai_agent.online.engines.retrieval_engine import RetrievalEngine
from proposal_ai_agent.online.engines.synthesis_engine import SynthesisEngine
from proposal_ai_agent.online.providers.embedding import OllamaEmbeddingProvider
from proposal_ai_agent.online.providers.llm.ollama import OllamaProvider
from proposal_ai_agent.online.providers.llm.router import LLMRouter, NoAvailableProviderError
from proposal_ai_agent.online.repositories import QdrantVectorRepository


class DemoPipelineError(RuntimeError):
    """A user-facing failure while running the real demo pipeline."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query an indexed BDIL collection with local Ollama.")
    parser.add_argument("--collection", required=True, help="Existing Qdrant collection name")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=None, help="Number of evidence chunks to use")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline diagnostics")
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def _collection_dimension(collection: Any) -> int:
    vectors = collection.config.params.vectors
    size = getattr(vectors, "size", None)
    if isinstance(size, int) and size > 0:
        return size
    raise DemoPipelineError("collection uses named vectors; this demo requires one unnamed dense vector")


def _override_top_k(request: Any, top_k: int | None) -> Any:
    if top_k is None:
        return request
    return replace(request, top_k=top_k, candidate_budget=max(request.candidate_budget, top_k))


def _prompt_text(prompt_package: Any) -> str:
    history = "\n".join(prompt_package.conversation_history)
    return "\n\n".join(
        item for item in (prompt_package.system_prompt, history, prompt_package.assembled_context.assembled_context, prompt_package.user_prompt) if item
    )


def _document_title(candidate: Any) -> str:
    metadata = candidate.metadata
    return str(metadata.get("document_title") or metadata.get("title") or metadata.get("source_file") or candidate.document_id)


def print_results(query: str, intent: str, embedding_model: str, retrieved_context: Any, prompt_package: Any, answer: Any, total_ms: float, verbose: bool) -> None:
    print("=" * 54)
    print("Query\n" + query)
    print("\nIntent\n" + intent)
    print("\nEmbedding Model\n" + embedding_model)
    print("\n" + "-" * 54)
    print("Retrieved Documents")
    for candidate in retrieved_context.candidates:
        print(f"Chunk ID: {candidate.chunk_id}")
        print(f"Document Title: {_document_title(candidate)}")
        print(f"Section: {' > '.join(candidate.header_path) or 'Unsectioned'}")
        print(f"Relevance Score: {candidate.score:.4f}")
        if verbose:
            print(f"Excerpt: {candidate.text[:300]}")
    print("\n" + "-" * 54)
    print("Prompt\n" + _prompt_text(prompt_package))
    statistics = prompt_package.prompt_statistics
    print(f"\nPrompt Tokens: {statistics['total_tokens']}")
    print(f"Context Tokens: {statistics['context_tokens']}")
    print("Project Facts Loaded: 0 (no project-facts source is configured)")
    print("\n" + "-" * 54)
    metadata = answer.response_metadata
    print("LLM")
    print(f"Provider: {metadata.provider_name}")
    print(f"Model: {metadata.model_name}")
    print(f"Latency: {metadata.latency_ms:.1f} ms")
    print(f"Completion Tokens: {metadata.completion_tokens}")
    print(f"Finish Reason: {metadata.finish_reason}")
    print("\n" + "-" * 54)
    print("ANSWER\n\n" + answer.final_answer)
    print("\n" + "-" * 54)
    print("SOURCES")
    for citation in answer.citations:
        print(f"Document: {citation.document_name or citation.document_id}")
        print(f"Section: {citation.section or 'Unsectioned'}")
        print(f"Chunk ID: {citation.chunk_id}")
    print("=" * 54)
    print(f"TOTAL LATENCY: {total_ms:.1f} ms")


def run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    qdrant_url = os.getenv("BDIL_QDRANT_URL", "http://localhost:6333")
    ollama_url = os.getenv("BDIL_OLLAMA_URL", "http://localhost:11434/api")
    embedding_model = os.getenv("BDIL_EMBEDDING_MODEL", "bge-m3")
    llm_model = os.getenv("BDIL_OLLAMA_MODEL", "qwen2.5:3b")
    timeout = int(os.getenv("BDIL_PROVIDER_TIMEOUT", "60"))
    started_at = perf_counter()
    try:
        client = QdrantClient(url=qdrant_url, timeout=timeout)
        if not client.collection_exists(args.collection):
            raise DemoPipelineError(f"Qdrant collection '{args.collection}' does not exist. Index a proposal before running the demo.")
        dimension = _collection_dimension(client.get_collection(args.collection))
        embedding_provider = OllamaEmbeddingProvider(embedding_model, ollama_url, timeout)
        query_engine = QueryEngine(embedding_provider=embedding_provider, embedding_dimension=dimension, embedding_model_id=embedding_model)
        retrieval_engine = RetrievalEngine(QdrantVectorRepository(client, args.collection, timeout))
        synthesis_engine = SynthesisEngine()
        response_engine = ResponseEngine(LLMRouter((OllamaProvider(base_url=ollama_url, model=llm_model, timeout=timeout),)))

        user_query = query_engine.receive_query(args.query)
        qualified_query = query_engine.qualify_query(user_query)
        processed_query = query_engine.process_query(qualified_query)
        query_embedding = query_engine.embed_query(processed_query)
        retrieval_request = _override_top_k(query_engine.plan_retrieval(query_embedding), args.top_k)
        retrieved = retrieval_engine.retrieve(retrieval_request)
        processed = retrieval_engine.process(retrieved)
        retrieved_context = retrieval_engine.rerank(processed)
        if not retrieved_context.candidates:
            raise DemoPipelineError("No relevant chunks were retrieved from the collection.")
        assembled_context = synthesis_engine.assemble(retrieved_context)
        prompt_package = synthesis_engine.build_prompt(assembled_context)
        generated = response_engine.generate(prompt_package)
        answer = ResponseBuilder().build_response(prompt_package, generated)
    except DemoPipelineError:
        raise
    except NoAvailableProviderError as error:
        raise DemoPipelineError(f"Ollama is unavailable or returned invalid output: {error}") from error
    except TimeoutError as error:
        raise DemoPipelineError(f"Provider request timed out: {error}") from error
    except ValueError as error:
        raise DemoPipelineError(f"Pipeline returned invalid data: {error}") from error
    except Exception as error:
        raise DemoPipelineError(f"Pipeline failed: {error}") from error

    print_results(args.query, qualified_query.intent, query_embedding.model_id, retrieved_context, prompt_package, answer, (perf_counter() - started_at) * 1000, args.verbose)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except DemoPipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
