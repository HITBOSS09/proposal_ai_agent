"""Integration coverage for the Qdrant retrieval adapter."""

from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
from proposal_ai_agent.online.engines.query_engine import QueryEngine
from proposal_ai_agent.online.engines.retrieval_engine import RetrievalEngine
from proposal_ai_agent.online.repositories import QdrantVectorRepository


def test_qdrant_repository_executes_a_real_dense_search_and_hydrates_chunk() -> None:
    client = QdrantClient(":memory:")
    collection_name = "bdil_demo"
    provider = MockEmbeddingProvider(dimensions=3)
    query_engine = QueryEngine(
        embedding_provider=provider,
        embedding_dimension=3,
        embedding_model_id="test-embedding-model",
    )
    query_embedding = query_engine.embed_query(
        query_engine.process_query(query_engine.qualify_query(query_engine.receive_query("cybersecurity requirements")))
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    point_id = str(uuid4())
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=list(query_embedding.vector),
                payload={
                    "chunk_id": "chunk-content-id",
                    "original_text": "Cybersecurity controls are mandatory.",
                    "embedding_text": "Document: Proposal\n\nCybersecurity controls are mandatory.",
                    "source_document": "proposal",
                    "page_number": 4,
                    "document_id": "proposal-1",
                    "document": {"document_id": "proposal-1"},
                    "section": {"section_path": ["Security", "Cybersecurity"]},
                    "chunk": {"chunk_index": 4},
                    "source_file": "proposal.docx",
                },
            )
        ],
        wait=True,
    )

    retrieved = RetrievalEngine(QdrantVectorRepository(client, collection_name)).retrieve(
        query_engine.plan_retrieval(query_embedding)
    )

    assert retrieved.candidate_count == 1
    candidate = retrieved.candidates[0]
    assert candidate.chunk_id == "chunk-content-id"
    assert candidate.point_id == point_id
    assert candidate.page_number == 4
    assert candidate.document_id == "proposal-1"
    assert candidate.header_path == ("Security", "Cybersecurity")
    assert candidate.chunk_index == 4
    assert candidate.text == "Cybersecurity controls are mandatory."


def test_qdrant_repository_maps_document_filter_to_indexed_source_document() -> None:
    client = QdrantClient(":memory:")
    collection_name = "bdil_demo"
    provider = MockEmbeddingProvider(dimensions=3)
    query_engine = QueryEngine(
        embedding_provider=provider,
        embedding_dimension=3,
        embedding_model_id="test-embedding-model",
    )
    query_embedding = query_engine.embed_query(
        query_engine.process_query(
            query_engine.qualify_query(
                query_engine.receive_query("cybersecurity requirements, document: proposal")
            )
        )
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=list(query_embedding.vector),
                payload={
                    "chunk_id": "chunk-content-id",
                    "original_text": "Cybersecurity controls are mandatory.",
                    "source_document": "proposal",
                    "document_id": "proposal-1",
                    "section_path": ["Security"],
                    "chunk_index": 0,
                },
            )
        ],
        wait=True,
    )

    retrieved = RetrievalEngine(QdrantVectorRepository(client, collection_name)).retrieve(
        query_engine.plan_retrieval(query_embedding)
    )

    assert retrieved.candidate_count == 1
