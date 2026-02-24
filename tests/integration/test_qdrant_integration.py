"""Integration tests for Qdrant vector store."""

from __future__ import annotations

import pytest

from quaerium.infra.embedding.ollama import OllamaEmbeddingClient
from quaerium.infra.vectorstores.qdrant import QdrantService


@pytest.mark.integration
def test_qdrant_full_workflow(
    qdrant_service: QdrantService,
    ollama_client: OllamaEmbeddingClient,
    test_collection_name: str,
):
    """Test complete Qdrant workflow with real services."""
    
    # Create collection
    qdrant_service.ensure_collection(
        collection_name=test_collection_name,
        vector_size=768,  # nomic-embed-text dimension
    )
    
    # Prepare documents
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
    ]
    
    # Generate embeddings
    embeddings = [ollama_client.embed(text) for text in texts]
    
    # Insert documents
    points = [
        {
            "id": f"doc-{i}",
            "vector": emb,
            "payload": {"text": text, "topic": "AI", "index": i}
        }
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    
    ids = qdrant_service.upsert(
        collection_name=test_collection_name,
        points=points
    )
    
    assert len(ids) == 3
    assert ids == ["doc-0", "doc-1", "doc-2"]
    
    # Search
    query_emb = ollama_client.embed("What is deep learning?")
    results = qdrant_service.search(
        collection_name=test_collection_name,
        query_vector=query_emb,
        limit=2
    )
    
    assert len(results) > 0
    assert results[0]["score"] > 0.5
    assert "neural networks" in results[0]["payload"]["text"].lower()
    
    # Search with filter
    filtered_results = qdrant_service.search(
        collection_name=test_collection_name,
        query_vector=query_emb,
        limit=5,
        query_filter={"topic": "AI"}
    )
    
    assert len(filtered_results) <= 3
    for result in filtered_results:
        assert result["payload"]["topic"] == "AI"
    
    # Retrieve by ID
    retrieved = qdrant_service.retrieve(
        collection_name=test_collection_name,
        ids=["doc-0"]
    )
    
    assert len(retrieved) == 1
    assert retrieved[0]["payload"]["text"] == texts[0]
    
    # Delete by ID
    qdrant_service.delete(
        collection_name=test_collection_name,
        ids=["doc-0"]
    )
    
    # Verify deletion
    after_delete = qdrant_service.retrieve(
        collection_name=test_collection_name,
        ids=["doc-0"]
    )
    assert len(after_delete) == 0
    
    # Cleanup
    qdrant_service.drop_collection(test_collection_name)


@pytest.mark.integration
def test_qdrant_batch_operations(
    qdrant_service: QdrantService,
    ollama_client: OllamaEmbeddingClient,
    test_collection_name: str,
):
    """Test Qdrant batch operations."""
    
    qdrant_service.ensure_collection(
        collection_name=test_collection_name,
        vector_size=768
    )
    
    # Batch insert
    texts = [f"Document {i} about topic {i % 3}" for i in range(10)]
    embeddings = [ollama_client.embed(text) for text in texts]
    
    points = [
        {
            "id": f"doc-{i}",
            "vector": emb,
            "payload": {"text": text, "category": i % 3}
        }
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    
    ids = qdrant_service.upsert(
        collection_name=test_collection_name,
        points=points
    )
    
    assert len(ids) == 10
    
    # Batch search
    queries = ["topic 0", "topic 1", "topic 2"]
    query_embeddings = [ollama_client.embed(q) for q in queries]
    
    batch_results = qdrant_service.batch_search(
        collection_name=test_collection_name,
        query_vectors=query_embeddings,
        limit=3
    )
    
    assert len(batch_results) == 3
    for results in batch_results:
        assert len(results) <= 3
    
    # Cleanup
    qdrant_service.drop_collection(test_collection_name)


@pytest.mark.integration
def test_qdrant_scroll(
    qdrant_service: QdrantService,
    ollama_client: OllamaEmbeddingClient,
    test_collection_name: str,
):
    """Test Qdrant scroll/pagination."""
    
    qdrant_service.ensure_collection(
        collection_name=test_collection_name,
        vector_size=768
    )
    
    # Insert documents
    texts = [f"Document {i}" for i in range(20)]
    embeddings = [ollama_client.embed(text) for text in texts]
    
    points = [
        {
            "id": f"doc-{i}",
            "vector": emb,
            "payload": {"text": text, "index": i}
        }
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    
    qdrant_service.upsert(
        collection_name=test_collection_name,
        points=points
    )
    
    # Scroll through results
    all_points = []
    offset = None
    
    while True:
        points_batch, next_offset = qdrant_service.scroll(
            collection_name=test_collection_name,
            limit=5,
            offset=offset
        )
        
        all_points.extend(points_batch)
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    assert len(all_points) == 20
    
    # Cleanup
    qdrant_service.drop_collection(test_collection_name)
