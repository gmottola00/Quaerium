"""Test core protocol compliance and imports."""

from __future__ import annotations

import pytest

from quaerium.core.embedding import EmbeddingClient
from quaerium.core.llm import LLMClient
from quaerium.core.vectorstore import VectorStoreClient


def test_embedding_protocol_compliance(mock_embedding):
    """Test that mock embedding client satisfies EmbeddingClient protocol."""
    # Check required methods exist (duck typing)
    assert hasattr(mock_embedding, "embed")
    assert hasattr(mock_embedding, "embed_batch")
    assert hasattr(mock_embedding, "model_name")
    assert hasattr(mock_embedding, "dimension")
    assert callable(mock_embedding.embed)
    assert callable(mock_embedding.embed_batch)
    
    # Test basic functionality
    embedding = mock_embedding.embed("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == mock_embedding.dimension
    assert all(isinstance(x, (int, float)) for x in embedding)


def test_llm_protocol_compliance(mock_llm):
    """Test that mock LLM client satisfies LLMClient protocol."""
    # Check required methods exist (duck typing)
    assert hasattr(mock_llm, "generate")
    assert hasattr(mock_llm, "generate_batch")
    assert hasattr(mock_llm, "model_name")
    assert callable(mock_llm.generate)
    assert callable(mock_llm.generate_batch)
    
    # Test basic functionality
    response = mock_llm.generate("What is RAG?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_vectorstore_protocol_compliance(mock_vectorstore):
    """Test that mock vector store satisfies VectorStoreClient protocol."""
    # Check required methods exist (duck typing)
    assert hasattr(mock_vectorstore, "create_collection")
    assert hasattr(mock_vectorstore, "insert")
    assert hasattr(mock_vectorstore, "search")
    assert hasattr(mock_vectorstore, "list_collections")
    assert hasattr(mock_vectorstore, "delete_collection")
    assert callable(mock_vectorstore.create_collection)
    assert callable(mock_vectorstore.insert)
    assert callable(mock_vectorstore.search)
    
    # Test basic functionality
    mock_vectorstore.create_collection("test", dimension=384)
    assert "test" in mock_vectorstore.list_collections()


def test_embedding_batch(mock_embedding, sample_chunks):
    """Test batch embedding functionality."""
    embeddings = mock_embedding.embed_batch(sample_chunks)
    
    assert len(embeddings) == len(sample_chunks)
    assert all(len(emb) == mock_embedding.dimension for emb in embeddings)
    
    # Embeddings should be deterministic for same text
    emb1 = mock_embedding.embed(sample_chunks[0])
    emb2 = mock_embedding.embed(sample_chunks[0])
    assert emb1 == emb2


def test_llm_batch(mock_llm, sample_chunks):
    """Test batch generation functionality."""
    responses = list(mock_llm.generate_batch(sample_chunks))
    
    assert len(responses) == len(sample_chunks)
    assert all(isinstance(r, str) for r in responses)
    assert all(len(r) > 0 for r in responses)


def test_vectorstore_insert_search(mock_vectorstore, mock_embedding, sample_chunks):
    """Test vector store insert and search operations."""
    # Create collection
    mock_vectorstore.create_collection("docs", dimension=384, metric="IP")
    
    # Generate embeddings
    vectors = mock_embedding.embed_batch(sample_chunks)
    metadata = [{"source": f"doc_{i}"} for i in range(len(sample_chunks))]
    
    # Insert
    ids = mock_vectorstore.insert(
        collection_name="docs",
        vectors=vectors,
        texts=sample_chunks,
        metadata=metadata,
    )
    
    assert len(ids) == len(sample_chunks)
    
    # Search
    query_vector = mock_embedding.embed("machine learning")
    results = mock_vectorstore.search(
        collection_name="docs", query_vector=query_vector, top_k=2
    )
    
    assert len(results) <= 2
    assert all("id" in r for r in results)
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_core_imports():
    """Test that all core imports work correctly."""
    # Core protocols
    from quaerium import EmbeddingClient, LLMClient, VectorStoreClient
    
    assert EmbeddingClient is not None
    assert LLMClient is not None
    assert VectorStoreClient is not None
    
    # Chunking types
    from quaerium import Chunk, TokenChunk
    
    assert Chunk is not None
    assert TokenChunk is not None
    
    # RAG components
    from quaerium import RagPipeline, RagResponse
    
    assert RagPipeline is not None
    assert RagResponse is not None


def test_lazy_import_helpers():
    """Test lazy import helper functions."""
    from quaerium import (
        get_ollama_embedding,
        get_ollama_llm,
        get_openai_embedding,
        get_openai_llm,
    )
    
    # These should be callable
    assert callable(get_ollama_embedding)
    assert callable(get_ollama_llm)
    assert callable(get_openai_embedding)
    assert callable(get_openai_llm)
    
    # They should raise ImportError if dependencies not installed
    # (unless ollama/openai are actually installed)
    # We can't test the actual import without installing dependencies


def test_version():
    """Test that version is accessible."""
    import quaerium
    
    assert hasattr(rag_toolkit, "__version__")
    assert isinstance(rag_toolkit.__version__, str)
    assert len(rag_toolkit.__version__) > 0
    # Version should follow semantic versioning pattern
    assert rag_toolkit.__version__.count(".") >= 2
