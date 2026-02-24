"""Pytest fixtures for ChromaDB tests."""

from __future__ import annotations

from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock

import pytest

from quaerium.infra.vectorstores.chroma.config import ChromaConfig, ChromaIndexConfig
from quaerium.infra.vectorstores.chroma.connection import ChromaConnectionManager


@pytest.fixture
def chroma_config() -> ChromaConfig:
    """Provide test ChromaDB configuration (in-memory)."""
    return ChromaConfig()


@pytest.fixture
def chroma_persistent_config(tmp_path) -> ChromaConfig:
    """Provide test ChromaDB configuration with persistent storage."""
    return ChromaConfig(path=str(tmp_path / "chroma_test"))


@pytest.fixture
def chroma_remote_config() -> ChromaConfig:
    """Provide test ChromaDB configuration for remote server."""
    return ChromaConfig(
        host="localhost",
        port=8000,
        ssl=False,
    )


@pytest.fixture
def chroma_index_config() -> ChromaIndexConfig:
    """Provide test ChromaDB index configuration."""
    return ChromaIndexConfig(
        distance="cosine",
        hnsw_space="cosine",
        hnsw_m=16,
        hnsw_ef_construction=100,
        hnsw_ef_search=50,
    )


@pytest.fixture
def mock_chroma_collection() -> Mock:
    """Provide mock ChromaDB collection."""
    mock_collection = MagicMock()
    
    # Collection properties
    mock_collection.name = "test_collection"
    mock_collection.metadata = {"hnsw:space": "cosine"}
    mock_collection.count.return_value = 0
    
    # Data operations
    mock_collection.add.return_value = None
    mock_collection.upsert.return_value = None
    mock_collection.query.return_value = {
        "ids": [[]],
        "distances": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }
    mock_collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
    }
    mock_collection.update.return_value = None
    mock_collection.delete.return_value = None
    
    return mock_collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection: Mock) -> Mock:
    """Provide mock ChromaDB client."""
    mock_client = MagicMock()
    
    # Collection operations
    mock_client.get_collection.return_value = mock_chroma_collection
    mock_client.create_collection.return_value = mock_chroma_collection
    mock_client.get_or_create_collection.return_value = mock_chroma_collection
    mock_client.delete_collection.return_value = None
    mock_client.list_collections.return_value = []
    
    # Health check
    mock_client.heartbeat.return_value = 1234567890
    
    return mock_client


@pytest.fixture
def mock_connection(
    chroma_config: ChromaConfig,
    mock_chroma_client: Mock,
) -> Generator[ChromaConnectionManager, None, None]:
    """Provide mock ChromaDB connection manager."""
    connection = ChromaConnectionManager(chroma_config)
    connection._client = mock_chroma_client
    
    yield connection
    
    connection.close()


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Provide sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_documents() -> List[str]:
    """Provide sample documents for testing."""
    return [
        "This is the first test document",
        "This is the second test document",
        "This is the third test document",
    ]


@pytest.fixture
def sample_metadatas() -> List[Dict[str, Any]]:
    """Provide sample metadata for testing."""
    return [
        {"source": "test", "page": 1, "category": "A"},
        {"source": "test", "page": 2, "category": "B"},
        {"source": "test", "page": 3, "category": "A"},
    ]


@pytest.fixture
def sample_ids() -> List[str]:
    """Provide sample document IDs for testing."""
    return ["doc-1", "doc-2", "doc-3"]
