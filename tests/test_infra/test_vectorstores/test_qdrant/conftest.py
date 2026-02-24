"""Pytest fixtures for Qdrant tests."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, Mock

import pytest

from quaerium.infra.vectorstores.qdrant.config import QdrantConfig, QdrantIndexConfig
from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager


@pytest.fixture
def qdrant_config() -> QdrantConfig:
    """Provide test Qdrant configuration."""
    return QdrantConfig(
        url="http://localhost:6333",
        api_key=None,
        timeout=30.0,
    )


@pytest.fixture
def qdrant_index_config() -> QdrantIndexConfig:
    """Provide test Qdrant index configuration."""
    return QdrantIndexConfig(
        distance="Cosine",
        hnsw_config={"m": 16, "ef_construct": 100},
    )


@pytest.fixture
def mock_qdrant_client() -> Mock:
    """Provide mock Qdrant client."""
    mock_client = MagicMock()
    
    # Mock collection operations
    mock_client.collection_exists.return_value = False
    mock_client.get_collections.return_value = Mock(collections=[])
    mock_client.create_collection.return_value = None
    mock_client.delete_collection.return_value = None
    
    # Mock data operations
    mock_client.upsert.return_value = Mock(status="completed")
    mock_client.search.return_value = []
    mock_client.search_batch.return_value = [[]]
    mock_client.retrieve.return_value = []
    mock_client.delete.return_value = None
    mock_client.scroll.return_value = ([], None)
    
    # Mock collection info
    mock_client.get_collection.return_value = Mock(
        vectors_count=0,
        points_count=0,
        status="green",
        config=Mock(params=Mock(vectors=Mock(size=384, distance="Cosine"))),
    )
    
    return mock_client


@pytest.fixture
def mock_connection(
    qdrant_config: QdrantConfig,
    mock_qdrant_client: Mock,
) -> Generator[QdrantConnectionManager, None, None]:
    """Provide mock Qdrant connection manager."""
    connection = QdrantConnectionManager(qdrant_config)
    connection._client = mock_qdrant_client
    
    yield connection
    
    connection.close()


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    """Provide sample vectors for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_points() -> list[dict]:
    """Provide sample points for testing."""
    return [
        {
            "id": "point-1",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"text": "First document", "source": "test"},
        },
        {
            "id": "point-2",
            "vector": [0.5, 0.6, 0.7, 0.8],
            "payload": {"text": "Second document", "source": "test"},
        },
        {
            "id": "point-3",
            "vector": [0.9, 0.8, 0.7, 0.6],
            "payload": {"text": "Third document", "source": "test"},
        },
    ]


@pytest.fixture
def sample_metadata() -> list[dict]:
    """Provide sample metadata for testing."""
    return [
        {"source": "doc1.pdf", "page": 1, "category": "tech"},
        {"source": "doc2.pdf", "page": 2, "category": "science"},
        {"source": "doc3.pdf", "page": 3, "category": "tech"},
    ]
