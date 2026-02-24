"""Shared fixtures for Milvus tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from quaerium.infra.vectorstores.milvus.config import MilvusConfig


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client."""
    with patch("rag_toolkit.infra.vectorstores.milvus.connection.MilvusClient") as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Mock basic methods
        client.list_collections.return_value = []
        client.has_collection.return_value = False
        client.describe_collection.return_value = {
            "collection_name": "test_collection",
            "num_entities": 0,
        }
        client.list_indexes.return_value = []
        client.prepare_index_params.return_value = MagicMock()
        client.query.return_value = [{"count(*)": 0}]
        
        yield client


@pytest.fixture
def milvus_config() -> MilvusConfig:
    """Milvus configuration for tests."""
    from quaerium.infra.vectorstores.milvus.config import MilvusConfig
    
    return MilvusConfig(
        uri="http://localhost:19530",
        user=None,
        password=None,
        db_name="default",
    )


@pytest.fixture
def mock_connection_manager(mock_milvus_client, milvus_config):
    """Mock Milvus connection manager."""
    from quaerium.infra.vectorstores.milvus.connection import MilvusConnectionManager
    
    manager = MilvusConnectionManager(milvus_config)
    manager._client = mock_milvus_client
    yield manager


@pytest.fixture
def mock_collection_manager(mock_connection_manager):
    """Mock Milvus collection manager."""
    from quaerium.infra.vectorstores.milvus.collection import MilvusCollectionManager
    
    return MilvusCollectionManager(mock_connection_manager)


@pytest.fixture
def mock_data_manager(mock_connection_manager):
    """Mock Milvus data manager."""
    from quaerium.infra.vectorstores.milvus.data import MilvusDataManager
    
    return MilvusDataManager(mock_connection_manager)


@pytest.fixture
def sample_schema():
    """Sample Milvus schema as dict."""
    return {
        "id": {"dtype": "INT64", "is_primary": True, "auto_id": False},
        "vector": {"dtype": "FLOAT_VECTOR", "dim": 768},
        "text": {"dtype": "VARCHAR", "max_length": 65535},
    }


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing."""
    import random
    
    return [
        [random.random() for _ in range(384)] for _ in range(3)
    ]


@pytest.fixture
def sample_data(sample_vectors):
    """Sample data for insertion."""
    return [
        {"id": 1, "vector": sample_vectors[0], "text": "doc1"},
        {"id": 2, "vector": sample_vectors[1], "text": "doc2"},
        {"id": 3, "vector": sample_vectors[2], "text": "doc3"},
    ]
