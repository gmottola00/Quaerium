"""Tests for Milvus service facade."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quaerium.infra.vectorstores.milvus import MilvusService
from quaerium.infra.vectorstores.milvus.config import MilvusConfig


@pytest.fixture
def milvus_service(mock_milvus_client):
    """Milvus service instance."""
    config = MilvusConfig(uri="http://localhost:19530")
    
    with patch("rag_toolkit.infra.vectorstores.milvus.connection.MilvusClient") as mock:
        mock.return_value = mock_milvus_client
        service = MilvusService(config=config)
        yield service


def test_service_initialization(milvus_service):
    """Test service initialization."""
    assert milvus_service.connection is not None
    assert milvus_service.collections is not None
    assert milvus_service.data is not None
    assert milvus_service.databases is not None
    assert milvus_service.explorer is not None


def test_ensure_collection(milvus_service, mock_milvus_client, sample_schema):
    """Test ensuring collection through service."""
    mock_milvus_client.has_collection.return_value = False
    
    milvus_service.ensure_collection("test_collection", sample_schema)
    
    mock_milvus_client.create_collection.assert_called_once()


def test_drop_collection(milvus_service, mock_milvus_client):
    """Test dropping collection through service."""
    mock_milvus_client.has_collection.return_value = True
    
    milvus_service.drop_collection("test_collection")
    
    mock_milvus_client.drop_collection.assert_called_once()


def test_insert_data(milvus_service, mock_milvus_client, sample_data):
    """Test inserting data through service."""
    mock_milvus_client.insert.return_value = {"insert_count": 3, "ids": [1, 2, 3]}
    
    result = milvus_service.insert("test_collection", sample_data)
    
    assert result["insert_count"] == 3


def test_search_vectors(milvus_service, mock_milvus_client, sample_vectors):
    """Test searching through service."""
    mock_results = [[{"id": 1, "distance": 0.1}]]
    mock_milvus_client.search.return_value = mock_results
    
    results = milvus_service.search(
        collection_name="test_collection",
        vectors=[sample_vectors[0]],
        anns_field="vector",
        param={"metric_type": "L2"},
        limit=1,
    )
    
    assert len(results[0]) == 1


def test_ensure_database(milvus_service, mock_milvus_client):
    """Test ensuring database through service."""
    mock_milvus_client.list_databases = lambda: []
    mock_milvus_client.create_database = lambda name: None
    
    # Should not raise
    milvus_service.ensure_database("test_db")
