"""Tests for Milvus collection manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_toolkit.infra.vectorstores.milvus.exceptions import CollectionError


def test_ensure_collection_creates_if_not_exists(
    mock_collection_manager, mock_milvus_client, sample_schema
):
    """Test ensure_collection creates collection if it doesn't exist."""
    mock_milvus_client.has_collection.return_value = False
    
    mock_collection_manager.ensure_collection(
        name="test_collection",
        schema=sample_schema,
    )
    
    mock_milvus_client.create_collection.assert_called_once()
    mock_milvus_client.load_collection.assert_called_once()


def test_ensure_collection_skips_if_exists(
    mock_collection_manager, mock_milvus_client, sample_schema
):
    """Test ensure_collection skips creation if collection exists."""
    mock_milvus_client.has_collection.return_value = True
    
    mock_collection_manager.ensure_collection(
        name="test_collection",
        schema=sample_schema,
    )
    
    mock_milvus_client.create_collection.assert_not_called()


def test_ensure_collection_with_index_params(
    mock_collection_manager, mock_milvus_client, sample_schema
):
    """Test ensure_collection with custom index params."""
    mock_milvus_client.has_collection.return_value = False
    mock_index_params = MagicMock()
    mock_milvus_client.prepare_index_params.return_value = mock_index_params
    
    index_params = {
        "field_name": "vector",
        "index_type": "HNSW",
        "metric_type": "L2",
        "M": 16,
    }
    
    mock_collection_manager.ensure_collection(
        name="test_collection",
        schema=sample_schema,
        index_params=index_params,
    )
    
    mock_milvus_client.create_index.assert_called_once()


def test_ensure_collection_without_load(
    mock_collection_manager, mock_milvus_client, sample_schema
):
    """Test ensure_collection without loading."""
    mock_milvus_client.has_collection.return_value = False
    
    mock_collection_manager.ensure_collection(
        name="test_collection",
        schema=sample_schema,
        load=False,
    )
    
    mock_milvus_client.load_collection.assert_not_called()


def test_drop_collection_exists(mock_collection_manager, mock_milvus_client):
    """Test dropping existing collection."""
    mock_milvus_client.has_collection.return_value = True
    
    mock_collection_manager.drop_collection("test_collection")
    
    mock_milvus_client.drop_collection.assert_called_once_with(
        collection_name="test_collection"
    )


def test_drop_collection_not_exists(mock_collection_manager, mock_milvus_client):
    """Test dropping non-existent collection (should not raise)."""
    mock_milvus_client.has_collection.return_value = False
    
    # Should not raise error
    mock_collection_manager.drop_collection("test_collection")
    
    mock_milvus_client.drop_collection.assert_not_called()


def test_load_collection(mock_collection_manager, mock_milvus_client):
    """Test loading collection into memory."""
    mock_collection_manager.load("test_collection")
    
    mock_milvus_client.load_collection.assert_called_once_with(
        collection_name="test_collection"
    )


def test_release_collection(mock_collection_manager, mock_milvus_client):
    """Test releasing collection from memory."""
    mock_collection_manager.release("test_collection")
    
    mock_milvus_client.release_collection.assert_called_once_with(
        collection_name="test_collection"
    )


def test_create_index(mock_collection_manager, mock_milvus_client):
    """Test creating index on field."""
    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16},
    }
    
    mock_collection_manager.create_index(
        name="test_collection",
        field_name="vector",
        index_params=index_params,
    )
    
    mock_milvus_client.create_index.assert_called_once()


def test_ensure_collection_error(mock_collection_manager, mock_milvus_client, sample_schema):
    """Test error handling in ensure_collection."""
    mock_milvus_client.has_collection.side_effect = Exception("Connection error")
    
    with pytest.raises(CollectionError, match="Failed to ensure collection"):
        mock_collection_manager.ensure_collection(
            name="test_collection",
            schema=sample_schema,
        )


def test_drop_collection_error(mock_collection_manager, mock_milvus_client):
    """Test error handling in drop_collection."""
    mock_milvus_client.has_collection.return_value = True
    mock_milvus_client.drop_collection.side_effect = Exception("Permission denied")
    
    with pytest.raises(CollectionError, match="Failed to drop collection"):
        mock_collection_manager.drop_collection("test_collection")


def test_load_error(mock_collection_manager, mock_milvus_client):
    """Test error handling in load."""
    mock_milvus_client.load_collection.side_effect = Exception("Memory error")
    
    with pytest.raises(CollectionError, match="Failed to load collection"):
        mock_collection_manager.load("test_collection")


def test_release_error(mock_collection_manager, mock_milvus_client):
    """Test error handling in release."""
    mock_milvus_client.release_collection.side_effect = Exception("Not loaded")
    
    with pytest.raises(CollectionError, match="Failed to release collection"):
        mock_collection_manager.release("test_collection")


def test_create_index_error(mock_collection_manager, mock_milvus_client):
    """Test error handling in create_index."""
    mock_milvus_client.create_index.side_effect = Exception("Invalid params")
    
    with pytest.raises(CollectionError, match="Failed to create index"):
        mock_collection_manager.create_index(
            name="test_collection",
            field_name="vector",
            index_params={},
        )
