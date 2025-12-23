"""Tests for Milvus data manager."""

from __future__ import annotations

import pytest

from rag_toolkit.infra.vectorstores.milvus.exceptions import DataOperationError


def test_insert_data(mock_data_manager, mock_milvus_client, sample_data):
    """Test inserting data."""
    mock_milvus_client.insert.return_value = {"insert_count": 3, "ids": [1, 2, 3]}
    
    result = mock_data_manager.insert(
        collection_name="test_collection",
        data=sample_data,
    )
    
    assert result["insert_count"] == 3
    assert result["ids"] == [1, 2, 3]
    mock_milvus_client.insert.assert_called_once()


def test_insert_empty_data(mock_data_manager):
    """Test inserting empty data."""
    with pytest.raises(DataOperationError, match="empty"):
        mock_data_manager.insert(
            collection_name="test_collection",
            data=[],
        )


def test_upsert_data(mock_data_manager, mock_milvus_client, sample_data):
    """Test upserting data."""
    mock_milvus_client.upsert.return_value = {"upsert_count": 3}
    
    result = mock_data_manager.upsert(
        collection_name="test_collection",
        data=sample_data,
    )
    
    assert result["upsert_count"] == 3
    mock_milvus_client.upsert.assert_called_once()


def test_search_vectors(mock_data_manager, mock_milvus_client, sample_vectors):
    """Test searching for similar vectors."""
    mock_results = [
        [
            {"id": 1, "distance": 0.1, "entity": {"text": "doc1"}},
            {"id": 2, "distance": 0.3, "entity": {"text": "doc2"}},
        ]
    ]
    mock_milvus_client.search.return_value = mock_results
    
    results = mock_data_manager.search(
        collection_name="test_collection",
        data=[sample_vectors[0]],
        limit=2,
        output_fields=["text"],
    )
    
    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0]["id"] == 1


def test_search_with_filter(mock_data_manager, mock_milvus_client, sample_vectors):
    """Test searching with filter expression."""
    mock_results = [[{"id": 1, "distance": 0.1}]]
    mock_milvus_client.search.return_value = mock_results
    
    filter_expr = 'text == "doc1"'
    
    results = mock_data_manager.search(
        collection_name="test_collection",
        data=[sample_vectors[0]],
        limit=1,
        filter=filter_expr,
    )
    
    assert len(results[0]) == 1


def test_search_empty_query(mock_data_manager):
    """Test searching with empty query."""
    with pytest.raises(DataOperationError, match="empty"):
        mock_data_manager.search(
            collection_name="test_collection",
            data=[],
            limit=10,
        )


def test_query_by_ids(mock_data_manager, mock_milvus_client):
    """Test querying by IDs."""
    mock_results = [
        {"id": 1, "text": "doc1"},
        {"id": 2, "text": "doc2"},
    ]
    mock_milvus_client.query.return_value = mock_results
    
    results = mock_data_manager.query(
        collection_name="test_collection",
        ids=[1, 2],
        output_fields=["text"],
    )
    
    assert len(results) == 2
    assert results[0]["id"] == 1


def test_query_by_filter(mock_data_manager, mock_milvus_client):
    """Test querying by filter expression."""
    mock_results = [{"id": 1, "text": "doc1"}]
    mock_milvus_client.query.return_value = mock_results
    
    filter_expr = 'text == "doc1"'
    
    results = mock_data_manager.query(
        collection_name="test_collection",
        filter=filter_expr,
        output_fields=["text"],
    )
    
    assert len(results) == 1


def test_delete_by_ids(mock_data_manager, mock_milvus_client):
    """Test deleting by IDs."""
    mock_milvus_client.delete.return_value = {"delete_count": 2}
    
    result = mock_data_manager.delete(
        collection_name="test_collection",
        ids=[1, 2],
    )
    
    assert result["delete_count"] == 2


def test_delete_by_filter(mock_data_manager, mock_milvus_client):
    """Test deleting by filter expression."""
    mock_milvus_client.delete.return_value = {"delete_count": 1}
    
    filter_expr = 'text == "doc1"'
    
    result = mock_data_manager.delete(
        collection_name="test_collection",
        filter=filter_expr,
    )
    
    assert result["delete_count"] == 1


def test_get_by_ids(mock_data_manager, mock_milvus_client):
    """Test getting entities by IDs."""
    mock_results = [
        {"id": 1, "text": "doc1"},
        {"id": 2, "text": "doc2"},
    ]
    mock_milvus_client.get.return_value = mock_results
    
    results = mock_data_manager.get(
        collection_name="test_collection",
        ids=[1, 2],
        output_fields=["text"],
    )
    
    assert len(results) == 2
    assert results[0]["id"] == 1


def test_insert_error(mock_data_manager, mock_milvus_client, sample_data):
    """Test error handling in insert."""
    mock_milvus_client.insert.side_effect = Exception("Insert failed")
    
    with pytest.raises(DataOperationError, match="Failed to insert"):
        mock_data_manager.insert(
            collection_name="test_collection",
            data=sample_data,
        )


def test_search_error(mock_data_manager, mock_milvus_client, sample_vectors):
    """Test error handling in search."""
    mock_milvus_client.search.side_effect = Exception("Search failed")
    
    with pytest.raises(DataOperationError, match="Failed to search"):
        mock_data_manager.search(
            collection_name="test_collection",
            data=[sample_vectors[0]],
            limit=10,
        )


def test_query_error(mock_data_manager, mock_milvus_client):
    """Test error handling in query."""
    mock_milvus_client.query.side_effect = Exception("Query failed")
    
    with pytest.raises(DataOperationError, match="Failed to query"):
        mock_data_manager.query(
            collection_name="test_collection",
            ids=[1],
        )


def test_delete_error(mock_data_manager, mock_milvus_client):
    """Test error handling in delete."""
    mock_milvus_client.delete.side_effect = Exception("Delete failed")
    
    with pytest.raises(DataOperationError, match="Failed to delete"):
        mock_data_manager.delete(
            collection_name="test_collection",
            ids=[1],
        )
