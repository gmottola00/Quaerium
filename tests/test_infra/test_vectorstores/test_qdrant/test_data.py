"""Tests for Qdrant data operations."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager
from quaerium.infra.vectorstores.qdrant.data import QdrantDataManager
from quaerium.infra.vectorstores.qdrant.exceptions import DataOperationError


def test_data_manager_initialization(mock_connection: QdrantConnectionManager) -> None:
    """Test data manager initialization."""
    manager = QdrantDataManager(mock_connection)
    
    assert manager.connection is mock_connection


def test_upsert_points(
    mock_connection: QdrantConnectionManager,
    sample_points: list[dict],
) -> None:
    """Test upserting points."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.PointStruct") as mock_point_struct:
        mock_point_struct.side_effect = lambda **kwargs: Mock(**kwargs)
        
        ids = manager.upsert(
            collection_name="test_collection",
            points=sample_points,
        )
        
        assert len(ids) == len(sample_points)
        assert ids == ["point-1", "point-2", "point-3"]
        mock_connection.client.upsert.assert_called_once()


def test_upsert_points_without_ids(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test upserting points without explicit IDs."""
    manager = QdrantDataManager(mock_connection)
    
    points = [
        {"vector": vec, "payload": {"text": f"Doc {i}"}}
        for i, vec in enumerate(sample_vectors)
    ]
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.PointStruct"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.uuid4") as mock_uuid:
        
        mock_uuid.side_effect = lambda: Mock(__str__=lambda self: f"uuid-{mock_uuid.call_count}")
        
        ids = manager.upsert(
            collection_name="test_collection",
            points=points,
        )
        
        assert len(ids) == len(points)
        assert all(isinstance(id, str) for id in ids)


def test_upsert_error(mock_connection: QdrantConnectionManager) -> None:
    """Test upsert error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.upsert.side_effect = RuntimeError("Upsert failed")
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.PointStruct"):
        with pytest.raises(DataOperationError, match="Failed to upsert points"):
            manager.upsert(
                collection_name="test_collection",
                points=[{"id": "test", "vector": [0.1, 0.2], "payload": {}}],
            )


def test_search(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test vector search."""
    manager = QdrantDataManager(mock_connection)
    
    # Mock search results
    mock_results = [
        Mock(id="point-1", score=0.95, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "First"}),
        Mock(id="point-2", score=0.85, vector=[0.5, 0.6, 0.7, 0.8], payload={"text": "Second"}),
    ]
    mock_connection.client.search.return_value = mock_results
    
    results = manager.search(
        collection_name="test_collection",
        query_vector=sample_vectors[0],
        limit=10,
    )
    
    assert len(results) == 2
    assert results[0]["id"] == "point-1"
    assert results[0]["score"] == 0.95
    assert results[1]["id"] == "point-2"
    mock_connection.client.search.assert_called_once()


def test_search_with_filter(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test search with metadata filter."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.search.return_value = []
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.QdrantFilter"):
        results = manager.search(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=5,
            query_filter={"source": "test.pdf"},
        )
        
        assert isinstance(results, list)
        mock_connection.client.search.assert_called_once()


def test_search_with_score_threshold(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test search with score threshold."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.search.return_value = []
    
    results = manager.search(
        collection_name="test_collection",
        query_vector=sample_vectors[0],
        limit=10,
        score_threshold=0.8,
    )
    
    assert isinstance(results, list)
    call_kwargs = mock_connection.client.search.call_args[1]
    assert call_kwargs["score_threshold"] == 0.8


def test_search_error(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test search error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.search.side_effect = RuntimeError("Search failed")
    
    with pytest.raises(DataOperationError, match="Failed to search"):
        manager.search(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=10,
        )


def test_batch_search(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test batch search."""
    manager = QdrantDataManager(mock_connection)
    
    mock_results = [
        [Mock(id="p1", score=0.9, vector=[0.1], payload={})],
        [Mock(id="p2", score=0.8, vector=[0.2], payload={})],
    ]
    mock_connection.client.search_batch.return_value = mock_results
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.SearchRequest"):
        results = manager.batch_search(
            collection_name="test_collection",
            query_vectors=sample_vectors[:2],
            limit=5,
        )
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0]["id"] == "p1"


def test_batch_search_error(
    mock_connection: QdrantConnectionManager,
    sample_vectors: list[list[float]],
) -> None:
    """Test batch search error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.search_batch.side_effect = RuntimeError("Batch search failed")
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.SearchRequest"):
        with pytest.raises(DataOperationError, match="Failed to batch search"):
            manager.batch_search(
                collection_name="test_collection",
                query_vectors=sample_vectors,
                limit=10,
            )


def test_retrieve_by_ids(mock_connection: QdrantConnectionManager) -> None:
    """Test retrieving points by IDs."""
    manager = QdrantDataManager(mock_connection)
    
    mock_results = [
        Mock(id="point-1", vector=[0.1, 0.2], payload={"text": "First"}),
        Mock(id="point-2", vector=[0.3, 0.4], payload={"text": "Second"}),
    ]
    mock_connection.client.retrieve.return_value = mock_results
    
    results = manager.retrieve(
        collection_name="test_collection",
        ids=["point-1", "point-2"],
    )
    
    assert len(results) == 2
    assert results[0]["id"] == "point-1"
    assert results[1]["id"] == "point-2"
    mock_connection.client.retrieve.assert_called_once_with(
        collection_name="test_collection",
        ids=["point-1", "point-2"],
    )


def test_retrieve_error(mock_connection: QdrantConnectionManager) -> None:
    """Test retrieve error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.retrieve.side_effect = RuntimeError("Retrieve failed")
    
    with pytest.raises(DataOperationError, match="Failed to retrieve points"):
        manager.retrieve(
            collection_name="test_collection",
            ids=["point-1"],
        )


def test_delete_by_ids(mock_connection: QdrantConnectionManager) -> None:
    """Test deleting points by IDs."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.PointIdsList") as mock_ids_list:
        mock_selector = Mock()
        mock_ids_list.return_value = mock_selector
        
        manager.delete(
            collection_name="test_collection",
            ids=["point-1", "point-2"],
        )
        
        mock_ids_list.assert_called_once_with(points=["point-1", "point-2"])
        mock_connection.client.delete.assert_called_once_with(
            collection_name="test_collection",
            points_selector=mock_selector,
        )


def test_delete_by_filter(mock_connection: QdrantConnectionManager) -> None:
    """Test deleting points by filter."""
    manager = QdrantDataManager(mock_connection)
    
    manager.delete(
        collection_name="test_collection",
        points_selector={"source": "old.pdf"},
    )
    
    mock_connection.client.delete.assert_called_once()


def test_delete_without_selector_or_ids(mock_connection: QdrantConnectionManager) -> None:
    """Test delete without IDs or selector raises error."""
    manager = QdrantDataManager(mock_connection)
    
    with pytest.raises(DataOperationError, match="Either 'ids' or 'points_selector' must be provided"):
        manager.delete(collection_name="test_collection")


def test_delete_error(mock_connection: QdrantConnectionManager) -> None:
    """Test delete error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.delete.side_effect = RuntimeError("Delete failed")
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.PointIdsList"):
        with pytest.raises(DataOperationError, match="Failed to delete points"):
            manager.delete(
                collection_name="test_collection",
                ids=["point-1"],
            )


def test_scroll(mock_connection: QdrantConnectionManager) -> None:
    """Test scrolling through points."""
    manager = QdrantDataManager(mock_connection)
    
    mock_points = [
        Mock(id="p1", vector=[0.1], payload={"text": "First"}),
        Mock(id="p2", vector=[0.2], payload={"text": "Second"}),
    ]
    mock_connection.client.scroll.return_value = (mock_points, "next-offset")
    
    points, next_offset = manager.scroll(
        collection_name="test_collection",
        limit=10,
    )
    
    assert len(points) == 2
    assert points[0]["id"] == "p1"
    assert next_offset == "next-offset"
    mock_connection.client.scroll.assert_called_once()


def test_scroll_with_filter(mock_connection: QdrantConnectionManager) -> None:
    """Test scrolling with filter."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.scroll.return_value = ([], None)
    
    points, next_offset = manager.scroll(
        collection_name="test_collection",
        limit=10,
        query_filter={"category": "tech"},
    )
    
    assert isinstance(points, list)
    mock_connection.client.scroll.assert_called_once()


def test_scroll_error(mock_connection: QdrantConnectionManager) -> None:
    """Test scroll error handling."""
    manager = QdrantDataManager(mock_connection)
    mock_connection.client.scroll.side_effect = RuntimeError("Scroll failed")
    
    with pytest.raises(DataOperationError, match="Failed to scroll"):
        manager.scroll(collection_name="test_collection", limit=10)


def test_build_filter_simple(mock_connection: QdrantConnectionManager) -> None:
    """Test building simple filter."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.FieldCondition") as mock_field, \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.Filter") as mock_filter, \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.MatchValue"):
        
        filter_dict = {"source": "test.pdf"}
        result = manager._build_filter(filter_dict)
        
        mock_field.assert_called_once()
        mock_filter.assert_called_once()


def test_build_filter_range(mock_connection: QdrantConnectionManager) -> None:
    """Test building range filter."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.FieldCondition") as mock_field, \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.Filter"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.Range"):
        
        filter_dict = {"year": {"$gte": 2020, "$lte": 2023}}
        result = manager._build_filter(filter_dict)
        
        mock_field.assert_called_once()


def test_build_filter_in(mock_connection: QdrantConnectionManager) -> None:
    """Test building IN filter."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.FieldCondition") as mock_field, \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.Filter"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.data.MatchValue"):
        
        filter_dict = {"category": {"$in": ["tech", "science"]}}
        result = manager._build_filter(filter_dict)
        
        # Should create condition for each value
        assert mock_field.call_count == 2


def test_build_filter_empty(mock_connection: QdrantConnectionManager) -> None:
    """Test building empty filter."""
    manager = QdrantDataManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.data.Filter"):
        result = manager._build_filter({})
        
        assert result is None
