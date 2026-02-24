"""Tests for Qdrant collection management."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.qdrant.collection import QdrantCollectionManager
from quaerium.infra.vectorstores.qdrant.config import QdrantIndexConfig
from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager
from quaerium.infra.vectorstores.qdrant.exceptions import CollectionError


def test_collection_manager_initialization(mock_connection: QdrantConnectionManager) -> None:
    """Test collection manager initialization."""
    manager = QdrantCollectionManager(mock_connection)
    
    assert manager.connection is mock_connection


def test_create_collection(
    mock_connection: QdrantConnectionManager,
    qdrant_index_config: QdrantIndexConfig,
) -> None:
    """Test collection creation."""
    manager = QdrantCollectionManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance") as mock_distance, \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams") as mock_vector_params:
        
        mock_distance.COSINE = "Cosine"
        mock_params = Mock()
        mock_vector_params.return_value = mock_params
        
        manager.create_collection(
            collection_name="test_collection",
            vector_size=384,
            index_config=qdrant_index_config,
        )
        
        mock_connection.client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config=mock_params,
        )


def test_create_collection_default_config(mock_connection: QdrantConnectionManager) -> None:
    """Test collection creation with default configuration."""
    manager = QdrantCollectionManager(mock_connection)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance") as mock_distance, \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams") as mock_vector_params:
        
        mock_distance.COSINE = "Cosine"
        mock_params = Mock()
        mock_vector_params.return_value = mock_params
        
        manager.create_collection(
            collection_name="test_collection",
            vector_size=1536,
        )
        
        mock_connection.client.create_collection.assert_called_once()


def test_create_collection_error(mock_connection: QdrantConnectionManager) -> None:
    """Test collection creation error handling."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.create_collection.side_effect = RuntimeError("Creation failed")
    
    with pytest.raises(CollectionError, match="Failed to create collection"):
        manager.create_collection(
            collection_name="test_collection",
            vector_size=384,
        )


def test_collection_exists_true(mock_connection: QdrantConnectionManager) -> None:
    """Test checking if collection exists."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = True
    
    assert manager.collection_exists("test_collection") is True
    mock_connection.client.collection_exists.assert_called_once_with("test_collection")


def test_collection_exists_false(mock_connection: QdrantConnectionManager) -> None:
    """Test checking if collection does not exist."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = False
    
    assert manager.collection_exists("test_collection") is False


def test_collection_exists_error(mock_connection: QdrantConnectionManager) -> None:
    """Test collection exists error handling."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.side_effect = RuntimeError("Check failed")
    
    assert manager.collection_exists("test_collection") is False


def test_ensure_collection_creates_if_not_exists(mock_connection: QdrantConnectionManager) -> None:
    """Test ensure_collection creates collection if it doesn't exist."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = False
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams"):
        
        manager.ensure_collection(
            collection_name="test_collection",
            vector_size=384,
        )
        
        mock_connection.client.create_collection.assert_called_once()


def test_ensure_collection_skips_if_exists(mock_connection: QdrantConnectionManager) -> None:
    """Test ensure_collection skips creation if collection exists."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = True
    
    manager.ensure_collection(
        collection_name="test_collection",
        vector_size=384,
    )
    
    mock_connection.client.create_collection.assert_not_called()


def test_drop_collection(mock_connection: QdrantConnectionManager) -> None:
    """Test collection deletion."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = True
    
    manager.drop_collection("test_collection")
    
    mock_connection.client.delete_collection.assert_called_once_with("test_collection")


def test_drop_collection_not_exists(mock_connection: QdrantConnectionManager) -> None:
    """Test dropping non-existent collection."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = False
    
    manager.drop_collection("test_collection")
    
    mock_connection.client.delete_collection.assert_not_called()


def test_drop_collection_error(mock_connection: QdrantConnectionManager) -> None:
    """Test collection deletion error handling."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.collection_exists.return_value = True
    mock_connection.client.delete_collection.side_effect = RuntimeError("Deletion failed")
    
    with pytest.raises(CollectionError, match="Failed to drop collection"):
        manager.drop_collection("test_collection")


def test_list_collections(mock_connection: QdrantConnectionManager) -> None:
    """Test listing all collections."""
    manager = QdrantCollectionManager(mock_connection)
    
    mock_collections = Mock()
    mock_collections.collections = [
        Mock(name="collection1"),
        Mock(name="collection2"),
        Mock(name="collection3"),
    ]
    mock_connection.client.get_collections.return_value = mock_collections
    
    collections = manager.list_collections()
    
    assert collections == ["collection1", "collection2", "collection3"]


def test_list_collections_empty(mock_connection: QdrantConnectionManager) -> None:
    """Test listing collections when none exist."""
    manager = QdrantCollectionManager(mock_connection)
    
    mock_collections = Mock()
    mock_collections.collections = []
    mock_connection.client.get_collections.return_value = mock_collections
    
    collections = manager.list_collections()
    
    assert collections == []


def test_list_collections_error(mock_connection: QdrantConnectionManager) -> None:
    """Test list collections error handling."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.get_collections.side_effect = RuntimeError("List failed")
    
    with pytest.raises(CollectionError, match="Failed to list collections"):
        manager.list_collections()


def test_get_collection_info(mock_connection: QdrantConnectionManager) -> None:
    """Test getting collection information."""
    manager = QdrantCollectionManager(mock_connection)
    
    mock_info = Mock(
        vectors_count=1000,
        points_count=1000,
        status="green",
        config=Mock(params=Mock(vectors=Mock(size=384, distance="Cosine"))),
    )
    mock_connection.client.get_collection.return_value = mock_info
    
    info = manager.get_collection_info("test_collection")
    
    assert info["name"] == "test_collection"
    assert info["vectors_count"] == 1000
    assert info["points_count"] == 1000
    assert info["status"] == "green"
    mock_connection.client.get_collection.assert_called_once_with("test_collection")


def test_get_collection_info_error(mock_connection: QdrantConnectionManager) -> None:
    """Test get collection info error handling."""
    manager = QdrantCollectionManager(mock_connection)
    mock_connection.client.get_collection.side_effect = RuntimeError("Get failed")
    
    with pytest.raises(CollectionError, match="Failed to get collection info"):
        manager.get_collection_info("test_collection")


def test_create_collection_euclid_distance(mock_connection: QdrantConnectionManager) -> None:
    """Test collection creation with Euclidean distance."""
    manager = QdrantCollectionManager(mock_connection)
    
    config = QdrantIndexConfig(distance="Euclid")
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance") as mock_distance, \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams"):
        
        mock_distance.EUCLID = "Euclid"
        
        manager.create_collection(
            collection_name="test_collection",
            vector_size=384,
            index_config=config,
        )
        
        mock_connection.client.create_collection.assert_called_once()


def test_create_collection_with_quantization(mock_connection: QdrantConnectionManager) -> None:
    """Test collection creation with quantization."""
    manager = QdrantCollectionManager(mock_connection)
    
    config = QdrantIndexConfig(
        distance="Cosine",
        quantization={"scalar": {"type": "int8"}},
    )
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance") as mock_distance, \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams") as mock_vector_params:
        
        mock_distance.COSINE = "Cosine"
        mock_params = Mock()
        mock_vector_params.return_value = mock_params
        
        manager.create_collection(
            collection_name="test_collection",
            vector_size=384,
            index_config=config,
        )
        
        mock_vector_params.assert_called_once()
        call_kwargs = mock_vector_params.call_args[1]
        assert call_kwargs["quantization_config"] == {"scalar": {"type": "int8"}}


def test_create_collection_on_disk(mock_connection: QdrantConnectionManager) -> None:
    """Test collection creation with on-disk storage."""
    manager = QdrantCollectionManager(mock_connection)
    
    config = QdrantIndexConfig(
        distance="Cosine",
        on_disk=True,
    )
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.collection.Distance") as mock_distance, \
         patch("rag_toolkit.infra.vectorstores.qdrant.collection.VectorParams") as mock_vector_params:
        
        mock_distance.COSINE = "Cosine"
        mock_params = Mock()
        mock_vector_params.return_value = mock_params
        
        manager.create_collection(
            collection_name="test_collection",
            vector_size=384,
            index_config=config,
        )
        
        mock_vector_params.assert_called_once()
        call_kwargs = mock_vector_params.call_args[1]
        assert call_kwargs["on_disk"] is True
