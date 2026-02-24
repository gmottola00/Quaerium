"""Test ChromaDB collection manager."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
from quaerium.infra.vectorstores.chroma.config import ChromaIndexConfig
from quaerium.infra.vectorstores.chroma.exceptions import CollectionError


class TestChromaCollectionManager:
    """Test ChromaCollectionManager class."""

    def test_init(self, mock_connection) -> None:
        """Test initialization."""
        manager = ChromaCollectionManager(mock_connection)
        assert manager.connection == mock_connection

    def test_create_collection_basic(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test creating collection with basic config."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.create_collection.return_value = mock_chroma_collection
        
        collection = manager.create_collection("test_collection")
        
        assert collection == mock_chroma_collection
        mock_connection._client.create_collection.assert_called_once()

    def test_create_collection_with_index_config(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        chroma_index_config: ChromaIndexConfig,
    ) -> None:
        """Test creating collection with index configuration."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.create_collection.return_value = mock_chroma_collection
        
        collection = manager.create_collection(
            "test_collection",
            index_config=chroma_index_config,
        )
        
        assert collection == mock_chroma_collection
        call_args = mock_connection._client.create_collection.call_args
        assert call_args[1]["name"] == "test_collection"
        assert "metadata" in call_args[1]

    def test_create_collection_with_metadata(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test creating collection with custom metadata."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.create_collection.return_value = mock_chroma_collection
        
        metadata = {"description": "Test collection", "version": "1.0"}
        collection = manager.create_collection(
            "test_collection",
            metadata=metadata,
        )
        
        assert collection == mock_chroma_collection
        call_args = mock_connection._client.create_collection.call_args
        assert "description" in call_args[1]["metadata"]

    def test_create_collection_already_exists(self, mock_connection) -> None:
        """Test creating collection that already exists."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.create_collection.side_effect = Exception("Already exists")
        
        with pytest.raises(CollectionError):
            manager.create_collection("existing_collection")

    def test_get_collection_success(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test getting existing collection."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        collection = manager.get_collection("test_collection")
        
        assert collection == mock_chroma_collection
        mock_connection._client.get_collection.assert_called_once_with(
            name="test_collection",
        )

    def test_get_collection_not_found(self, mock_connection) -> None:
        """Test getting non-existent collection."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_collection.side_effect = Exception("Not found")
        
        with pytest.raises(CollectionError):
            manager.get_collection("nonexistent_collection")

    def test_get_or_create_collection_exists(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test get_or_create when collection exists."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_or_create_collection.return_value = mock_chroma_collection
        
        collection = manager.get_or_create_collection("test_collection")
        
        assert collection == mock_chroma_collection
        mock_connection._client.get_or_create_collection.assert_called_once()

    def test_get_or_create_collection_with_config(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        chroma_index_config: ChromaIndexConfig,
    ) -> None:
        """Test get_or_create with index config."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_or_create_collection.return_value = mock_chroma_collection
        
        collection = manager.get_or_create_collection(
            "test_collection",
            index_config=chroma_index_config,
        )
        
        assert collection == mock_chroma_collection
        call_args = mock_connection._client.get_or_create_collection.call_args
        assert "metadata" in call_args[1]

    def test_collection_exists_true(self, mock_connection) -> None:
        """Test checking if collection exists (true)."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_collection.return_value = Mock()
        
        assert manager.collection_exists("test_collection") is True
        mock_connection._client.get_collection.assert_called_once_with(
            name="test_collection",
        )

    def test_collection_exists_false(self, mock_connection) -> None:
        """Test checking if collection exists (false)."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.get_collection.side_effect = Exception("Not found")
        
        assert manager.collection_exists("nonexistent_collection") is False

    def test_drop_collection_success(self, mock_connection) -> None:
        """Test dropping collection."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.delete_collection.return_value = None
        
        manager.drop_collection("test_collection")
        
        mock_connection._client.delete_collection.assert_called_once_with(
            name="test_collection",
        )

    def test_drop_collection_not_found(self, mock_connection) -> None:
        """Test dropping non-existent collection."""
        manager = ChromaCollectionManager(mock_connection)
        mock_connection._client.delete_collection.side_effect = Exception("Not found")
        
        with pytest.raises(CollectionError):
            manager.drop_collection("nonexistent_collection")

    def test_list_collections(self, mock_connection) -> None:
        """Test listing collections."""
        manager = ChromaCollectionManager(mock_connection)
        mock_collections = [
            Mock(name="collection1"),
            Mock(name="collection2"),
        ]
        mock_connection._client.list_collections.return_value = mock_collections
        
        collections = manager.list_collections()
        
        assert len(collections) == 2
        assert collections[0].name == "collection1"
        assert collections[1].name == "collection2"
        mock_connection._client.list_collections.assert_called_once()

    def test_get_collection_info(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test getting collection info."""
        manager = ChromaCollectionManager(mock_connection)
        mock_chroma_collection.count.return_value = 42
        mock_chroma_collection.name = "test_collection"
        mock_chroma_collection.metadata = {"hnsw:space": "cosine"}
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        info = manager.get_collection_info("test_collection")
        
        assert info["name"] == "test_collection"
        assert info["count"] == 42
        assert "metadata" in info
