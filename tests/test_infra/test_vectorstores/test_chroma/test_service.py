"""Test ChromaDB service facade."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.chroma.config import ChromaConfig, ChromaIndexConfig
from quaerium.infra.vectorstores.chroma.service import ChromaService


class TestChromaService:
    """Test ChromaService class."""

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_init(self, mock_connection_class: Mock, chroma_config: ChromaConfig) -> None:
        """Test initialization."""
        service = ChromaService(chroma_config)
        
        assert service.connection is not None
        assert service.collections is not None
        assert service.data is not None

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_create_collection(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test creating collection through service."""
        service = ChromaService(chroma_config)
        service.collections.create_collection = Mock(return_value=mock_chroma_collection)
        
        collection = service.create_collection("test_collection")
        
        assert collection == mock_chroma_collection
        service.collections.create_collection.assert_called_once_with(
            "test_collection", None, None
        )

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_create_collection_with_config(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        chroma_index_config: ChromaIndexConfig,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test creating collection with index config."""
        service = ChromaService(chroma_config)
        service.collections.create_collection = Mock(return_value=mock_chroma_collection)
        
        metadata = {"description": "Test collection"}
        collection = service.create_collection(
            "test_collection",
            index_config=chroma_index_config,
            metadata=metadata,
        )
        
        assert collection == mock_chroma_collection
        service.collections.create_collection.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_get_or_create_collection(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test get_or_create collection."""
        service = ChromaService(chroma_config)
        service.collections.get_or_create_collection = Mock(return_value=mock_chroma_collection)
        
        collection = service.get_or_create_collection("test_collection")
        
        assert collection == mock_chroma_collection
        service.collections.get_or_create_collection.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_drop_collection(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test dropping collection."""
        service = ChromaService(chroma_config)
        service.collections.drop_collection = Mock()
        
        service.drop_collection("test_collection")
        
        service.collections.drop_collection.assert_called_once_with("test_collection")

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_add(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        sample_embeddings: List[List[float]],
        sample_ids: List[str],
    ) -> None:
        """Test adding documents."""
        service = ChromaService(chroma_config)
        service.data.add = Mock(return_value=sample_ids)
        
        result_ids = service.add(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
        )
        
        assert result_ids == sample_ids
        service.data.add.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_upsert(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        sample_embeddings: List[List[float]],
        sample_ids: List[str],
    ) -> None:
        """Test upserting documents."""
        service = ChromaService(chroma_config)
        service.data.upsert = Mock(return_value=sample_ids)
        
        result_ids = service.upsert(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
        )
        
        assert result_ids == sample_ids
        service.data.upsert.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_query(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test querying documents."""
        service = ChromaService(chroma_config)
        expected_results = {
            "ids": [["doc-1", "doc-2"]],
            "distances": [[0.1, 0.2]],
        }
        service.data.query = Mock(return_value=expected_results)
        
        results = service.query(
            collection_name="test_collection",
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            n_results=2,
        )
        
        assert results == expected_results
        service.data.query.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_query_with_filters(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test querying with filters."""
        service = ChromaService(chroma_config)
        service.data.query = Mock(return_value={"ids": [[]]})
        
        where_filter = {"category": "A"}
        service.query(
            collection_name="test_collection",
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            where=where_filter,
        )
        
        call_args = service.data.query.call_args
        assert call_args[1]["where"] == where_filter

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_get(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        sample_ids: List[str],
    ) -> None:
        """Test getting documents."""
        service = ChromaService(chroma_config)
        expected_results = {"ids": sample_ids, "documents": ["Doc 1", "Doc 2", "Doc 3"]}
        service.data.get = Mock(return_value=expected_results)
        
        results = service.get(
            collection_name="test_collection",
            ids=sample_ids,
        )
        
        assert results == expected_results
        service.data.get.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_update(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        sample_ids: List[str],
        sample_embeddings: List[List[float]],
    ) -> None:
        """Test updating documents."""
        service = ChromaService(chroma_config)
        service.data.update = Mock()
        
        service.update(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
        )
        
        service.data.update.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_delete(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
        sample_ids: List[str],
    ) -> None:
        """Test deleting documents."""
        service = ChromaService(chroma_config)
        service.data.delete = Mock()
        
        service.delete(
            collection_name="test_collection",
            ids=sample_ids,
        )
        
        service.data.delete.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_count(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test counting documents."""
        service = ChromaService(chroma_config)
        service.data.count = Mock(return_value=42)
        
        count = service.count(collection_name="test_collection")
        
        assert count == 42
        service.data.count.assert_called_once_with("test_collection")

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_health_check(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test health check."""
        service = ChromaService(chroma_config)
        service.connection.health_check = Mock(return_value=True)
        
        assert service.health_check() is True
        service.connection.health_check.assert_called_once()

    @patch("quaerium.infra.vectorstores.chroma.connection.ChromaConnectionManager")
    def test_close(
        self,
        mock_connection_class: Mock,
        chroma_config: ChromaConfig,
    ) -> None:
        """Test closing connection."""
        service = ChromaService(chroma_config)
        service.connection.close = Mock()
        
        service.close()
        
        service.connection.close.assert_called_once()
