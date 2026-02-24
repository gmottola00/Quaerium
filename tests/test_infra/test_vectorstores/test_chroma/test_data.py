"""Test ChromaDB data operations manager."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from quaerium.infra.vectorstores.chroma.data import ChromaDataManager
from quaerium.infra.vectorstores.chroma.exceptions import DataOperationError


class TestChromaDataManager:
    """Test ChromaDataManager class."""

    def test_init(self, mock_connection) -> None:
        """Test initialization."""
        manager = ChromaDataManager(mock_connection)
        assert manager.connection == mock_connection

    def test_add_with_embeddings(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_embeddings: List[List[float]],
        sample_ids: List[str],
        sample_metadatas: List[Dict[str, Any]],
    ) -> None:
        """Test adding documents with embeddings."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        result_ids = manager.add(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
        )
        
        assert result_ids == sample_ids
        mock_chroma_collection.add.assert_called_once()

    def test_add_with_documents(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_documents: List[str],
        sample_ids: List[str],
    ) -> None:
        """Test adding documents with text."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        result_ids = manager.add(
            collection_name="test_collection",
            ids=sample_ids,
            documents=sample_documents,
        )
        
        assert result_ids == sample_ids
        mock_chroma_collection.add.assert_called_once()

    def test_add_auto_generate_ids(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_embeddings: List[List[float]],
    ) -> None:
        """Test adding documents with auto-generated IDs."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        result_ids = manager.add(
            collection_name="test_collection",
            embeddings=sample_embeddings,
        )
        
        assert len(result_ids) == len(sample_embeddings)
        assert all(isinstance(id, str) for id in result_ids)
        mock_chroma_collection.add.assert_called_once()

    def test_add_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test add operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.add.side_effect = Exception("Add failed")
        
        with pytest.raises(DataOperationError):
            manager.add(
                collection_name="test_collection",
                embeddings=[[0.1, 0.2]],
            )

    def test_upsert(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_embeddings: List[List[float]],
        sample_ids: List[str],
        sample_metadatas: List[Dict[str, Any]],
    ) -> None:
        """Test upserting documents."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        result_ids = manager.upsert(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
        )
        
        assert result_ids == sample_ids
        mock_chroma_collection.upsert.assert_called_once()

    def test_upsert_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test upsert operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.upsert.side_effect = Exception("Upsert failed")
        
        with pytest.raises(DataOperationError):
            manager.upsert(
                collection_name="test_collection",
                ids=["id-1"],
                embeddings=[[0.1, 0.2]],
            )

    def test_query_with_embeddings(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test querying with embeddings."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        query_embedding = [[0.1, 0.2, 0.3, 0.4]]
        mock_chroma_collection.query.return_value = {
            "ids": [["doc-1", "doc-2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["First doc", "Second doc"]],
            "metadatas": [[{"source": "test"}, {"source": "test"}]],
        }
        
        results = manager.query(
            collection_name="test_collection",
            query_embeddings=query_embedding,
            n_results=2,
        )
        
        assert "ids" in results
        assert len(results["ids"][0]) == 2
        mock_chroma_collection.query.assert_called_once()

    def test_query_with_filters(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test querying with metadata filters."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        where_filter = {"source": "test", "category": "A"}
        manager.query(
            collection_name="test_collection",
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            where=where_filter,
            n_results=5,
        )
        
        call_args = mock_chroma_collection.query.call_args
        assert call_args[1]["where"] == where_filter

    def test_query_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test query operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.query.side_effect = Exception("Query failed")
        
        with pytest.raises(DataOperationError):
            manager.query(
                collection_name="test_collection",
                query_embeddings=[[0.1, 0.2]],
            )

    def test_get_by_ids(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_ids: List[str],
    ) -> None:
        """Test getting documents by IDs."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        mock_chroma_collection.get.return_value = {
            "ids": sample_ids,
            "documents": ["Doc 1", "Doc 2", "Doc 3"],
            "metadatas": [{"source": "test"}] * 3,
            "embeddings": [[0.1, 0.2]] * 3,
        }
        
        results = manager.get(
            collection_name="test_collection",
            ids=sample_ids,
        )
        
        assert results["ids"] == sample_ids
        assert len(results["documents"]) == 3
        mock_chroma_collection.get.assert_called_once()

    def test_get_with_filters(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test getting documents with filters."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        where_filter = {"category": "A"}
        manager.get(
            collection_name="test_collection",
            where=where_filter,
            limit=10,
        )
        
        call_args = mock_chroma_collection.get.call_args
        assert call_args[1]["where"] == where_filter
        assert call_args[1]["limit"] == 10

    def test_get_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test get operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.get.side_effect = Exception("Get failed")
        
        with pytest.raises(DataOperationError):
            manager.get(collection_name="test_collection", ids=["id-1"])

    def test_update(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_ids: List[str],
        sample_embeddings: List[List[float]],
    ) -> None:
        """Test updating documents."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        manager.update(
            collection_name="test_collection",
            ids=sample_ids,
            embeddings=sample_embeddings,
            metadatas=[{"updated": True}] * 3,
        )
        
        mock_chroma_collection.update.assert_called_once()

    def test_update_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test update operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.update.side_effect = Exception("Update failed")
        
        with pytest.raises(DataOperationError):
            manager.update(
                collection_name="test_collection",
                ids=["id-1"],
                embeddings=[[0.1, 0.2]],
            )

    def test_delete_by_ids(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
        sample_ids: List[str],
    ) -> None:
        """Test deleting documents by IDs."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        manager.delete(
            collection_name="test_collection",
            ids=sample_ids,
        )
        
        mock_chroma_collection.delete.assert_called_once_with(
            ids=sample_ids,
            where=None,
            where_document=None,
        )

    def test_delete_by_filter(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test deleting documents by filter."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        
        where_filter = {"source": "test"}
        manager.delete(
            collection_name="test_collection",
            where=where_filter,
        )
        
        call_args = mock_chroma_collection.delete.call_args
        assert call_args[1]["where"] == where_filter

    def test_delete_no_criteria(self, mock_connection) -> None:
        """Test delete without any criteria raises error."""
        manager = ChromaDataManager(mock_connection)
        
        with pytest.raises(DataOperationError, match="Must provide at least one"):
            manager.delete(collection_name="test_collection")

    def test_delete_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test delete operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.delete.side_effect = Exception("Delete failed")
        
        with pytest.raises(DataOperationError):
            manager.delete(collection_name="test_collection", ids=["id-1"])

    def test_count(
        self,
        mock_connection,
        mock_chroma_collection: Mock,
    ) -> None:
        """Test counting documents."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.count.return_value = 42
        
        count = manager.count(collection_name="test_collection")
        
        assert count == 42
        mock_chroma_collection.count.assert_called_once()

    def test_count_failure(self, mock_connection, mock_chroma_collection: Mock) -> None:
        """Test count operation failure."""
        manager = ChromaDataManager(mock_connection)
        mock_connection._client.get_collection.return_value = mock_chroma_collection
        mock_chroma_collection.count.side_effect = Exception("Count failed")
        
        with pytest.raises(DataOperationError):
            manager.count(collection_name="test_collection")
