"""Tests for Qdrant service facade."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.qdrant.config import QdrantConfig, QdrantIndexConfig
from quaerium.infra.vectorstores.qdrant.service import QdrantService


def test_service_initialization(qdrant_config: QdrantConfig) -> None:
    """Test service initialization."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"):
        service = QdrantService(qdrant_config)
        
        assert service.connection is not None
        assert service.collections is not None
        assert service.data is not None


def test_ensure_collection(qdrant_config: QdrantConfig) -> None:
    """Test ensure collection through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantCollectionManager") as mock_collections_class:
        
        service = QdrantService(qdrant_config)
        mock_collections = Mock()
        mock_collections_class.return_value = mock_collections
        service.collections = mock_collections
        
        service.ensure_collection(
            collection_name="test_collection",
            vector_size=384,
        )
        
        mock_collections.ensure_collection.assert_called_once_with(
            "test_collection",
            384,
            None,
        )


def test_ensure_collection_with_config(
    qdrant_config: QdrantConfig,
    qdrant_index_config: QdrantIndexConfig,
) -> None:
    """Test ensure collection with index config."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantCollectionManager") as mock_collections_class:
        
        service = QdrantService(qdrant_config)
        mock_collections = Mock()
        mock_collections_class.return_value = mock_collections
        service.collections = mock_collections
        
        service.ensure_collection(
            collection_name="test_collection",
            vector_size=1536,
            index_config=qdrant_index_config,
        )
        
        mock_collections.ensure_collection.assert_called_once_with(
            "test_collection",
            1536,
            qdrant_index_config,
        )


def test_drop_collection(qdrant_config: QdrantConfig) -> None:
    """Test drop collection through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantCollectionManager") as mock_collections_class:
        
        service = QdrantService(qdrant_config)
        mock_collections = Mock()
        mock_collections_class.return_value = mock_collections
        service.collections = mock_collections
        
        service.drop_collection("test_collection")
        
        mock_collections.drop_collection.assert_called_once_with("test_collection")


def test_upsert(
    qdrant_config: QdrantConfig,
    sample_points: list[dict],
) -> None:
    """Test upsert through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_data.upsert.return_value = ["id1", "id2", "id3"]
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        ids = service.upsert(
            collection_name="test_collection",
            points=sample_points,
        )
        
        assert ids == ["id1", "id2", "id3"]
        mock_data.upsert.assert_called_once_with("test_collection", sample_points)


def test_search(
    qdrant_config: QdrantConfig,
    sample_vectors: list[list[float]],
) -> None:
    """Test search through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_results = [{"id": "p1", "score": 0.9}]
        mock_data.search.return_value = mock_results
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        results = service.search(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=10,
        )
        
        assert results == mock_results
        mock_data.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=10,
            query_filter=None,
            score_threshold=None,
        )


def test_search_with_filter(
    qdrant_config: QdrantConfig,
    sample_vectors: list[list[float]],
) -> None:
    """Test search with filter through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_data.search.return_value = []
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        query_filter = {"source": "test.pdf"}
        
        service.search(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=5,
            query_filter=query_filter,
            score_threshold=0.8,
        )
        
        mock_data.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=5,
            query_filter=query_filter,
            score_threshold=0.8,
        )


def test_batch_search(
    qdrant_config: QdrantConfig,
    sample_vectors: list[list[float]],
) -> None:
    """Test batch search through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_results = [[{"id": "p1"}], [{"id": "p2"}]]
        mock_data.batch_search.return_value = mock_results
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        results = service.batch_search(
            collection_name="test_collection",
            query_vectors=sample_vectors[:2],
            limit=5,
        )
        
        assert results == mock_results
        mock_data.batch_search.assert_called_once_with(
            collection_name="test_collection",
            query_vectors=sample_vectors[:2],
            limit=5,
            query_filter=None,
        )


def test_retrieve(qdrant_config: QdrantConfig) -> None:
    """Test retrieve through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_points = [{"id": "p1"}, {"id": "p2"}]
        mock_data.retrieve.return_value = mock_points
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        results = service.retrieve(
            collection_name="test_collection",
            ids=["p1", "p2"],
        )
        
        assert results == mock_points
        mock_data.retrieve.assert_called_once_with("test_collection", ["p1", "p2"])


def test_delete_by_ids(qdrant_config: QdrantConfig) -> None:
    """Test delete by IDs through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        service.delete(
            collection_name="test_collection",
            ids=["p1", "p2"],
        )
        
        mock_data.delete.assert_called_once_with("test_collection", None, ["p1", "p2"])


def test_delete_by_filter(qdrant_config: QdrantConfig) -> None:
    """Test delete by filter through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        filter_dict = {"source": "old.pdf"}
        
        service.delete(
            collection_name="test_collection",
            points_selector=filter_dict,
        )
        
        mock_data.delete.assert_called_once_with("test_collection", filter_dict, None)


def test_scroll(qdrant_config: QdrantConfig) -> None:
    """Test scroll through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_points = [{"id": "p1"}]
        mock_data.scroll.return_value = (mock_points, "next-offset")
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        points, next_offset = service.scroll(
            collection_name="test_collection",
            limit=10,
            offset="current-offset",
        )
        
        assert points == mock_points
        assert next_offset == "next-offset"
        mock_data.scroll.assert_called_once_with(
            "test_collection",
            10,
            "current-offset",
            None,
        )


def test_health_check(qdrant_config: QdrantConfig) -> None:
    """Test health check through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager") as mock_conn_class:
        mock_connection = Mock()
        mock_connection.health_check.return_value = True
        mock_conn_class.return_value = mock_connection
        
        service = QdrantService(qdrant_config)
        
        assert service.health_check() is True
        mock_connection.health_check.assert_called_once()


def test_close(qdrant_config: QdrantConfig) -> None:
    """Test close connection through service."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager") as mock_conn_class:
        mock_connection = Mock()
        mock_conn_class.return_value = mock_connection
        
        service = QdrantService(qdrant_config)
        service.close()
        
        mock_connection.close.assert_called_once()


def test_service_integration_flow(
    qdrant_config: QdrantConfig,
    sample_points: list[dict],
    sample_vectors: list[list[float]],
) -> None:
    """Test complete service flow: create, insert, search, delete."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantCollectionManager") as mock_collections_class, \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        
        # Mock managers
        mock_collections = Mock()
        mock_data = Mock()
        mock_collections_class.return_value = mock_collections
        mock_data_class.return_value = mock_data
        service.collections = mock_collections
        service.data = mock_data
        
        # Mock responses
        mock_data.upsert.return_value = ["id1", "id2", "id3"]
        mock_data.search.return_value = [{"id": "id1", "score": 0.9}]
        
        # Complete flow
        service.ensure_collection("test_collection", 384)
        ids = service.upsert("test_collection", sample_points)
        results = service.search("test_collection", sample_vectors[0], limit=5)
        service.delete("test_collection", ids=ids)
        service.drop_collection("test_collection")
        
        # Verify calls
        mock_collections.ensure_collection.assert_called_once()
        mock_data.upsert.assert_called_once()
        mock_data.search.assert_called_once()
        mock_data.delete.assert_called_once()
        mock_collections.drop_collection.assert_called_once()


def test_service_with_custom_search_params(
    qdrant_config: QdrantConfig,
    sample_vectors: list[list[float]],
) -> None:
    """Test service search with custom parameters."""
    with patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantConnectionManager"), \
         patch("rag_toolkit.infra.vectorstores.qdrant.service.QdrantDataManager") as mock_data_class:
        
        service = QdrantService(qdrant_config)
        mock_data = Mock()
        mock_data.search.return_value = []
        mock_data_class.return_value = mock_data
        service.data = mock_data
        
        service.search(
            collection_name="test_collection",
            query_vector=sample_vectors[0],
            limit=20,
            query_filter={"category": "tech"},
            score_threshold=0.75,
            with_vectors=True,  # Extra kwarg
        )
        
        # Verify call includes custom params
        call_args = mock_data.search.call_args
        assert "with_vectors" in call_args[1]
        assert call_args[1]["with_vectors"] is True
