"""Tests for Qdrant connection management."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.qdrant.config import QdrantConfig
from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager
from quaerium.infra.vectorstores.qdrant.exceptions import ConnectionError


def test_connection_manager_initialization(qdrant_config: QdrantConfig) -> None:
    """Test connection manager initialization."""
    manager = QdrantConnectionManager(qdrant_config)
    
    assert manager.config == qdrant_config
    assert manager._client is None


def test_connection_manager_creates_client(qdrant_config: QdrantConfig) -> None:
    """Test that connection manager creates client on first access."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        
        client = manager.client
        
        assert client is mock_instance
        assert manager._client is mock_instance
        mock_client_class.assert_called_once_with(
            url=qdrant_config.url,
            api_key=qdrant_config.api_key,
            timeout=qdrant_config.timeout,
            https=qdrant_config.https,
            grpc_port=qdrant_config.grpc_port,
            prefer_grpc=qdrant_config.prefer_grpc,
        )


def test_connection_manager_reuses_client(qdrant_config: QdrantConfig) -> None:
    """Test that connection manager reuses existing client."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        
        client1 = manager.client
        client2 = manager.client
        
        assert client1 is client2
        mock_client_class.assert_called_once()


def test_connection_manager_missing_dependency(qdrant_config: QdrantConfig) -> None:
    """Test error when qdrant-client is not installed."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient", side_effect=ImportError):
        with pytest.raises(ConnectionError, match="qdrant-client is not installed"):
            _ = manager.client


def test_connection_manager_creation_failure(qdrant_config: QdrantConfig) -> None:
    """Test error handling when client creation fails."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient", side_effect=RuntimeError("Connection failed")):
        with pytest.raises(ConnectionError, match="Failed to create Qdrant client"):
            _ = manager.client


def test_health_check_success(mock_connection: QdrantConnectionManager) -> None:
    """Test successful health check."""
    mock_connection.client.get_collections.return_value = Mock(collections=[])
    
    assert mock_connection.health_check() is True


def test_health_check_failure(mock_connection: QdrantConnectionManager) -> None:
    """Test failed health check."""
    mock_connection.client.get_collections.side_effect = RuntimeError("Connection refused")
    
    assert mock_connection.health_check() is False


def test_connection_close(qdrant_config: QdrantConfig) -> None:
    """Test connection cleanup."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        
        _ = manager.client
        assert manager._client is not None
        
        manager.close()
        
        mock_instance.close.assert_called_once()
        assert manager._client is None


def test_connection_close_without_client(qdrant_config: QdrantConfig) -> None:
    """Test closing connection when client was never created."""
    manager = QdrantConnectionManager(qdrant_config)
    
    # Should not raise an error
    manager.close()
    
    assert manager._client is None


def test_connection_close_with_error(qdrant_config: QdrantConfig) -> None:
    """Test connection close handles errors gracefully."""
    manager = QdrantConnectionManager(qdrant_config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_instance.close.side_effect = RuntimeError("Close failed")
        mock_client_class.return_value = mock_instance
        
        _ = manager.client
        
        # Should not raise despite error
        manager.close()
        
        assert manager._client is None


def test_connection_with_api_key() -> None:
    """Test connection configuration with API key."""
    config = QdrantConfig(
        url="https://xyz.cloud.qdrant.io:6333",
        api_key="secret-key",
        https=True,
    )
    
    manager = QdrantConnectionManager(config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        
        _ = manager.client
        
        mock_client_class.assert_called_once_with(
            url="https://xyz.cloud.qdrant.io:6333",
            api_key="secret-key",
            timeout=None,
            https=True,
            grpc_port=None,
            prefer_grpc=False,
        )


def test_connection_with_grpc() -> None:
    """Test connection configuration with gRPC."""
    config = QdrantConfig(
        url="http://localhost:6333",
        grpc_port=6334,
        prefer_grpc=True,
    )
    
    manager = QdrantConnectionManager(config)
    
    with patch("rag_toolkit.infra.vectorstores.qdrant.connection.QdrantClient") as mock_client_class:
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        
        _ = manager.client
        
        mock_client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=None,
            https=False,
            grpc_port=6334,
            prefer_grpc=True,
        )
