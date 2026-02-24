"""Tests for Milvus connection manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quaerium.infra.vectorstores.milvus.config import MilvusConfig
from quaerium.infra.vectorstores.milvus.connection import MilvusConnectionManager
from quaerium.infra.vectorstores.milvus.exceptions import ConnectionError


def test_connection_initialization(milvus_config):
    """Test connection manager initialization."""
    manager = MilvusConnectionManager(milvus_config)
    assert manager.config == milvus_config
    assert manager._client is None


def test_client_creation(mock_connection_manager, mock_milvus_client):
    """Test Milvus client creation."""
    client = mock_connection_manager.client
    assert client is not None
    assert client == mock_milvus_client


def test_client_reuse(mock_connection_manager):
    """Test that client is reused."""
    client1 = mock_connection_manager.client
    client2 = mock_connection_manager.client
    assert client1 is client2


def test_connection_with_token(mock_milvus_client):
    """Test connection with authentication."""
    config = MilvusConfig(
        uri="http://localhost:19530",
        user="test-user",
        password="test-pass",
    )
    
    with patch("rag_toolkit.infra.vectorstores.milvus.connection.MilvusClient") as mock:
        mock.return_value = mock_milvus_client
        manager = MilvusConnectionManager(config)
        _ = manager.client
        
        # Check that token is constructed from user:password
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["token"] == "test-user:test-pass"


def test_connection_with_custom_db(mock_milvus_client):
    """Test connection with custom database."""
    config = MilvusConfig(
        uri="http://localhost:19530",
        db_name="custom_db",
    )
    
    with patch("rag_toolkit.infra.vectorstores.milvus.connection.MilvusClient") as mock:
        mock.return_value = mock_milvus_client
        manager = MilvusConnectionManager(config)
        _ = manager.client
        
        assert mock.call_args.kwargs["db_name"] == "custom_db"


def test_connection_error_handling():
    """Test connection error handling."""
    config = MilvusConfig(uri="http://invalid:19530")
    
    with patch("rag_toolkit.infra.vectorstores.milvus.connection.MilvusClient") as mock:
        mock.side_effect = Exception("Connection failed")
        
        manager = MilvusConnectionManager(config)
        with pytest.raises(ConnectionError, match="Failed to connect"):
            _ = manager.client


def test_health_check_success(mock_connection_manager, mock_milvus_client):
    """Test successful health check."""
    mock_milvus_client.list_collections.return_value = []
    
    is_healthy = mock_connection_manager.health_check()
    assert is_healthy is True
    mock_milvus_client.list_collections.assert_called_once()


def test_health_check_failure(mock_connection_manager, mock_milvus_client):
    """Test health check failure."""
    mock_milvus_client.list_collections.side_effect = Exception("Connection lost")
    
    is_healthy = mock_connection_manager.health_check()
    assert is_healthy is False


def test_disconnect(mock_connection_manager, mock_milvus_client):
    """Test disconnecting."""
    _ = mock_connection_manager.client  # Initialize client
    
    mock_connection_manager.disconnect()
    assert mock_connection_manager._client is None


def test_get_alias(mock_connection_manager):
    """Test getting alias."""
    alias = mock_connection_manager.get_alias()
    assert alias == "default"


def test_ensure_connection(mock_connection_manager, mock_milvus_client):
    """Test ensuring connection."""
    client = mock_connection_manager.ensure()
    assert client == mock_milvus_client
