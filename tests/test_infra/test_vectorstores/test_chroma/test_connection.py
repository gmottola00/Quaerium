"""Test ChromaDB connection manager."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from quaerium.infra.vectorstores.chroma.config import ChromaConfig
from quaerium.infra.vectorstores.chroma.connection import ChromaConnectionManager
from quaerium.infra.vectorstores.chroma.exceptions import ConnectionError


class TestChromaConnectionManager:
    """Test ChromaConnectionManager class."""

    def test_init_in_memory(self, chroma_config: ChromaConfig) -> None:
        """Test initialization with in-memory config."""
        connection = ChromaConnectionManager(chroma_config)
        assert connection.config == chroma_config
        connection.close()

    def test_init_persistent(self, chroma_persistent_config: ChromaConfig) -> None:
        """Test initialization with persistent storage."""
        connection = ChromaConnectionManager(chroma_persistent_config)
        assert connection.config == chroma_persistent_config
        assert connection.config.path is not None
        connection.close()

    def test_init_remote(self, chroma_remote_config: ChromaConfig) -> None:
        """Test initialization with remote server."""
        connection = ChromaConnectionManager(chroma_remote_config)
        assert connection.config == chroma_remote_config
        assert connection.config.host == "localhost"
        assert connection.config.port == 8000
        connection.close()

    @patch("chromadb.Client")
    def test_get_client_in_memory(self, mock_client_class: Mock, chroma_config: ChromaConfig) -> None:
        """Test getting in-memory client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        connection = ChromaConnectionManager(chroma_config)
        client = connection.get_client()
        
        assert client == mock_client
        mock_client_class.assert_called_once()
        connection.close()

    @patch("chromadb.PersistentClient")
    def test_get_client_persistent(
        self,
        mock_client_class: Mock,
        chroma_persistent_config: ChromaConfig,
    ) -> None:
        """Test getting persistent client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        connection = ChromaConnectionManager(chroma_persistent_config)
        client = connection.get_client()
        
        assert client == mock_client
        mock_client_class.assert_called_once_with(path=chroma_persistent_config.path)
        connection.close()

    @patch("chromadb.HttpClient")
    def test_get_client_remote(
        self,
        mock_client_class: Mock,
        chroma_remote_config: ChromaConfig,
    ) -> None:
        """Test getting remote HTTP client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        connection = ChromaConnectionManager(chroma_remote_config)
        client = connection.get_client()
        
        assert client == mock_client
        mock_client_class.assert_called_once()
        connection.close()

    def test_health_check_success(self, mock_connection: ChromaConnectionManager) -> None:
        """Test successful health check."""
        mock_connection._client.heartbeat.return_value = 1234567890
        
        assert mock_connection.health_check() is True
        mock_connection._client.heartbeat.assert_called_once()

    def test_health_check_failure(self, mock_connection: ChromaConnectionManager) -> None:
        """Test failed health check."""
        mock_connection._client.heartbeat.side_effect = Exception("Connection failed")
        
        assert mock_connection.health_check() is False

    def test_close(self, mock_connection: ChromaConnectionManager) -> None:
        """Test closing connection."""
        mock_connection.close()
        # ChromaDB client doesn't have explicit close, just verify no errors
        assert True
