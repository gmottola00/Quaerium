"""ChromaDB connection management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from quaerium.infra.vectorstores.chroma.config import ChromaConfig
from quaerium.infra.vectorstores.chroma.exceptions import ConnectionError

if TYPE_CHECKING:
    import chromadb


class ChromaConnectionManager:
    """Manage connection lifecycle to ChromaDB."""

    def __init__(self, config: ChromaConfig) -> None:
        """Initialize connection manager.
        
        Args:
            config: ChromaDB connection configuration
        """
        self.config = config
        self._client: Optional[chromadb.ClientAPI] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Get or create ChromaDB client.
        
        Returns:
            Connected ChromaDB client instance
            
        Raises:
            ConnectionError: If connection fails
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> chromadb.ClientAPI:
        """Create new ChromaDB client instance.
        
        Returns:
            Configured ChromaDB client
            
        Raises:
            ConnectionError: If client creation fails
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ConnectionError(
                "chromadb is not installed. Install it with: pip install chromadb"
            ) from exc

        try:
            # Remote client
            if self.config.host:
                settings = Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=self.config.host,
                    chroma_server_http_port=self.config.port or 8000,
                    chroma_server_ssl_enabled=self.config.ssl,
                    chroma_server_headers=self.config.headers or {},
                )
                return chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port or 8000,
                    ssl=self.config.ssl,
                    headers=self.config.headers or {},
                    tenant=self.config.tenant,
                    database=self.config.database,
                )
            
            # Persistent client
            if self.config.path:
                return chromadb.PersistentClient(
                    path=self.config.path,
                    tenant=self.config.tenant,
                    database=self.config.database,
                )
            
            # In-memory client (default)
            return chromadb.Client(
                tenant=self.config.tenant,
                database=self.config.database,
            )
            
        except Exception as exc:
            raise ConnectionError(f"Failed to create ChromaDB client: {exc}") from exc

    def health_check(self) -> bool:
        """Check if ChromaDB is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check by getting heartbeat
            self.client.heartbeat()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close connection to ChromaDB."""
        # ChromaDB doesn't require explicit close for most clients
        # but we reset the reference
        self._client = None


__all__ = ["ChromaConnectionManager"]
