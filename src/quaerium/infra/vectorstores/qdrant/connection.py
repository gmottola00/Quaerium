"""Qdrant connection management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from quaerium.infra.vectorstores.qdrant.config import QdrantConfig
from quaerium.infra.vectorstores.qdrant.exceptions import ConnectionError

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class QdrantConnectionManager:
    """Manage connection lifecycle to Qdrant server."""

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize connection manager.
        
        Args:
            config: Qdrant connection configuration
        """
        self.config = config
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client.
        
        Returns:
            Connected Qdrant client instance
            
        Raises:
            ConnectionError: If connection fails
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> QdrantClient:
        """Create new Qdrant client instance.
        
        Returns:
            Configured Qdrant client
            
        Raises:
            ConnectionError: If client creation fails
        """
        try:
            from qdrant_client import QdrantClient

            return QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                https=self.config.https,
                grpc_port=self.config.grpc_port,
                prefer_grpc=self.config.prefer_grpc,
            )
        except ImportError as exc:
            raise ConnectionError(
                "qdrant-client is not installed. Install it with: pip install qdrant-client"
            ) from exc
        except Exception as exc:
            raise ConnectionError(f"Failed to create Qdrant client: {exc}") from exc

    def health_check(self) -> bool:
        """Check if Qdrant server is reachable.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Simple health check by getting cluster info
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close connection to Qdrant."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._client = None


__all__ = ["QdrantConnectionManager"]
