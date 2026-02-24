"""Connection management for Milvus using MilvusClient."""

from __future__ import annotations

from typing import Optional

from .config import MilvusConfig
from .exceptions import ConfigurationError, ConnectionError

try:
    from pymilvus import MilvusClient
except ImportError as exc:  # pragma: no cover - optional dependency not installed
    MilvusClient = None  # type: ignore
    _pymilvus_import_error = exc


class MilvusConnectionManager:
    """Manage a singleton MilvusClient connection."""

    def __init__(self, config: MilvusConfig) -> None:
        if MilvusClient is None:
            raise ImportError("pymilvus is required for Milvus operations") from _pymilvus_import_error
        if not config.uri:
            raise ConfigurationError("Milvus URI is required")
        self.config = config
        self._client: Optional[MilvusClient] = None

    def connect(self) -> MilvusClient:
        """Instantiate the MilvusClient if not already created."""
        if self._client is not None:
            return self._client
        try:
            self._client = MilvusClientFactory.get(self.config)
            return self._client
        except Exception as exc:  # pragma: no cover
            raise ConnectionError(f"Failed to connect to Milvus at {self.config.uri}") from exc

    def ensure(self) -> MilvusClient:
        """Ensure an active client."""
        return self.connect()

    @property
    def client(self) -> MilvusClient:
        return self.ensure()

    def disconnect(self) -> None:
        """MilvusClient does not expose an explicit disconnect; noop kept for API symmetry."""
        self._client = None

    def get_alias(self) -> str:
        """Return the logical alias (kept for backward compatibility)."""
        return self.config.alias


class MilvusClientFactory:
    """Factory + singleton helper to build MilvusClient instances from config."""

    _instance: Optional[MilvusClient] = None

    @classmethod
    def get(cls, config: MilvusConfig) -> MilvusClient:
        if cls._instance is not None:
            return cls._instance
        if MilvusClient is None:
            raise ImportError("pymilvus is required for Milvus operations") from _pymilvus_import_error
        token = None
        if config.user and config.password:
            token = f"{config.user}:{config.password}"
        cls._instance = MilvusClient(
            uri=config.uri,
            token=token,
            db_name=config.db_name,
            secure=config.secure,
            timeout=config.timeout,
        )
        return cls._instance


__all__ = ["MilvusConnectionManager"]
