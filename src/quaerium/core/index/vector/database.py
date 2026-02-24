"""Database-level operations for Milvus."""

from __future__ import annotations

from typing import List

from .connection import MilvusConnectionManager
from .exceptions import ConnectionError, VectorStoreError

try:
    from pymilvus import db as milvus_db
except ImportError:  # pragma: no cover - optional dependency not installed
    milvus_db = None  # type: ignore


class MilvusDatabaseManager:
    """Manage Milvus databases/spaces."""

    def __init__(self, connection: MilvusConnectionManager) -> None:
        if milvus_db is None:
            raise ImportError("pymilvus is required for Milvus operations")
        self.connection = connection

    def list_databases(self) -> List[str]:
        """List available databases."""
        self.connection.ensure()
        try:
            return list(milvus_db.list_databases())
        except Exception as exc:  # pragma: no cover - passthrough
            raise VectorStoreError("Failed to list Milvus databases") from exc

    def create_database(self, name: str) -> None:
        """Create a database if it does not exist."""
        self.connection.ensure()
        try:
            if name not in milvus_db.list_databases():
                milvus_db.create_database(name)
        except Exception as exc:  # pragma: no cover - passthrough
            raise VectorStoreError(f"Failed to create database '{name}'") from exc

    def drop_database(self, name: str, *, force: bool = False) -> None:
        """Drop a database."""
        self.connection.ensure()
        try:
            milvus_db.drop_database(name, force=force)
        except Exception as exc:  # pragma: no cover - passthrough
            raise VectorStoreError(f"Failed to drop database '{name}'") from exc

