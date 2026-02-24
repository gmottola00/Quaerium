"""High-level facade for Milvus vector operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from quaerium.infra.vectorstores.milvus.collection import MilvusCollectionManager
from quaerium.infra.vectorstores.milvus.config import MilvusConfig
from quaerium.infra.vectorstores.milvus.connection import MilvusConnectionManager
from quaerium.infra.vectorstores.milvus.data import MilvusDataManager
from quaerium.infra.vectorstores.milvus.database import MilvusDatabaseManager
from quaerium.infra.vectorstores.milvus.explorer import MilvusExplorer


class MilvusService:
    """Compose Milvus managers into a cohesive service."""

    def __init__(self, config: MilvusConfig) -> None:
        self.connection = MilvusConnectionManager(config)
        self.databases = MilvusDatabaseManager(self.connection)
        self.collections = MilvusCollectionManager(self.connection)
        self.data = MilvusDataManager(self.connection)
        self.explorer = MilvusExplorer(self.connection)

    # Convenience wrappers
    def ensure_database(self, name: str) -> None:
        """Ensure a database exists."""
        self.databases.create_database(name)

    def ensure_collection(self, name: str, schema: Any, **kwargs: Any) -> Any:
        """Ensure a collection exists and is loaded."""
        return self.collections.ensure_collection(name, schema, **kwargs)

    def drop_collection(self, name: str) -> None:
        """Drop a collection if present."""
        self.collections.drop_collection(name)

    def list_collections(self) -> list[str]:
        """List all collections."""
        return self.connection.client.list_collections()

    def insert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        **kwargs: Any,
    ) -> Any:
        """Insert rows into a collection."""
        return self.data.insert(collection_name, data, **kwargs)

    def search(
        self,
        collection_name: str,
        vectors: Sequence[Sequence[float]],
        anns_field: str,
        param: Dict[str, Any],
        limit: int,
        **kwargs: Any,
    ) -> Any:
        """Run a vector search."""
        return self.data.search(
            collection_name=collection_name,
            data=vectors,
            anns_field=anns_field,
            param=param,
            limit=limit,
            **kwargs,
        )

    def upsert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        **kwargs: Any,
    ) -> Any:
        """Upsert rows into a collection."""
        return self.data.upsert(collection_name, data, **kwargs)

    def count(self, collection_name: str) -> int:
        """Count entities in a collection."""
        result = self.connection.client.query(
            collection_name=collection_name,
            output_fields=["count(*)"],
        )
        return result[0]["count(*)"] if result else 0


__all__ = ["MilvusService"]
