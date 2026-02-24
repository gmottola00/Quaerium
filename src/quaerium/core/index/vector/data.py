"""Data operations (insert/search/query) for Milvus collections using MilvusClient."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .connection import MilvusConnectionManager
from .exceptions import DataOperationError


class MilvusDataManager:
    """Perform data-level operations on Milvus collections."""

    def __init__(self, connection: MilvusConnectionManager) -> None:
        self.connection = connection

    @property
    def client(self):
        return self.connection.client

    def insert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        *,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Insert entities into a collection."""
        self.connection.ensure()
        try:
            return self.client.insert(
                collection_name=collection_name,
                data=data,
                partition_name=partition_name,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Insert failed for collection '{collection_name}'") from exc

    def upsert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        *,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Upsert entities if supported; fallback to insert."""
        self.connection.ensure()
        try:
            if hasattr(self.client, "upsert"):
                return self.client.upsert(
                    collection_name=collection_name,
                    data=data,
                    partition_name=partition_name,
                    timeout=timeout,
                )
            return self.client.insert(
                collection_name=collection_name,
                data=data,
                partition_name=partition_name,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Upsert failed for collection '{collection_name}'") from exc

    def delete(
        self,
        collection_name: str,
        expr: str,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """Delete entities matching an expression."""
        self.connection.ensure()
        try:
            return self.client.delete(collection_name=collection_name, filter=expr, timeout=timeout)
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Delete failed for collection '{collection_name}'") from exc

    def search(
        self,
        collection_name: str,
        data: Sequence[Sequence[float]],
        anns_field: str,
        param: Dict[str, Any],
        limit: int,
        *,
        output_fields: Optional[List[str]] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        consistency_level: Optional[str] = None,
    ) -> Any:
        """Perform a vector search."""
        self.connection.ensure()
        try:
            return self.client.search(
                collection_name=collection_name,
                data=data,
                anns_field=anns_field,
                search_params=param,
                limit=limit,
                filter=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                consistency_level=consistency_level,
            )
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Search failed for collection '{collection_name}'") from exc

    def query(
        self,
        collection_name: str,
        expr: str,
        *,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        timeout: Optional[float] = None,
        consistency_level: Optional[str] = None,
    ) -> Any:
        """Run a scalar query on a collection."""
        self.connection.ensure()
        try:
            return self.client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset,
                timeout=timeout,
                consistency_level=consistency_level,
            )
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Query failed for collection '{collection_name}'") from exc

    def flush(self, collection_name: str) -> None:
        """Flush pending data to storage."""
        self.connection.ensure()
        try:
            self.client.flush(collection_name=collection_name)
        except Exception as exc:  # pragma: no cover
            raise DataOperationError(f"Flush failed for collection '{collection_name}'") from exc
