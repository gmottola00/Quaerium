"""Protocol definitions for vector store abstractions.

This module defines the interfaces for vector database operations,
enabling decoupling from specific implementations like Milvus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorStoreConfig(Protocol):
    """Configuration protocol for vector store connections."""

    uri: str
    user: Optional[str]
    password: Optional[str]
    db_name: str
    secure: bool
    timeout: Optional[float]
    alias: str


@runtime_checkable
class VectorConnection(Protocol):
    """Protocol for managing vector database connections."""

    def connect(self) -> Any:
        """Establish connection to vector store."""
        ...

    def ensure(self) -> Any:
        """Ensure an active connection."""
        ...

    def disconnect(self) -> None:
        """Close the connection."""
        ...

    @property
    def client(self) -> Any:
        """Get the underlying client."""
        ...


@runtime_checkable
class VectorCollectionManager(Protocol):
    """Protocol for collection-level operations."""

    def ensure_collection(
        self,
        name: str,
        schema: Any,
        *,
        index_params: Optional[Dict[str, Any]] = None,
        load: bool = True,
        shards_num: int = 2,
    ) -> None:
        """Create collection if missing, with optional index and load."""
        ...

    def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        ...

    def load(self, name: str) -> None:
        """Load collection into memory."""
        ...

    def release(self, name: str) -> None:
        """Release collection from memory."""
        ...

    def create_index(self, name: str, field_name: str, index_params: Dict[str, Any]) -> None:
        """Create an index on a field."""
        ...


@runtime_checkable
class VectorDataManager(Protocol):
    """Protocol for data operations (CRUD)."""

    def insert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        *,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Insert entities into collection."""
        ...

    def upsert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        *,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Upsert entities if supported."""
        ...

    def delete(
        self,
        collection_name: str,
        expr: str,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """Delete entities matching expression."""
        ...

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
        """Perform vector search."""
        ...

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
        """Run scalar query."""
        ...

    def flush(self, collection_name: str) -> None:
        """Flush pending data."""
        ...


@runtime_checkable
class VectorDatabaseManager(Protocol):
    """Protocol for database-level operations."""

    def list_databases(self) -> List[str]:
        """List available databases."""
        ...

    def create_database(self, name: str) -> None:
        """Create database if not exists."""
        ...

    def drop_database(self, name: str, *, force: bool = False) -> None:
        """Drop a database."""
        ...


@runtime_checkable
class VectorStoreService(Protocol):
    """High-level protocol for vector store facade."""

    connection: VectorConnection
    databases: VectorDatabaseManager
    collections: VectorCollectionManager
    data: VectorDataManager

    def ensure_database(self, name: str) -> None:
        """Ensure database exists."""
        ...

    def ensure_collection(self, name: str, schema: Any, **kwargs: Any) -> Any:
        """Ensure collection exists and loaded."""
        ...

    def drop_collection(self, name: str) -> None:
        """Drop collection."""
        ...

    def insert(
        self,
        collection_name: str,
        data: Sequence[Sequence[Any]] | Dict[str, Sequence[Any]] | Dict[str, List[Any]],
        **kwargs: Any,
    ) -> Any:
        """Insert rows."""
        ...

    def search(
        self,
        collection_name: str,
        vectors: Sequence[Sequence[float]],
        anns_field: str,
        param: Dict[str, Any],
        limit: int,
        **kwargs: Any,
    ) -> Any:
        """Run vector search."""
        ...


@runtime_checkable
class SearchStrategy(Protocol):
    """Protocol for search implementations (vector, keyword, hybrid)."""

    def search(self, query: str, *, top_k: int = 5, **kwargs: Any) -> List[Dict[str, object]]:
        """Execute search and return ranked results."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking search results."""

    def rerank(self, query: str, results: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
        """Rerank results and return top_k."""
        ...


__all__ = [
    "VectorStoreConfig",
    "VectorConnection",
    "VectorCollectionManager",
    "VectorDataManager",
    "VectorDatabaseManager",
    "VectorStoreService",
    "SearchStrategy",
    "Reranker",
]
