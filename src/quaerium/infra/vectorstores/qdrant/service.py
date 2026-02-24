"""High-level facade for Qdrant vector operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from quaerium.infra.vectorstores.qdrant.collection import QdrantCollectionManager
from quaerium.infra.vectorstores.qdrant.config import QdrantConfig, QdrantIndexConfig
from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager
from quaerium.infra.vectorstores.qdrant.data import QdrantDataManager


class QdrantService:
    """Compose Qdrant managers into a cohesive service."""

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize Qdrant service.
        
        Args:
            config: Qdrant connection configuration
        """
        self.connection = QdrantConnectionManager(config)
        self.collections = QdrantCollectionManager(self.connection)
        self.data = QdrantDataManager(self.connection)

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        index_config: Optional[QdrantIndexConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Ensure a collection exists, create if not.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            index_config: Index configuration
            **kwargs: Additional collection parameters
        """
        self.collections.ensure_collection(collection_name, vector_size, index_config, **kwargs)

    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection if present.
        
        Args:
            collection_name: Name of the collection to delete
        """
        self.collections.drop_collection(collection_name)

    def upsert(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        """Insert or update points into a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points with id, vector, and payload
            **kwargs: Additional upsert parameters
            
        Returns:
            List of point IDs
        """
        return self.data.upsert(collection_name, points, **kwargs)

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run a vector search.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Number of results to return
            query_filter: Filter conditions for search
            score_threshold: Minimum score threshold
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        return self.data.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            **kwargs,
        )

    def batch_search(
        self,
        collection_name: str,
        query_vectors: List[Sequence[float]],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Search multiple query vectors at once.
        
        Args:
            collection_name: Name of the collection
            query_vectors: List of query vectors
            limit: Number of results per query
            query_filter: Filter conditions for search
            **kwargs: Additional search parameters
            
        Returns:
            List of search results for each query
        """
        return self.data.batch_search(
            collection_name=collection_name,
            query_vectors=query_vectors,
            limit=limit,
            query_filter=query_filter,
            **kwargs,
        )

    def retrieve(
        self,
        collection_name: str,
        ids: List[Union[str, int]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of point IDs to retrieve
            **kwargs: Additional retrieve parameters
            
        Returns:
            List of retrieved points
        """
        return self.data.retrieve(collection_name, ids, **kwargs)

    def delete(
        self,
        collection_name: str,
        points_selector: Optional[Dict[str, Any]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete points from collection.
        
        Args:
            collection_name: Name of the collection
            points_selector: Filter for points to delete
            ids: Specific point IDs to delete
            **kwargs: Additional delete parameters
        """
        self.data.delete(collection_name, points_selector, ids, **kwargs)

    def scroll(
        self,
        collection_name: str,
        limit: int = 10,
        offset: Optional[Union[str, int]] = None,
        query_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[List[Dict[str, Any]], Optional[Union[str, int]]]:
        """Scroll through points in collection.
        
        Args:
            collection_name: Name of the collection
            limit: Number of points to return
            offset: Offset for pagination
            query_filter: Filter conditions
            **kwargs: Additional scroll parameters
            
        Returns:
            Tuple of (points, next_offset)
        """
        return self.data.scroll(collection_name, limit, offset, query_filter, **kwargs)

    def health_check(self) -> bool:
        """Check if Qdrant server is reachable.
        
        Returns:
            True if server is healthy, False otherwise
        """
        return self.connection.health_check()

    def close(self) -> None:
        """Close connection to Qdrant."""
        self.connection.close()


__all__ = ["QdrantService"]
