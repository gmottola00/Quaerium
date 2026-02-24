"""Qdrant collection management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from quaerium.infra.vectorstores.qdrant.config import QdrantIndexConfig
from quaerium.infra.vectorstores.qdrant.exceptions import CollectionError

if TYPE_CHECKING:
    from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager


class QdrantCollectionManager:
    """Manage Qdrant collection operations."""

    def __init__(self, connection: QdrantConnectionManager) -> None:
        """Initialize collection manager.
        
        Args:
            connection: Qdrant connection manager
        """
        self.connection = connection

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        index_config: Optional[QdrantIndexConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            index_config: Index configuration
            **kwargs: Additional collection parameters
            
        Raises:
            CollectionError: If collection creation fails
        """
        try:
            from qdrant_client.models import Distance, VectorParams

            if index_config is None:
                index_config = QdrantIndexConfig()

            # Map distance string to Qdrant Distance enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            distance = distance_map.get(index_config.distance, Distance.COSINE)

            # Build vector params
            vector_params = VectorParams(
                size=vector_size,
                distance=distance,
                hnsw_config=index_config.hnsw_config,
                quantization_config=index_config.quantization,
                on_disk=index_config.on_disk,
            )

            self.connection.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                **kwargs,
            )
        except Exception as exc:
            raise CollectionError(f"Failed to create collection '{collection_name}': {exc}") from exc

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        index_config: Optional[QdrantIndexConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Ensure collection exists, create if not.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            index_config: Index configuration
            **kwargs: Additional collection parameters
        """
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size, index_config, **kwargs)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            return self.connection.client.collection_exists(collection_name)
        except Exception:
            return False

    def drop_collection(self, collection_name: str) -> None:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Raises:
            CollectionError: If collection deletion fails
        """
        try:
            if self.collection_exists(collection_name):
                self.connection.client.delete_collection(collection_name)
        except Exception as exc:
            raise CollectionError(f"Failed to drop collection '{collection_name}': {exc}") from exc

    def list_collections(self) -> List[str]:
        """List all collections.
        
        Returns:
            List of collection names
            
        Raises:
            CollectionError: If listing fails
        """
        try:
            collections = self.connection.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as exc:
            raise CollectionError(f"Failed to list collections: {exc}") from exc

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection metadata and statistics
            
        Raises:
            CollectionError: If retrieval fails
        """
        try:
            info = self.connection.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "status": info.status,
                "config": info.config,
            }
        except Exception as exc:
            raise CollectionError(f"Failed to get collection info for '{collection_name}': {exc}") from exc


__all__ = ["QdrantCollectionManager"]
