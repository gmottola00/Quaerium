"""ChromaDB collection management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from quaerium.infra.vectorstores.chroma.config import ChromaIndexConfig
from quaerium.infra.vectorstores.chroma.exceptions import CollectionError

if TYPE_CHECKING:
    from quaerium.infra.vectorstores.chroma.connection import ChromaConnectionManager


class ChromaCollectionManager:
    """Manage ChromaDB collection operations."""

    def __init__(self, connection: ChromaConnectionManager) -> None:
        """Initialize collection manager.
        
        Args:
            connection: ChromaDB connection manager
        """
        self.connection = connection

    def create_collection(
        self,
        collection_name: str,
        index_config: Optional[ChromaIndexConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            index_config: Index configuration
            metadata: Collection metadata
            **kwargs: Additional collection parameters
            
        Returns:
            ChromaDB collection object
            
        Raises:
            CollectionError: If collection creation fails
        """
        try:
            if index_config is None:
                index_config = ChromaIndexConfig()

            # Map distance to HNSW space
            hnsw_space = index_config.hnsw_space or index_config.distance
            
            # Build metadata
            collection_metadata = metadata or {}
            collection_metadata.update({
                "hnsw:space": hnsw_space,
                "hnsw:construction_ef": index_config.hnsw_construction_ef,
                "hnsw:search_ef": index_config.hnsw_search_ef,
                "hnsw:M": index_config.hnsw_M,
            })

            return self.connection.client.create_collection(
                name=collection_name,
                metadata=collection_metadata,
                **kwargs,
            )
        except Exception as exc:
            raise CollectionError(f"Failed to create collection '{collection_name}': {exc}") from exc

    def get_or_create_collection(
        self,
        collection_name: str,
        index_config: Optional[ChromaIndexConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Get existing collection or create if not exists.
        
        Args:
            collection_name: Name of the collection
            index_config: Index configuration
            metadata: Collection metadata
            **kwargs: Additional collection parameters
            
        Returns:
            ChromaDB collection object
        """
        try:
            if index_config is None:
                index_config = ChromaIndexConfig()

            hnsw_space = index_config.hnsw_space or index_config.distance
            
            collection_metadata = metadata or {}
            collection_metadata.update({
                "hnsw:space": hnsw_space,
                "hnsw:construction_ef": index_config.hnsw_construction_ef,
                "hnsw:search_ef": index_config.hnsw_search_ef,
                "hnsw:M": index_config.hnsw_M,
            })

            return self.connection.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata,
                **kwargs,
            )
        except Exception as exc:
            raise CollectionError(
                f"Failed to get or create collection '{collection_name}': {exc}"
            ) from exc

    def get_collection(self, collection_name: str) -> Any:
        """Get existing collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection object
            
        Raises:
            CollectionError: If collection not found
        """
        try:
            return self.connection.client.get_collection(name=collection_name)
        except Exception as exc:
            raise CollectionError(f"Failed to get collection '{collection_name}': {exc}") from exc

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            self.connection.client.get_collection(name=collection_name)
            return True
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
            self.connection.client.delete_collection(name=collection_name)
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
            collections = self.connection.client.list_collections()
            return [col.name for col in collections]
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
            collection = self.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception as exc:
            raise CollectionError(f"Failed to get collection info for '{collection_name}': {exc}") from exc


__all__ = ["ChromaCollectionManager"]
