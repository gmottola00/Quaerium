"""High-level facade for ChromaDB vector operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
from quaerium.infra.vectorstores.chroma.config import ChromaConfig, ChromaIndexConfig
from quaerium.infra.vectorstores.chroma.connection import ChromaConnectionManager
from quaerium.infra.vectorstores.chroma.data import ChromaDataManager


class ChromaService:
    """Compose ChromaDB managers into a cohesive service."""

    def __init__(self, config: ChromaConfig) -> None:
        """Initialize ChromaDB service.
        
        Args:
            config: ChromaDB connection configuration
        """
        self.connection = ChromaConnectionManager(config)
        self.collections = ChromaCollectionManager(self.connection)
        self.data = ChromaDataManager(self.connection)

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
            **kwargs: Additional parameters
            
        Returns:
            ChromaDB collection object
        """
        return self.collections.create_collection(
            collection_name, index_config, metadata, **kwargs
        )

    def get_or_create_collection(
        self,
        collection_name: str,
        index_config: Optional[ChromaIndexConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Get or create a collection.
        
        Args:
            collection_name: Name of the collection
            index_config: Index configuration
            metadata: Collection metadata
            **kwargs: Additional parameters
            
        Returns:
            ChromaDB collection object
        """
        return self.collections.get_or_create_collection(
            collection_name, index_config, metadata, **kwargs
        )

    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        self.collections.drop_collection(collection_name)

    def add(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs (auto-generated if None)
            embeddings: Vector embeddings
            documents: Document texts
            metadatas: Document metadata
            **kwargs: Additional parameters
            
        Returns:
            List of document IDs
        """
        return self.data.add(
            collection_name, ids, embeddings, documents, metadatas, **kwargs
        )

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upsert documents in collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs
            embeddings: Vector embeddings
            documents: Document texts
            metadatas: Document metadata
            **kwargs: Additional parameters
            
        Returns:
            List of document IDs
        """
        return self.data.upsert(
            collection_name, ids, embeddings, documents, metadatas, **kwargs
        )

    def query(
        self,
        collection_name: str,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Query collection for similar documents.
        
        Args:
            collection_name: Name of the collection
            query_embeddings: Query vectors
            query_texts: Query texts
            n_results: Number of results
            where: Metadata filter
            where_document: Document filter
            **kwargs: Additional parameters
            
        Returns:
            Query results
        """
        return self.data.query(
            collection_name,
            query_embeddings,
            query_texts,
            n_results,
            where,
            where_document,
            **kwargs,
        )

    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get documents from collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs
            where: Metadata filter
            where_document: Document filter
            limit: Max results
            offset: Skip results
            **kwargs: Additional parameters
            
        Returns:
            Retrieved documents
        """
        return self.data.get(
            collection_name, ids, where, where_document, limit, offset, **kwargs
        )

    def update(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Update documents in collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs
            embeddings: New embeddings
            documents: New documents
            metadatas: New metadata
            **kwargs: Additional parameters
        """
        self.data.update(collection_name, ids, embeddings, documents, metadatas, **kwargs)

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete documents from collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs
            where: Metadata filter
            where_document: Document filter
            **kwargs: Additional parameters
        """
        self.data.delete(collection_name, ids, where, where_document, **kwargs)

    def count(self, collection_name: str) -> int:
        """Get document count in collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Document count
        """
        return self.data.count(collection_name)

    def health_check(self) -> bool:
        """Check if ChromaDB is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        return self.connection.health_check()

    def close(self) -> None:
        """Close connection to ChromaDB."""
        self.connection.close()


__all__ = ["ChromaService"]
