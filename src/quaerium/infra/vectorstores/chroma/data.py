"""ChromaDB data operations (add, query, update, delete)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from quaerium.infra.vectorstores.chroma.exceptions import DataOperationError

if TYPE_CHECKING:
    from quaerium.infra.vectorstores.chroma.connection import ChromaConnectionManager


class ChromaDataManager:
    """Manage data operations in ChromaDB collections."""

    def __init__(self, connection: ChromaConnectionManager) -> None:
        """Initialize data manager.
        
        Args:
            connection: ChromaDB connection manager
        """
        self.connection = connection

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
            
        Raises:
            DataOperationError: If add fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            # Generate IDs if not provided
            if ids is None:
                count = len(embeddings) if embeddings else len(documents) if documents else 0
                ids = [str(uuid4()) for _ in range(count)]
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                **kwargs,
            )
            
            return ids
        except Exception as exc:
            raise DataOperationError(f"Failed to add documents: {exc}") from exc

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upsert (insert or update) documents in collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs
            embeddings: Vector embeddings
            documents: Document texts
            metadatas: Document metadata
            **kwargs: Additional parameters
            
        Returns:
            List of document IDs
            
        Raises:
            DataOperationError: If upsert fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                **kwargs,
            )
            
            return ids
        except Exception as exc:
            raise DataOperationError(f"Failed to upsert documents: {exc}") from exc

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
            query_texts: Query texts (requires embedding function)
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            **kwargs: Additional query parameters
            
        Returns:
            Query results with ids, distances, documents, metadatas
            
        Raises:
            DataOperationError: If query fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            results = collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                **kwargs,
            )
            
            return results
        except Exception as exc:
            raise DataOperationError(f"Failed to query: {exc}") from exc

    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get documents from collection.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs to retrieve
            where: Metadata filter
            where_document: Document content filter
            limit: Maximum number of results
            offset: Number of results to skip
            include: Fields to include (embeddings, documents, metadatas, distances)
            **kwargs: Additional parameters
            
        Returns:
            Retrieved documents
            
        Raises:
            DataOperationError: If get fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            results = collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                limit=limit,
                offset=offset,
                include=include or ["embeddings", "documents", "metadatas"],
                **kwargs,
            )
            
            return results
        except Exception as exc:
            raise DataOperationError(f"Failed to get documents: {exc}") from exc

    def update(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Update existing documents.
        
        Args:
            collection_name: Name of the collection
            ids: Document IDs to update
            embeddings: New embeddings
            documents: New documents
            metadatas: New metadata
            **kwargs: Additional parameters
            
        Raises:
            DataOperationError: If update fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                **kwargs,
            )
        except Exception as exc:
            raise DataOperationError(f"Failed to update documents: {exc}") from exc

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
            ids: Document IDs to delete
            where: Metadata filter for deletion
            where_document: Document content filter for deletion
            **kwargs: Additional parameters
            
        Raises:
            DataOperationError: If deletion fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            if not ids and not where and not where_document:
                raise DataOperationError(
                    "Must provide at least one of: ids, where, or where_document"
                )
            
            collection.delete(
                ids=ids,
                where=where,
                where_document=where_document,
                **kwargs,
            )
        except Exception as exc:
            raise DataOperationError(f"Failed to delete documents: {exc}") from exc

    def count(self, collection_name: str) -> int:
        """Get number of documents in collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Document count
            
        Raises:
            DataOperationError: If count fails
        """
        try:
            from quaerium.infra.vectorstores.chroma.collection import ChromaCollectionManager
            
            collection_manager = ChromaCollectionManager(self.connection)
            collection = collection_manager.get_collection(collection_name)
            
            return collection.count()
        except Exception as exc:
            raise DataOperationError(f"Failed to count documents: {exc}") from exc


__all__ = ["ChromaDataManager"]
