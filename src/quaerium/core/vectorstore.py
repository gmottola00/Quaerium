"""
Core vector store abstraction.

This module defines the Protocol interface that all vector store implementations
must satisfy, enabling seamless switching between different vector databases
(Milvus, Pinecone, Qdrant, Weaviate, etc.) without changing application code.

Design Philosophy:
    - Protocol-based: No inheritance required, duck typing with type safety
    - Provider-agnostic: Works with any vector database
    - Minimal interface: Only essential operations
    - Metadata support: Rich filtering and search capabilities
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quaerium.core.types import SearchResult, VectorMetadata


@runtime_checkable
class VectorStoreClient(Protocol):
    """
    Protocol for vector store operations.
    
    This defines the interface that all vector store implementations must provide.
    It abstracts common operations across different vector databases.
    
    Implementations:
        - MilvusVectorStore: Milvus 2.x implementation
        - PineconeVectorStore: Pinecone cloud implementation (future)
        - QdrantVectorStore: Qdrant implementation (future)
    
    Example:
        >>> store: VectorStoreClient = MilvusVectorStore(host="localhost")
        >>> store.create_collection("docs", dimension=384)
        >>> ids = store.insert(
        ...     collection_name="docs",
        ...     vectors=[[0.1, 0.2, ...], ...],
        ...     texts=["doc 1", "doc 2"],
        ...     metadata=[{"source": "a"}, {"source": "b"}]
        ... )
        >>> results = store.search(
        ...     collection_name="docs",
        ...     query_vector=[0.1, 0.2, ...],
        ...     top_k=5
        ... )
    """

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: str = "IP",
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a new collection/index for storing vectors.
        
        Args:
            name: Collection name (unique identifier)
            dimension: Vector dimension (e.g., 384, 768, 1536)
            metric: Distance metric ("IP"=inner product, "L2"=euclidean, "COSINE")
            description: Optional collection description
            **kwargs: Provider-specific parameters
            
        Raises:
            CollectionExistsError: If collection already exists
            ValueError: If dimension <= 0 or invalid metric
            
        Example:
            >>> store.create_collection(
            ...     name="my_docs",
            ...     dimension=384,
            ...     metric="IP",
            ...     description="Product documentation embeddings"
            ... )
        """
        ...

    def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            True if collection exists, False otherwise
        """
        ...

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection and all its data.
        
        Args:
            name: Collection name
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            
        Warning:
            This operation is irreversible!
        """
        ...

    def insert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[VectorMetadata],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Insert vectors with associated text and metadata.
        
        Args:
            collection_name: Target collection
            vectors: List of embedding vectors (must match collection dimension)
            texts: Source texts for each vector
            metadata: Metadata dict for each vector (for filtering)
            ids: Optional custom IDs (auto-generated if None)
            **kwargs: Provider-specific parameters
            
        Returns:
            List of assigned IDs (in same order as input)
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            ValueError: If vectors/texts/metadata lengths don't match
            DimensionMismatchError: If vector dimension doesn't match collection
            
        Example:
            >>> ids = store.insert(
            ...     collection_name="docs",
            ...     vectors=[[0.1] * 384, [0.2] * 384],
            ...     texts=["First doc", "Second doc"],
            ...     metadata=[
            ...         {"source": "manual", "page": 1},
            ...         {"source": "api", "page": 5}
            ...     ]
            ... )
            >>> print(ids)  # ["id_001", "id_002"]
        """
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        *,
        filters: VectorMetadata | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Search for similar vectors using ANN (approximate nearest neighbors).
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"source": "manual"})
            **kwargs: Provider-specific parameters (e.g., nprobe, ef)
            
        Returns:
            List of search results ordered by similarity (best first)
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            DimensionMismatchError: If query dimension doesn't match collection
            
        Example:
            >>> results = store.search(
            ...     collection_name="docs",
            ...     query_vector=[0.15] * 384,
            ...     top_k=5,
            ...     filters={"source": "manual"}
            ... )
            >>> for result in results:
            ...     print(f"ID: {result.id}, Score: {result.score:.4f}")
            ...     print(f"Text: {result.text[:100]}...")
        """
        ...

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        *,
        alpha: float = 0.5,
        filters: VectorMetadata | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            query_text: Query text for keyword search
            top_k: Number of results to return
            alpha: Weight for vector vs keyword (0=keyword only, 1=vector only)
            filters: Metadata filters
            **kwargs: Provider-specific parameters
            
        Returns:
            List of search results with combined scores
            
        Note:
            Not all vector stores support hybrid search natively.
            Default implementation may combine separate vector + keyword searches.
            
        Example:
            >>> results = store.hybrid_search(
            ...     collection_name="docs",
            ...     query_vector=[0.15] * 384,
            ...     query_text="installation guide",
            ...     top_k=5,
            ...     alpha=0.7  # Favor vector search
            ... )
        """
        ...

    def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Collection name
            ids: List of vector IDs to delete
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            
        Example:
            >>> store.delete("docs", ids=["id_001", "id_003"])
        """
        ...

    def get_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dictionary with stats:
                - "count": Number of vectors
                - "dimension": Vector dimension
                - "metric": Distance metric
                - Additional provider-specific stats
                
        Example:
            >>> stats = store.get_stats("docs")
            >>> print(f"Collection has {stats['count']} vectors")
        """
        ...


__all__ = ["VectorStoreClient"]
