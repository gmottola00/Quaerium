"""Generic index service using dependency injection.

This orchestrator manages vector indexing and search operations
through abstract protocols, enabling swappable implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from quaerium.core.index.base import VectorStoreService


class IndexService:
    """Generic vector index service with dependency injection.
    
    This service orchestrates indexing and search operations through
    a VectorStoreService interface, remaining agnostic to the specific
    vector database implementation (Milvus, Pinecone, etc).
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_dim: int,
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        collection_name: str,
        metric_type: str = "IP",
        index_type: str = "HNSW",
    ) -> None:
        """Initialize index service with injected dependencies.
        
        Args:
            vector_store: The vector store implementation.
            embedding_dim: Dimensionality of embeddings.
            embed_fn: Function to generate embeddings from text.
            collection_name: Name of the collection to use.
            metric_type: Distance metric (IP, L2, COSINE).
            index_type: Index type (HNSW, IVF_FLAT, etc).
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        self.vector_store = vector_store
        self.embedding_dim = embedding_dim
        self.embed_fn = embed_fn
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.index_type = index_type

    def ensure_collection(self, schema: Any, index_params: Optional[Dict[str, Any]] = None) -> None:
        """Ensure collection exists with schema and index.
        
        Args:
            schema: Collection schema (implementation-specific).
            index_params: Optional index parameters.
        """
        self.vector_store.connection.ensure()
        self.vector_store.ensure_collection(
            name=self.collection_name,
            schema=schema,
            index_params=index_params,
            load=True,
            shards_num=2,
        )

    def upsert(self, data: Sequence[Dict[str, Any]]) -> None:
        """Upsert data into the collection.
        
        Args:
            data: Sequence of dictionaries with entity data.
        """
        if not data:
            return
        
        self.vector_store.data.upsert(self.collection_name, data)
        self.vector_store.data.flush(self.collection_name)

    def search(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        expr: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Search collection by embedding vector.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            output_fields: Fields to return in results.
            search_params: Search-specific parameters.
            expr: Filter expression.
            
        Returns:
            List of result dictionaries with scores and fields.
        """
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dim mismatch: expected {self.embedding_dim}")

        params = search_params or {"metric_type": self.metric_type, "params": {}}
        
        results = self.vector_store.data.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            param=params,
            limit=top_k,
            output_fields=output_fields or [],
            expr=expr,
        )
        
        return self._parse_search_results(results)

    def query(
        self,
        expr: str,
        *,
        output_fields: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, object]]:
        """Execute scalar query on collection.
        
        Args:
            expr: Filter expression.
            output_fields: Fields to return.
            limit: Max results.
            
        Returns:
            List of matching entities.
        """
        return self.vector_store.data.query(
            collection_name=self.collection_name,
            expr=expr,
            output_fields=output_fields,
            limit=limit,
        )

    def drop_collection(self) -> None:
        """Drop the collection."""
        self.vector_store.drop_collection(self.collection_name)

    @staticmethod
    def _parse_search_results(results: Any) -> List[Dict[str, object]]:
        """Parse implementation-specific search results into standard format.
        
        This method handles different result formats from various vector stores.
        Override or extend for custom parsing logic.
        
        Args:
            results: Raw search results from vector store.
            
        Returns:
            Normalized list of result dictionaries.
        """
        hits: List[Dict[str, object]] = []
        
        # Handle Milvus-style results (list of lists)
        if isinstance(results, list) and results:
            for hit in results[0]:
                entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
                
                # Extract score
                score = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", None)
                
                # Build result dict
                result = {"score": score}
                
                # Add all entity fields
                if isinstance(entity, dict):
                    result.update(entity)
                else:
                    # Handle object-based entities
                    for attr in dir(entity):
                        if not attr.startswith("_"):
                            result[attr] = getattr(entity, attr, None)
                
                hits.append(result)
        
        return hits


__all__ = ["IndexService"]
