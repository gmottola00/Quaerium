"""Generic search orchestrator using dependency injection.

This module provides abstract search implementations that work with
any IndexService, keeping search logic decoupled from vector stores.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from quaerium.core.index.base import Reranker, SearchStrategy
from quaerium.core.index.service import IndexService


class VectorSearch(SearchStrategy):
    """Generic vector search implementation."""

    def __init__(
        self,
        index_service: IndexService,
        embed_fn: Callable[[str], List[float]],
    ) -> None:
        """Initialize vector search.
        
        Args:
            index_service: Generic index service for vector operations.
            embed_fn: Function to convert text query to embedding.
        """
        self.index_service = index_service
        self.embed_fn = embed_fn

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> List[Dict[str, object]]:
        """Execute semantic vector search.
        
        Args:
            query: Text query.
            top_k: Number of results.
            output_fields: Fields to return.
            search_params: Vector store specific parameters.
            
        Returns:
            List of ranked results with scores.
        """
        query_embedding = self.embed_fn(query)
        return self.index_service.search(
            query_embedding=query_embedding,
            top_k=top_k,
            output_fields=output_fields,
            search_params=search_params,
        )


class KeywordSearch(SearchStrategy):
    """Generic keyword search using collection query."""

    def __init__(self, index_service: IndexService) -> None:
        """Initialize keyword search.
        
        Args:
            index_service: Generic index service for query operations.
        """
        self.index_service = index_service

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        text_field: str = "text",
        **kwargs,
    ) -> List[Dict[str, object]]:
        """Execute keyword search using LIKE expressions.
        
        Args:
            query: Text query.
            top_k: Number of results.
            text_field: Field name to search in.
            
        Returns:
            List of matching results.
        """
        expr = self._build_like_expression(query, text_field)
        results = self.index_service.query(
            expr=expr or "1 == 1",
            output_fields=None,  # Return all fields
            limit=top_k,
        )
        
        # Normalize to standard format
        return [
            {
                "score": None,  # Keyword search has no score
                **r,
            }
            for r in results
        ]

    @staticmethod
    def _build_like_expression(query: str, field: str) -> str:
        """Build LIKE expression for keyword matching."""
        terms = [t.strip() for t in query.split() if t.strip()]
        if not terms:
            return ""
        clauses = [f'{field} like "%{term}%"' for term in terms]
        return " and ".join(clauses)


class HybridSearch(SearchStrategy):
    """Hybrid search combining vector and keyword with optional reranking."""

    def __init__(
        self,
        vector_search: VectorSearch,
        keyword_search: KeywordSearch,
        reranker: Optional[Reranker] = None,
        *,
        alpha: float = 0.7,
    ) -> None:
        """Initialize hybrid search.
        
        Args:
            vector_search: Semantic search component.
            keyword_search: Keyword search component.
            reranker: Optional reranker (cross-encoder).
            alpha: Weight for vector score vs keyword (0..1).
        """
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.reranker = reranker
        self.alpha = alpha

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, object]]:
        """Execute hybrid search combining multiple strategies.
        
        Args:
            query: Text query.
            top_k: Number of final results.
            
        Returns:
            List of ranked results.
        """
        vec_results = self.vector_search.search(query, top_k=top_k)
        kw_results = self.keyword_search.search(query, top_k=top_k)

        combined = self._merge_results(vec_results, kw_results)
        
        if self.reranker:
            reranked = self.reranker.rerank(query, combined, top_k)
            return reranked[:top_k]
        
        return combined[:top_k]

    def _merge_results(
        self,
        vec_results: List[Dict[str, object]],
        kw_results: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Merge vector and keyword results with weighted scoring."""
        merged: Dict[str, Dict[str, object]] = {}

        def add_or_update(hit: Dict[str, object], score: float) -> None:
            hit_id = hit.get("id") or hit.get("source_chunk_id")
            if hit_id is None:
                return
            existing = merged.get(hit_id)
            if existing is None or score > existing.get("score", float("-inf")):
                updated = dict(hit)
                updated["score"] = score
                merged[hit_id] = updated

        # Normalize vector scores
        vec_scores = [h.get("score", 0.0) for h in vec_results if isinstance(h.get("score"), (int, float))]
        max_vec = max(vec_scores) if vec_scores else 1.0

        for hit in vec_results:
            raw_score = hit.get("score", 0.0)
            norm_score = float(raw_score) / max_vec if max_vec else 0.0
            add_or_update(hit, self.alpha * norm_score)

        for hit in kw_results:
            add_or_update(hit, (1 - self.alpha))

        return sorted(merged.values(), key=lambda h: h.get("score", 0.0), reverse=True)


class IdentityReranker:
    """No-op reranker that returns results unchanged."""

    def rerank(
        self,
        query: str,
        results: List[Dict[str, object]],
        top_k: int,
    ) -> List[Dict[str, object]]:
        """Return results without reranking."""
        return results


__all__ = [
    "VectorSearch",
    "KeywordSearch",
    "HybridSearch",
    "IdentityReranker",
]
