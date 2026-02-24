"""Qdrant data operations (insert, search, delete)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from quaerium.infra.vectorstores.qdrant.exceptions import DataOperationError

if TYPE_CHECKING:
    from quaerium.infra.vectorstores.qdrant.connection import QdrantConnectionManager


class QdrantDataManager:
    """Manage data operations in Qdrant collections."""

    def __init__(self, connection: QdrantConnectionManager) -> None:
        """Initialize data manager.
        
        Args:
            connection: Qdrant connection manager
        """
        self.connection = connection

    def upsert(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        """Insert or update points in collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points with id, vector, and payload
            **kwargs: Additional upsert parameters
            
        Returns:
            List of point IDs
            
        Raises:
            DataOperationError: If upsert fails
        """
        try:
            from qdrant_client.models import PointStruct

            # Convert to PointStruct
            qdrant_points = []
            point_ids = []
            
            for point in points:
                point_id = point.get("id", str(uuid4()))
                point_ids.append(point_id)
                
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=point["vector"],
                        payload=point.get("payload", {}),
                    )
                )

            self.connection.client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                **kwargs,
            )
            
            return point_ids
        except Exception as exc:
            raise DataOperationError(f"Failed to upsert points: {exc}") from exc

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Number of results to return
            query_filter: Filter conditions for search
            score_threshold: Minimum score threshold
            **kwargs: Additional search parameters
            
        Returns:
            List of search results with id, score, vector, and payload
            
        Raises:
            DataOperationError: If search fails
        """
        try:
            from qdrant_client.models import Filter as QdrantFilter

            # Convert filter dict to Qdrant Filter if provided
            qdrant_filter = None
            if query_filter:
                qdrant_filter = self._build_filter(query_filter)

            # Use query() instead of search() in newer qdrant-client versions
            results = self.connection.client.query_points(
                collection_name=collection_name,
                query=list(query_vector),
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=True,
                **kwargs,
            ).points

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "vector": result.vector,
                    "payload": result.payload,
                }
                for result in results
            ]
        except Exception as exc:
            raise DataOperationError(f"Failed to search: {exc}") from exc

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
            
        Raises:
            DataOperationError: If batch search fails
        """
        try:
            from qdrant_client.models import Filter as QdrantFilter, SearchRequest

            qdrant_filter = None
            if query_filter:
                qdrant_filter = self._build_filter(query_filter)

            # Build search requests
            requests = [
                SearchRequest(
                    vector=list(vec),
                    limit=limit,
                    filter=qdrant_filter,
                    **kwargs,
                )
                for vec in query_vectors
            ]

            results = self.connection.client.search_batch(
                collection_name=collection_name,
                requests=requests,
            )

            return [
                [
                    {
                        "id": result.id,
                        "score": result.score,
                        "vector": result.vector,
                        "payload": result.payload,
                    }
                    for result in batch_results
                ]
                for batch_results in results
            ]
        except Exception as exc:
            raise DataOperationError(f"Failed to batch search: {exc}") from exc

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
            
        Raises:
            DataOperationError: If retrieval fails
        """
        try:
            results = self.connection.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                **kwargs,
            )

            return [
                {
                    "id": result.id,
                    "vector": result.vector,
                    "payload": result.payload,
                }
                for result in results
            ]
        except Exception as exc:
            raise DataOperationError(f"Failed to retrieve points: {exc}") from exc

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
            
        Raises:
            DataOperationError: If deletion fails
        """
        try:
            from qdrant_client.models import PointIdsList

            if ids:
                # Delete by IDs
                self.connection.client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=ids),
                    **kwargs,
                )
            elif points_selector:
                # Delete by filter
                qdrant_filter = self._build_filter(points_selector)
                self.connection.client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_filter,
                    **kwargs,
                )
            else:
                raise DataOperationError("Either 'ids' or 'points_selector' must be provided")
        except Exception as exc:
            raise DataOperationError(f"Failed to delete points: {exc}") from exc

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
            
        Raises:
            DataOperationError: If scroll fails
        """
        try:
            qdrant_filter = None
            if query_filter:
                qdrant_filter = self._build_filter(query_filter)

            results, next_offset = self.connection.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                **kwargs,
            )

            points = [
                {
                    "id": result.id,
                    "vector": result.vector,
                    "payload": result.payload,
                }
                for result in results
            ]

            return points, next_offset
        except Exception as exc:
            raise DataOperationError(f"Failed to scroll: {exc}") from exc

    def _build_filter(self, filter_dict: Dict[str, Any]) -> Any:
        """Convert filter dictionary to Qdrant Filter object.
        
        Args:
            filter_dict: Filter conditions as dictionary
            
        Returns:
            Qdrant Filter object
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            Range,
        )

        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Range or complex condition
                if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("$gte"),
                                lte=value.get("$lte"),
                                gt=value.get("$gt"),
                                lt=value.get("$lt"),
                            ),
                        )
                    )
                elif "$in" in value:
                    # Multiple values - create multiple match conditions
                    for v in value["$in"]:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=v),
                            )
                        )
            else:
                # Simple equality match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=conditions) if conditions else None


__all__ = ["QdrantDataManager"]
