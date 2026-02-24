"""Unified wrapper for vector store services."""

from __future__ import annotations

from typing import Any, List, Dict, Union
from quaerium.infra.vectorstores.milvus import MilvusService
from quaerium.infra.vectorstores.qdrant import QdrantService
from quaerium.infra.vectorstores.chroma import ChromaService


class VectorStoreWrapper:
    """Wrapper providing unified API for different vector stores."""
    
    def __init__(
        self,
        service: Union[MilvusService, QdrantService, ChromaService],
        collection_name: str,
        store_type: str,
    ):
        self.service = service
        self.collection_name = collection_name
        self.store_type = store_type
    
    def add_vectors(self, data: List[Dict[str, Any]], batch_size: int = 500) -> None:
        if len(data) > batch_size:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self._add_batch(batch)
        else:
            self._add_batch(data)
    
    def _add_batch(self, data: List[Dict[str, Any]]) -> None:
        if self.store_type == "milvus":
            milvus_data = [
                {
                    "id": item["id"],
                    "vector": item["vector"],
                    "text": item.get("text", ""),
                    "category": item.get("category", ""),
                    "score": item.get("score", 0),
                }
                for item in data
            ]
            self.service.insert(self.collection_name, milvus_data)
            self.service.connection.client.flush(self.collection_name)
            
        elif self.store_type == "qdrant":
            points = [
                {
                    "id": item["id"],
                    "vector": item["vector"],
                    "payload": {
                        k: v for k, v in item.items() 
                        if k not in ["id", "vector"]
                    },
                }
                for item in data
            ]
            self.service.data.upsert(self.collection_name, points)
            
        elif self.store_type == "chroma":
            ids = [str(item["id"]) for item in data]
            embeddings = [item["vector"] for item in data]
            metadatas = [
                {k: v for k, v in item.items() if k not in ["id", "vector"]}
                for item in data
            ]
            self.service.add(
                collection_name=self.collection_name,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Any]:
        if self.store_type == "milvus":
            results = self.service.search(
                collection_name=self.collection_name,
                vectors=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            )
            return results[0] if results else []
            
        elif self.store_type == "qdrant":
            results = self.service.data.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            return results
            
        elif self.store_type == "chroma":
            results = self.service.query(
                collection_name=self.collection_name,
                query_embeddings=[query_vector],
                n_results=top_k,
            )
            return results
    
    def delete_vectors(self, ids: List[Any], batch_size: int = 500) -> None:
        if len(ids) > batch_size:
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                self._delete_batch(batch)
        else:
            self._delete_batch(ids)
    
    def _delete_batch(self, ids: List[Any]) -> None:
        if self.store_type == "milvus":
            expr = f"id in {ids}"
            self.service.data.delete(
                collection_name=self.collection_name,
                expr=expr,
            )
            
        elif self.store_type == "qdrant":
            from qdrant_client.models import PointIdsList
            self.service.data.connection.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )
            
        elif self.store_type == "chroma":
            # ChromaDB delete through service
            self.service.delete(
                collection_name=self.collection_name,
                ids=[str(id) for id in ids],
            )
    
    def count(self) -> int:
        if self.store_type == "milvus":
            stats = self.service.collection.get_stats(self.collection_name)
            return stats.get("row_count", 0)
            
        elif self.store_type == "qdrant":
            info = self.service.collection.get_collection_info(self.collection_name)
            return info.points_count if info else 0
            
        elif self.store_type == "chroma":
            count = self.service.collection.count(self.collection_name)
            return count


__all__ = ["VectorStoreWrapper"]
