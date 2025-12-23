"""Parameterized tests for all vector stores.

This module provides pytest parameterization to run the same tests
across multiple vector store implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

import pytest


@pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chroma"])
class TestUnifiedVectorStoreCollections:
    """Test collection operations across all vector stores."""
    
    def test_create_and_drop_collection(
        self,
        vector_store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test creating and dropping collections."""
        service = request.getfixturevalue(f"{vector_store_name}_service")
        collection_name = f"test_collection_{vector_store_name}"
        
        # Create collection
        if vector_store_name == "milvus":
            # Milvus requires schema
            schema = {
                "id": {"dtype": "INT64", "is_primary": True, "auto_id": True},
                "vector": {"dtype": "FLOAT_VECTOR", "dim": 384},
            }
            service.ensure_collection(collection_name, schema)
        elif vector_store_name == "qdrant":
            service.ensure_collection(collection_name, vector_size=384)
        elif vector_store_name == "chroma":
            service.create_collection(collection_name)
        
        # Verify collection exists
        if vector_store_name == "milvus":
            collections = service.list_collections()
            assert collection_name in collections
        elif vector_store_name == "qdrant":
            assert service.collections.collection_exists(collection_name)
        elif vector_store_name == "chroma":
            assert service.collections.collection_exists(collection_name)
        
        # Drop collection
        service.drop_collection(collection_name)
        
        # Verify collection is gone
        if vector_store_name == "milvus":
            collections = service.list_collections()
            assert collection_name not in collections
        elif vector_store_name == "qdrant":
            assert not service.collections.collection_exists(collection_name)
        elif vector_store_name == "chroma":
            assert not service.collections.collection_exists(collection_name)
    
    def test_list_collections(
        self,
        vector_store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test listing collections."""
        service = request.getfixturevalue(f"{vector_store_name}_service")
        
        if vector_store_name == "milvus":
            collections = service.list_collections()
        else:
            collections = service.collections.list_collections()
        
        assert isinstance(collections, list)


@pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chroma"])
class TestUnifiedVectorStoreData:
    """Test data operations across all vector stores."""
    
    @pytest.fixture
    def sample_vectors(self) -> List[List[float]]:
        """Generate sample vectors."""
        import random
        random.seed(42)
        return [[random.random() for _ in range(384)] for _ in range(3)]
    
    @pytest.fixture
    def sample_ids(self) -> List[str]:
        """Generate sample IDs."""
        return [str(uuid4()) for _ in range(3)]
    
    @pytest.fixture
    def sample_metadata(self) -> List[Dict[str, Any]]:
        """Generate sample metadata."""
        return [
            {"source": "test", "page": 1},
            {"source": "test", "page": 2},
            {"source": "test", "page": 3},
        ]
    def test_add_and_search_vectors(
        self,
        vector_store_name: str,
        request: pytest.FixtureRequest,
        sample_vectors: List[List[float]],
        sample_ids: List[str],
        sample_metadata: List[Dict[str, Any]],
    ) -> None:
        """Test adding and searching vectors."""
        service = request.getfixturevalue(f"{vector_store_name}_service")
        collection_name = f"test_data_{vector_store_name}"
        
        # Create collection
        if vector_store_name == "milvus":
            # Milvus requires schema with INT64 IDs
            schema = {
                "id": {"dtype": "INT64", "is_primary": True, "auto_id": False},
                "vector": {"dtype": "FLOAT_VECTOR", "dim": 384},
                "text": {"dtype": "VARCHAR", "max_length": 1000},
            }
            service.ensure_collection(collection_name, schema)
            # Insert data - Milvus uses list of dicts
            data = [
                {
                    "id": i + 1,
                    "vector": sample_vectors[i],
                    "text": f"doc_{i}",
                }
                for i in range(len(sample_vectors))
            ]
            service.insert(collection_name, data)
        elif vector_store_name == "qdrant":
            service.ensure_collection(collection_name, vector_size=384)
            points = [
                {
                    "id": sample_ids[i],
                    "vector": sample_vectors[i],
                    "payload": sample_metadata[i],
                }
                for i in range(len(sample_ids))
            ]
            service.data.upsert(collection_name, points)
        elif vector_store_name == "chroma":
            service.get_or_create_collection(collection_name)
            service.add(
                collection_name=collection_name,
                ids=sample_ids,
                embeddings=sample_vectors,
                metadatas=sample_metadata,
            )
        
        # Search
        query_vector = sample_vectors[0]
        
        if vector_store_name == "milvus":
            results = service.search(
                collection_name=collection_name,
                vectors=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {}},  # Use IP to match index
                limit=3,
            )
            assert len(results) > 0
            assert len(results[0]) > 0
    def test_count_vectors(
        self,
        vector_store_name: str,
        request: pytest.FixtureRequest,
        sample_vectors: List[List[float]],
        sample_ids: List[str],
    ) -> None:
        """Test counting vectors."""
        service = request.getfixturevalue(f"{vector_store_name}_service")
        collection_name = f"test_count_{vector_store_name}"
        
        # Ensure clean slate - drop collection if exists
        try:
            service.drop_collection(collection_name)
        except Exception:
            pass
        
        # Create and populate collection
        if vector_store_name == "milvus":
            schema = {
                "id": {"dtype": "INT64", "is_primary": True, "auto_id": False},
                "vector": {"dtype": "FLOAT_VECTOR", "dim": 384},
            }
            service.ensure_collection(collection_name, schema)
            data = [
                {"id": i + 1, "vector": sample_vectors[i]}
                for i in range(len(sample_vectors))
            ]
            service.insert(collection_name, data)
            # Flush to ensure data is persisted (Milvus has eventual consistency)
            service.connection.client.flush(collection_name)
            count = service.count(collection_name)
        elif vector_store_name == "qdrant":
            service.ensure_collection(collection_name, vector_size=384)
            points = [
                {"id": sample_ids[i], "vector": sample_vectors[i], "payload": {}}
                for i in range(len(sample_ids))
            ]
            service.data.upsert(collection_name, points)
            count = service.collections.get_collection_info(collection_name)["points_count"]
        elif vector_store_name == "chroma":
            service.get_or_create_collection(collection_name)
            service.add(
                collection_name=collection_name,
                ids=sample_ids,
                embeddings=sample_vectors,
            )
            count = service.count(collection_name)
        
        assert count == 3
        
        # Cleanup
        service.drop_collection(collection_name)


@pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chroma"])
class TestUnifiedVectorStoreHealth:
    """Test health checks across all vector stores."""
    
    def test_health_check(
        self,
        vector_store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test health check works."""
        service = request.getfixturevalue(f"{vector_store_name}_service")
        
        if vector_store_name == "milvus":
            # Milvus doesn't have health_check method
            assert service.connection.health_check() is True
        else:
            assert service.health_check() is True


__all__ = [
    "TestUnifiedVectorStoreCollections",
    "TestUnifiedVectorStoreData",
    "TestUnifiedVectorStoreHealth",
]
