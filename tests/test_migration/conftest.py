"""Pytest fixtures for migration tests."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from quaerium.core.types import SearchResult


@pytest.fixture
def mock_source_store():
    """Mock source vector store with test data."""
    store = Mock()
    
    # Mock count
    store.count.return_value = 100
    
    # Mock search to return test vectors
    def mock_search(collection_name: str, query_vector: List[float], top_k: int):
        results = []
        for i in range(min(top_k, 10)):
            results.append(
                SearchResult(
                    id=f"source_{i}",
                    score=0.9 - i * 0.05,
                    text=f"Document {i} content",
                    metadata={"category": "test", "index": i},
                    vector=[0.1] * len(query_vector),
                )
            )
        return results
    
    store.search.side_effect = mock_search
    
    return store


@pytest.fixture
def mock_target_store():
    """Mock target vector store."""
    store = Mock()
    
    # Mock count (starts at 0, increases with add_vectors)
    # Using a list to make it mutable in nested function
    count_holder = [0]
    
    def mock_count(collection_name: str):
        return count_holder[0]
    
    def mock_add_vectors(
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
    ):
        count_holder[0] += len(vectors)
    
    store.count.side_effect = mock_count
    store.add_vectors.side_effect = mock_add_vectors
    
    return store


@pytest.fixture
def sample_vectors():
    """Sample vector data for testing."""
    return [
        {
            "id": "vec_1",
            "vector": [0.1, 0.2, 0.3],
            "text": "First document",
            "metadata": {"category": "test", "index": 1},
        },
        {
            "id": "vec_2",
            "vector": [0.4, 0.5, 0.6],
            "text": "Second document",
            "metadata": {"category": "test", "index": 2},
        },
        {
            "id": "vec_3",
            "vector": [0.7, 0.8, 0.9],
            "text": "Third document",
            "metadata": {"category": "prod", "index": 3},
        },
    ]
