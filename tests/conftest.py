"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import pytest


# ============================================================================
# Concrete Implementations for Testing (Protocols need concrete classes)
# ============================================================================


@dataclass
class TestChunk:
    """Concrete implementation of ChunkLike for testing."""

    id: str
    title: str
    heading_level: int
    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)

    def to_dict(self, *, include_blocks: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "title": self.title,
            "heading_level": self.heading_level,
            "text": self.text,
            "page_numbers": self.page_numbers,
        }
        if include_blocks:
            data["blocks"] = self.blocks
        return data


@dataclass
class TestTokenChunk:
    """Concrete implementation of TokenChunkLike for testing."""

    id: str
    text: str
    section_path: str
    metadata: Dict[str, str] = field(default_factory=dict)
    page_numbers: List[int] = field(default_factory=list)
    source_chunk_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "section_path": self.section_path,
            "metadata": self.metadata,
            "page_numbers": self.page_numbers,
            "source_chunk_id": self.source_chunk_id,
        }


# ============================================================================
# Mock Implementations for Testing
# ============================================================================


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, model: str = "mock-model", dim: int = 384):
        self._model = model
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        """Return mock embedding based on text length."""
        # Simple deterministic mock: use text length to generate vector
        length_factor = len(text) / 100.0
        return [length_factor] * self._dim

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch embed with mock embeddings."""
        return [self.embed(text) for text in texts]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, model: str = "mock-llm"):
        self._model = model
        self._responses: List[str] = []

    def set_responses(self, responses: List[str]) -> None:
        """Set predefined responses for testing."""
        self._responses = responses

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return mock response or echo prompt."""
        if self._responses:
            return self._responses.pop(0)
        return f"Mock response to: {prompt[:50]}..."

    def generate_batch(self, prompts: Iterable[str], **kwargs: Any) -> Iterable[str]:
        """Batch generate with mock responses."""
        return [self.generate(p, **kwargs) for p in prompts]

    @property
    def model_name(self) -> str:
        return self._model


class MockVectorStoreClient:
    """Mock vector store client for testing."""

    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.data: Dict[str, List[Dict[str, Any]]] = {}

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: str = "IP",
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create mock collection."""
        self.collections[name] = {
            "dimension": dimension,
            "metric": metric,
            "description": description,
        }
        self.data[name] = []

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]] | None = None,
        ids: List[str] | None = None,
    ) -> List[str]:
        """Insert mock data."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        if metadata is None:
            metadata = [{} for _ in texts]
        if ids is None:
            ids = [f"id_{i}" for i in range(len(texts))]

        for i, (vec, text, meta, doc_id) in enumerate(
            zip(vectors, texts, metadata, ids)
        ):
            self.data[collection_name].append(
                {"id": doc_id, "vector": vec, "text": text, "metadata": meta}
            )

        return ids

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: str | None = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Mock search returning top_k results."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        # Return mock results (just first top_k items)
        results = []
        for i, item in enumerate(self.data[collection_name][:top_k]):
            results.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": 1.0 - (i * 0.1),  # Mock descending scores
                }
            )
        return results

    def list_collections(self) -> List[str]:
        """List all mock collections."""
        return list(self.collections.keys())

    def delete_collection(self, name: str) -> None:
        """Delete mock collection."""
        if name in self.collections:
            del self.collections[name]
            del self.data[name]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding() -> MockEmbeddingClient:
    """Provide mock embedding client."""
    return MockEmbeddingClient(model="test-embed", dim=384)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Provide mock LLM client."""
    return MockLLMClient(model="test-llm")


@pytest.fixture
def mock_vectorstore() -> MockVectorStoreClient:
    """Provide mock vector store client."""
    return MockVectorStoreClient()


@pytest.fixture
def sample_text() -> str:
    """Sample text for chunking tests."""
    return """# Introduction to RAG

Retrieval-Augmented Generation (RAG) combines retrieval and generation.

## Key Components

RAG systems have three main components:

### Vector Store
Stores document embeddings for efficient retrieval.

### Embedding Model
Converts text into vector representations.

### Language Model
Generates responses based on retrieved context.

## Benefits

RAG provides several advantages:
- Reduces hallucinations
- Enables knowledge updates without retraining
- Improves answer quality with citations
"""


@pytest.fixture
def sample_chunks() -> List[str]:
    """Sample chunks for testing."""
    return [
        "This is the first chunk about machine learning.",
        "The second chunk discusses natural language processing.",
        "Finally, the third chunk covers deep learning architectures.",
    ]
