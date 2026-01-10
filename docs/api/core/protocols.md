# Core Protocols

Core protocol definitions for RAG Toolkit components.

## Overview

RAG Toolkit uses Python Protocols (PEP 544) for type safety without inheritance requirements. Any class implementing the protocol interface works seamlessly.

## Embedding Protocols

::: rag_toolkit.core.embedding.base.EmbeddingClient
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## LLM Protocols

::: rag_toolkit.core.llm.base.LLMClient
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Vector Store Protocols

::: rag_toolkit.core.vectorstore.VectorStoreClient
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Chunking Protocols

::: rag_toolkit.core.chunking.types.ChunkLike
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: rag_toolkit.core.chunking.types.TokenChunkLike
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Implementing EmbeddingClient

```python
from typing import List
from rag_toolkit.core import EmbeddingClient

class MyEmbedding(EmbeddingClient):
    """Custom embedding implementation."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # Your implementation here
        return [[0.1, 0.2, ...] for _ in texts]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return 384

# Use it anywhere EmbeddingClient is expected
embedding = MyEmbedding()
vectors = embedding.embed(["Hello", "World"])
```

### Implementing VectorStoreClient

```python
from typing import List, Dict, Any, Optional
from rag_toolkit.core import VectorStoreClient

class MyVectorStore(VectorStoreClient):
    """Custom vector store implementation."""
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to collection."""
        # Your implementation
        pass
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Your implementation
        return []

# Use it with any RAG Toolkit component
store = MyVectorStore()
pipeline = RagPipeline(vector_store=store, ...)
```

## Benefits of Protocols

!!! success "No Inheritance Required"
    Your classes don't need to inherit from base classes. Just implement the interface.

!!! tip "Type Safety"
    Full type checking support with mypy, pyright, and IDE autocomplete.

!!! example "Testing Friendly"
    Easy to mock for testing - just implement the protocol methods you need.

!!! gear "Extensible"
    Add new implementations without modifying core code or dealing with complex inheritance hierarchies.
