# :material-protocol: Protocols Guide

RAG Toolkit uses Python Protocols for maximum flexibility and type safety. This guide explains protocols and how to leverage them effectively.

---

## :material-help-circle: What Are Protocols?

!!! abstract "Structural Typing (Duck Typing)"
    Protocols define interfaces using **structural typing** rather than inheritance—if it walks like a duck and quacks like a duck, it's a duck!

### :material-close-circle: Traditional Approach

!!! failure "Inheritance-Based"
    ```python
    from abc import ABC, abstractmethod
    
    class VectorStore(ABC):
        @abstractmethod
        def search(self, query: list[float]) -> list:
            pass
    
    class MyStore(VectorStore):  # Must inherit ❌
        def search(self, query: list[float]) -> list:
            return []
    ```
    
    **Problems:**
    - Requires inheritance
    - Tight coupling
    - Difficult to mock
    - Limited flexibility

### :material-check-circle: Protocol Approach

!!! success "Structural Typing"
    ```python
    from typing import Protocol, runtime_checkable
    
    @runtime_checkable
    class VectorStoreClient(Protocol):
        def search(self, query: list[float]) -> list: ...
    
    class MyStore:  # No inheritance! ✓
        def search(self, query: list[float]) -> list:
            return []
    
    # Works! MyStore matches the protocol structure
    store: VectorStoreClient = MyStore()
    ```
    
    **Benefits:**
    - :material-puzzle-outline: No inheritance required
    - :material-duck: Duck typing with type safety
    - :material-test-tube: Easy mocking for tests
    - :material-swap-horizontal: Flexible implementations

!!! tip "Runtime Checking"
    The `@runtime_checkable` decorator enables `isinstance()` checks:
    ```python
    assert isinstance(MyStore(), VectorStoreClient)  # ✓ True
    ```

---

## :material-puzzle: Core Protocols

!!! info "The Three Pillars"
    RAG Toolkit defines three core protocols that form the foundation of the system.

### :material-vector-polyline: EmbeddingClient

!!! abstract "Text-to-Vector Transformation"

```python title="core/embedding/base.py" linenums="1" hl_lines="2 5 21"
@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for text embedding models."""
    
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text into a vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation as list of floats
            
        Example:
            >>> embedding = OllamaEmbedding()
            >>> vector = embedding.embed("Hello world")
            >>> len(vector)
            768
        """
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of vectors
            
        Example:
            >>> texts = ["Hello", "World"]
            >>> vectors = embedding.embed_batch(texts)
            >>> len(vectors)
            2
        """
        ...
```

!!! example "Implementation Example"
    ```python
    class MyEmbedding:
        def embed(self, text: str) -> list[float]:
            # Your embedding logic
            return [0.1, 0.2, 0.3, ...]
        
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            # Batch processing
            return [self.embed(t) for t in texts]
    
    # Works with RAG Pipeline! ✨
    pipeline = RagPipeline(
        embedding_client=MyEmbedding(),
        llm_client=llm,
        vector_store=store,
    )
    ```

### LLMClient

```python
@runtime_checkable
class LLMClient(Protocol):
    """Protocol for language model clients."""
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
            **kwargs: Provider-specific options
            
        Returns:
            Generated text
            
        Example:
            >>> llm = OllamaLLMClient()
            >>> response = llm.generate("Explain RAG in one sentence")
            >>> print(response)
            "RAG combines retrieval and generation..."
        """
        ...
    
    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Async version of generate."""
        ...
```

### VectorStoreClient

```python
@runtime_checkable
class VectorStoreClient(Protocol):
    """Protocol for vector store operations."""
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "IP",
        **kwargs
    ) -> None:
        """
        Create a new collection for storing vectors.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric ("IP", "L2", "COSINE")
            **kwargs: Store-specific options
            
        Example:
            >>> store.create_collection("docs", dimension=768)
        """
        ...
    
    def insert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
        **kwargs
    ) -> list[str]:
        """
        Insert vectors into collection.
        
        Args:
            collection_name: Target collection
            vectors: List of vector embeddings
            texts: Original texts
            metadata: Associated metadata
            **kwargs: Store-specific options
            
        Returns:
            List of IDs for inserted vectors
            
        Example:
            >>> ids = store.insert(
            ...     "docs",
            ...     vectors=[[0.1, ...], [0.2, ...]],
            ...     texts=["text1", "text2"],
            ...     metadata=[{"source": "doc1"}, {"source": "doc2"}]
            ... )
        """
        ...
    
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        **kwargs
    ) -> list[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            top_k: Number of results to return
            filters: Metadata filters
            **kwargs: Store-specific options
            
        Returns:
            List of SearchResult objects sorted by relevance
            
        Example:
            >>> results = store.search(
            ...     "docs",
            ...     query_vector=[0.1, ...],
            ...     top_k=5,
            ...     filters={"source": "manual"}
            ... )
        """
        ...
    
    def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        **kwargs
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector and keyword search.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            query_text: Query text for keyword search
            top_k: Number of results
            alpha: Weight between vector (0.0) and keyword (1.0) search
            **kwargs: Store-specific options
            
        Returns:
            List of SearchResult objects
        """
        ...
    
    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        ...
    
    def get_stats(self, collection_name: str) -> dict:
        """Get collection statistics."""
        ...
```

## Implementing Custom Protocols

### Example: Custom Embedding Provider

```python
class HuggingFaceEmbedding:
    """Custom embedding using HuggingFace models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> list[float]:
        """Implements EmbeddingClient.embed"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Implements EmbeddingClient.embed_batch"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

# Use it with RagPipeline - no inheritance needed!
embedding = HuggingFaceEmbedding()
pipeline = RagPipeline(
    embedding_client=embedding,  # ✅ Works!
    llm_client=llm,
    vector_store=store,
)
```

### Example: Custom Vector Store

```python
import chromadb
from quaerium.core.types import SearchResult

class ChromaVectorStore:
    """Custom vector store using ChromaDB."""
    
    def __init__(self, path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collections = {}
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "IP",
        **kwargs
    ) -> None:
        """Implements VectorStoreClient.create_collection"""
        self.collections[name] = self.client.create_collection(
            name=name,
            metadata={"dimension": dimension, "metric": metric}
        )
    
    def insert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
        **kwargs
    ) -> list[str]:
        """Implements VectorStoreClient.insert"""
        collection = self.collections[collection_name]
        ids = [f"{collection_name}_{i}" for i in range(len(vectors))]
        
        collection.add(
            embeddings=vectors,
            documents=texts,
            metadatas=metadata,
            ids=ids,
        )
        return ids
    
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        **kwargs
    ) -> list[SearchResult]:
        """Implements VectorStoreClient.search"""
        collection = self.collections[collection_name]
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters,
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(
                SearchResult(
                    id=results['ids'][0][i],
                    score=results['distances'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                )
            )
        
        return search_results
    
    def hybrid_search(self, *args, **kwargs) -> list[SearchResult]:
        """Optional: implement hybrid search"""
        raise NotImplementedError("ChromaDB doesn't support hybrid search")
    
    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Implements VectorStoreClient.delete"""
        collection = self.collections[collection_name]
        collection.delete(ids=ids)
    
    def get_stats(self, collection_name: str) -> dict:
        """Implements VectorStoreClient.get_stats"""
        collection = self.collections[collection_name]
        return {
            "count": collection.count(),
            "name": collection_name,
        }

# Use with RagPipeline seamlessly!
store = ChromaVectorStore()
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=store,  # ✅ Works perfectly!
)
```

### Example: Custom LLM Client

```python
class AnthropicLLMClient:
    """Custom LLM client for Anthropic Claude."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Implements LLMClient.generate"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Implements LLMClient.agenerate"""
        # Anthropic has async client
        async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        response = await async_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# Works with RagPipeline!
llm = AnthropicLLMClient(api_key="your-key")
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,  # ✅ Anthropic support!
    vector_store=store,
)
```

## Runtime Type Checking

Use `@runtime_checkable` for runtime validation:

```python
from typing import runtime_checkable

@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...

# Check if object implements protocol
class MyEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.0] * 768

embedding = MyEmbedding()
print(isinstance(embedding, EmbeddingClient))  # True ✅

# Missing method
class BadEmbedding:
    pass

bad = BadEmbedding()
print(isinstance(bad, EmbeddingClient))  # False ❌
```

## Testing with Protocols

Protocols make testing incredibly easy:

```python
import pytest
from quaerium import RagPipeline

class MockEmbedding:
    """Simple mock - no complex setup needed!"""
    def embed(self, text: str) -> list[float]:
        return [0.0] * 768
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]

class MockLLM:
    """Mock LLM that returns predictable responses."""
    def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        return "Mock async response"

class MockVectorStore:
    """Mock vector store for testing."""
    def create_collection(self, name, dimension, **kwargs): pass
    def insert(self, *args, **kwargs): return ["id1", "id2"]
    def search(self, *args, **kwargs): 
        return [
            SearchResult(id="1", score=0.9, text="result 1", metadata={}),
            SearchResult(id="2", score=0.8, text="result 2", metadata={}),
        ]
    def hybrid_search(self, *args, **kwargs): return []
    def delete(self, *args, **kwargs): pass
    def get_stats(self, *args, **kwargs): return {}

def test_rag_pipeline():
    """Test pipeline with mocks - super easy!"""
    pipeline = RagPipeline(
        embedding_client=MockEmbedding(),
        llm_client=MockLLM(),
        vector_store=MockVectorStore(),
    )
    
    response = pipeline.query("test query")
    assert response.answer == "Mock response"
```

## Best Practices

### 1. Keep Protocols Minimal

```python
# ✅ Good: Minimal required interface
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

# ❌ Bad: Too many requirements
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def get_model_name(self) -> str: ...
    def get_dimension(self) -> int: ...
    def fine_tune(self, data): ...  # Too specific!
```

### 2. Use Type Hints

```python
# ✅ Good: Clear type hints
def embed(self, text: str) -> list[float]: ...

# ❌ Bad: No type hints
def embed(self, text): ...
```

### 3. Document Expected Behavior

```python
class VectorStoreClient(Protocol):
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.
        
        Expected behavior:
        - Results should be sorted by relevance (highest first)
        - Should return at most `top_k` results
        - Empty list if no results found
        - Raise ValueError if collection doesn't exist
        """
        ...
```

### 4. Provide Default Implementations

```python
# Provide base classes for common use cases
class BaseEmbedding:
    """Optional base class with common functionality."""
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Default batch implementation using single embed."""
        return [self.embed(text) for text in texts]

# Users can inherit if useful, but don't have to!
class MyEmbedding(BaseEmbedding):  # Optional!
    def embed(self, text: str) -> list[float]:
        # Only implement embed, get embed_batch for free
        return [0.0] * 768
```

## Protocol Composition

Compose protocols for complex interfaces:

```python
class Searchable(Protocol):
    def search(self, query: str) -> list: ...

class Indexable(Protocol):
    def index(self, documents: list[str]) -> None: ...

class VectorStore(Searchable, Indexable, Protocol):
    """Combines both interfaces."""
    pass
```

## Next Steps

- See [Vector Stores](vector_stores.md) for implementation examples
- Learn about [Embeddings](embeddings.md)
- Explore [LLMs](llms.md)
- Read [API Reference](../api/index.md) for complete protocol definitions
