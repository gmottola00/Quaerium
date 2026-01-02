# Architecture Overview

rag-toolkit is designed with a clean, layered architecture that emphasizes testability, extensibility, and maintainability.

## Design Principles

### 1. Protocol-Based Design

We use Python Protocols instead of abstract base classes:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingClient(Protocol):
    """Any class with this signature is an EmbeddingClient."""
    def embed(self, text: str) -> list[float]: ...
```

**Benefits**:
- ✅ No inheritance required
- ✅ Duck typing with type safety
- ✅ Easy mocking for tests
- ✅ Flexible implementations

### 2. Dependency Injection

Components receive their dependencies explicitly:

```python
pipeline = RagPipeline(
    embedding_client=my_embedding,  # Injected
    llm_client=my_llm,              # Injected
    vector_store=my_store,          # Injected
)
```

**Benefits**:
- ✅ Easy to test with mocks
- ✅ Clear dependencies
- ✅ Runtime flexibility
- ✅ No global state

### 3. Layered Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│   (Your RAG applications)               │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         RAG Pipeline Layer              │
│   (Orchestration & Business Logic)      │
│   - RagPipeline                         │
│   - QueryRewriter                       │
│   - ContextAssembler                    │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Infrastructure Layer               │
│   (Concrete Implementations)            │
│   - OllamaEmbedding                     │
│   - OpenAILLMClient                     │
│   - MilvusVectorStore                   │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           Core Layer                    │
│   (Protocol Definitions)                │
│   - EmbeddingClient                     │
│   - LLMClient                           │
│   - VectorStoreClient                   │
└─────────────────────────────────────────┘
```

## Core Layer

The foundation of rag-toolkit. Contains only protocol definitions with **zero external dependencies**.

### EmbeddingClient Protocol

```python
@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding text into vectors."""
    
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation as list of floats
        """
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vectors
        """
        ...
```

### LLMClient Protocol

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
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options
            
        Returns:
            Generated text
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

### VectorStoreClient Protocol

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
        """Create a new collection for vectors."""
        ...
    
    def insert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
        **kwargs
    ) -> list[str]:
        """Insert vectors into collection.
        
        Returns:
            List of IDs for inserted vectors
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
        """Search for similar vectors.
        
        Returns:
            List of SearchResult objects
        """
        ...
```

## Infrastructure Layer

Concrete implementations of core protocols.

### Embedding Implementations

- **OllamaEmbedding**: Local embedding models via Ollama
- **OpenAIEmbedding**: OpenAI's embedding API

```python
from rag_toolkit.infra.embedding.ollama import OllamaEmbedding

embedding = OllamaEmbedding(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)

# Implements EmbeddingClient Protocol
vector = embedding.embed("Hello world")
```

### LLM Implementations

- **OllamaLLMClient**: Local LLMs via Ollama
- **OpenAILLMClient**: OpenAI's GPT models

```python
from rag_toolkit.infra.llm.ollama import OllamaLLMClient

llm = OllamaLLMClient(
    base_url="http://localhost:11434",
    model="llama2"
)

# Implements LLMClient Protocol
response = llm.generate("Explain RAG")
```

### Vector Store Implementations

- **MilvusVectorStore**: Milvus vector database
- **PineconeVectorStore**: Coming soon
- **QdrantVectorStore**: Coming soon

```python
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

store = MilvusVectorStore(
    host="localhost",
    port="19530",
    collection_name="my_docs"
)

# Implements VectorStoreClient Protocol
store.create_collection("docs", dimension=768)
```

## RAG Pipeline Layer

High-level orchestration and business logic.

### RagPipeline

The main entry point for RAG operations:

```python
from rag_toolkit.rag.pipeline import RagPipeline

pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
    chunk_size=512,
    chunk_overlap=50,
)

# Orchestrates: chunk → embed → store
pipeline.index_documents(documents)

# Orchestrates: embed → search → rerank → generate
response = pipeline.query("What is RAG?")
```

### QueryRewriter

Enhances queries before retrieval:

```python
from rag_toolkit.rag.rewriter import QueryRewriter

rewriter = QueryRewriter(llm_client=llm)

# HyDE (Hypothetical Document Embeddings)
expanded = rewriter.rewrite_hyde(
    query="What is RAG?",
    num_variations=3
)

# Multi-query generation
queries = rewriter.generate_multi_query(
    query="Explain embeddings",
    num_queries=5
)
```

### ContextAssembler

Assembles retrieved chunks into context:

```python
from rag_toolkit.rag.assembler import ContextAssembler

assembler = ContextAssembler(
    max_context_length=2048,
    include_metadata=True
)

context = assembler.assemble(
    chunks=retrieved_chunks,
    query="What is RAG?",
    format="markdown"
)
```

## Data Flow

### Indexing Flow

```
Documents
    │
    ▼
[Chunking]
    │
    ▼
Chunks
    │
    ▼
[Embedding] ──► EmbeddingClient
    │
    ▼
Vectors
    │
    ▼
[Storage] ──► VectorStoreClient
    │
    ▼
Indexed Data
```

### Query Flow

```
Query
    │
    ▼
[Query Rewriting] ──► QueryRewriter
    │
    ▼
Enhanced Query
    │
    ▼
[Embedding] ──► EmbeddingClient
    │
    ▼
Query Vector
    │
    ▼
[Search] ──► VectorStoreClient
    │
    ▼
Retrieved Chunks
    │
    ▼
[Reranking] ──► Reranker (optional)
    │
    ▼
Ranked Chunks
    │
    ▼
[Context Assembly] ──► ContextAssembler
    │
    ▼
Context
    │
    ▼
[Generation] ──► LLMClient
    │
    ▼
Response
```

## Extension Points

### Custom Embedding Provider

```python
class MyEmbeddingProvider:
    """Custom embedding - no inheritance needed!"""
    
    def embed(self, text: str) -> list[float]:
        # Your implementation
        return [0.1, 0.2, ...]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        return [[0.1, ...], [0.2, ...]]

# Works with RagPipeline!
pipeline = RagPipeline(
    embedding_client=MyEmbeddingProvider(),
    llm_client=llm,
    vector_store=store,
)
```

### Custom Vector Store

```python
class MyVectorDB:
    """Custom vector store - Protocol-based!"""
    
    def create_collection(self, name, dimension, **kwargs):
        # Connect to your DB
        pass
    
    def insert(self, collection, vectors, texts, metadata):
        # Store vectors
        return ["id1", "id2"]
    
    def search(self, collection, query_vector, top_k):
        # Search and return SearchResult objects
        return [SearchResult(...)]

# Seamless integration!
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=MyVectorDB(),  # ✅ It just works
)
```

## Testing Strategy

### Unit Tests

Mock protocols easily:

```python
class MockEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.0] * 768

def test_pipeline():
    pipeline = RagPipeline(
        embedding_client=MockEmbedding(),  # Easy mock!
        llm_client=MockLLM(),
        vector_store=MockVectorStore(),
    )
    
    result = pipeline.query("test")
    assert result is not None
```

### Integration Tests

Test with real services in CI/CD:

```python
@pytest.mark.integration
def test_milvus_integration():
    store = MilvusVectorStore(
        host="localhost",
        port="19530"
    )
    
    # Test real operations
    store.create_collection("test", 768)
    ids = store.insert("test", vectors, texts, metadata)
    results = store.search("test", query_vector, top_k=5)
    
    assert len(results) == 5
```

## Performance Considerations

### Batch Processing

Always prefer batch operations:

```python
# ❌ Slow: One at a time
for text in texts:
    vector = embedding.embed(text)

# ✅ Fast: Batch processing
vectors = embedding.embed_batch(texts)
```

### Async Operations

Use async for I/O-bound operations:

```python
# Async LLM calls
response = await llm.agenerate(prompt)

# Concurrent queries
tasks = [llm.agenerate(p) for p in prompts]
responses = await asyncio.gather(*tasks)
```

### Caching

Cache embeddings for repeated queries:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(text: str) -> tuple[float, ...]:
    return tuple(embedding.embed(text))
```

## Next Steps

- Learn about [Core Concepts](user_guide/core_concepts.md)
- Understand [Protocols](user_guide/protocols.md)
- See [Implementation Examples](examples/index.md)
- Read [API Documentation](autoapi/index.html)
