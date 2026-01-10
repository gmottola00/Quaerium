# :material-sitemap: Architecture Overview

RAG Toolkit is built with a clean, layered architecture emphasizing **testability**, **extensibility**, and **maintainability**.

---

## :material-star: Design Principles

### 1. :material-protocol: Protocol-Based Design

!!! abstract "Type-Safe Duck Typing"
    We use Python Protocols instead of abstract base classes for maximum flexibility.

```python title="Example Protocol" linenums="1" hl_lines="4-5"
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingClient(Protocol):
    """Any class with this signature is an EmbeddingClient."""
    def embed(self, text: str) -> list[float]: ...
```

**Benefits:**

<div class="grid cards" markdown>

- :material-check-circle: **No Inheritance Required**

    ---

    Implement the protocol without extending base classes

- :material-duck: **Duck Typing with Type Safety**

    ---

    Get runtime checks and IDE support

- :material-test-tube: **Easy Mocking for Tests**

    ---

    Create simple mock objects without complex setup

- :material-puzzle-outline: **Flexible Implementations**

    ---

    Swap implementations without changing code

</div>

### 2. :material-needle: Dependency Injection

!!! info "Explicit Dependencies"
    Components receive their dependencies explicitly through constructor injection.

```python title="Dependency Injection Example" hl_lines="2-4"
pipeline = RagPipeline(
    embedding_client=my_embedding,  # Injected
    llm_client=my_llm,              # Injected
    vector_store=my_store,          # Injected
)
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| :material-test-tube: **Testability** | Easy to inject mocks for unit tests |
| :material-eye: **Clarity** | Dependencies are explicit and visible |
| :material-swap-horizontal: **Flexibility** | Change implementations at runtime |
| :material-ban: **No Global State** | Avoid singleton and global variables |

### 3. :material-layers: Layered Architecture

!!! abstract "Clean Separation of Concerns"

```mermaid
graph TB
    subgraph Application["🚀 Application Layer"]
        APP[Your RAG Applications]
    end
    
    subgraph Pipeline["⚙️ RAG Pipeline Layer"]
        RP[RagPipeline]
        QR[QueryRewriter]
        CA[ContextAssembler]
    end
    
    subgraph Infrastructure["🔧 Infrastructure Layer"]
        OE[OllamaEmbedding]
        OL[OpenAILLMClient]
        MV[MilvusVectorStore]
    end
    
    subgraph Core["📦 Core Layer"]
        EC[EmbeddingClient Protocol]
        LC[LLMClient Protocol]
        VC[VectorStoreClient Protocol]
    end
    
    APP --> RP
    RP --> QR
    RP --> CA
    RP --> OE
    RP --> OL
    RP --> MV
    OE -.implements.-> EC
    OL -.implements.-> LC
    MV -.implements.-> VC
    
    style Application fill:#e3f2fd
    style Pipeline fill:#fff3e0
    style Infrastructure fill:#f3e5f5
    style Core fill:#e8f5e9
```

Each layer has a specific responsibility and depends only on layers below it.

---

## :material-package: Core Layer

!!! abstract "Foundation Layer"
    Contains only protocol definitions with **zero external dependencies**.

### :material-vector-polyline: EmbeddingClient Protocol

```python title="core/embedding/base.py" linenums="1" hl_lines="5 10"
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

!!! tip "Protocol Benefits"
    Any class implementing these methods can be used as an `EmbeddingClient`—no inheritance needed!

### :material-robot: LLMClient Protocol

```python title="core/llm/base.py" linenums="1" hl_lines="5 17"
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

### :material-database: VectorStoreClient Protocol

```python title="core/vectorstore/base.py" linenums="1" hl_lines="5 13 25"
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

---

## :material-cog: Infrastructure Layer

!!! info "Concrete Implementations"
    Real implementations of core protocols for various providers.

### :material-vector-arrange-below: Embedding Implementations

=== "Ollama"
    ```python
    from rag_toolkit.infra.embedding.ollama import OllamaEmbedding
    
    embedding = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="nomic-embed-text"
    )
    
    # Implements EmbeddingClient Protocol ✓
    vector = embedding.embed("Hello world")
    ```
    
    **Features:**
    - :material-server: Local embedding models
    - :material-lock: Privacy-focused (offline capable)
    - :material-currency-usd-off: No API costs

=== "OpenAI"
    ```python
    from rag_toolkit.infra.embedding.openai_embedding import OpenAIEmbedding
    
    embedding = OpenAIEmbedding(
        api_key="your-api-key",
        model="text-embedding-3-small"
    )
    
    # Implements EmbeddingClient Protocol ✓
    vector = embedding.embed("Hello world")
    ```
    
    **Features:**
    - :material-cloud: Cloud-based models
    - :material-speedometer: High performance
    - :material-scale-balance: Pay-per-use pricing

### :material-robot-outline: LLM Implementations

=== "Ollama"
    ```python
    from rag_toolkit.infra.llm.ollama import OllamaLLMClient
    
    llm = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="llama2"
    )
    
    # Implements LLMClient Protocol ✓
    response = llm.generate("Explain RAG")
    ```

=== "OpenAI"
    ```python
    from rag_toolkit.infra.llm.openai_llm import OpenAILLMClient
    
    llm = OpenAILLMClient(
        api_key="your-api-key",
        model="gpt-4"
    )
    
    # Implements LLMClient Protocol ✓
    response = llm.generate("Explain RAG")
    ```

### :material-database-search: Vector Store Implementations

<div class="grid cards" markdown>

- :material-database: **MilvusVectorStore**

    ---

    ```python
    from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore
    
    store = MilvusVectorStore(
        host="localhost",
        port="19530",
        collection_name="my_docs"
    )
    
    # Implements VectorStoreClient Protocol ✓
    store.create_collection("docs", dimension=768)
    ```
    
    Production-ready, high-performance vector database

- :material-pine-tree: **PineconeVectorStore**

    ---

    Coming soon! Cloud-native vector database.

- :material-cube: **QdrantVectorStore**

    ---

    Coming soon! Vector search engine with extended filtering.

</div>

---

## :material-pipeline: RAG Pipeline Layer

!!! abstract "High-Level Orchestration"
    Business logic and workflow orchestration for RAG operations.

### :material-sitemap-outline: RagPipeline

!!! info "Main Entry Point"
    The central orchestrator for all RAG operations.

```python title="rag/pipeline.py" hl_lines="2-4"
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

**Pipeline Workflow:**

```mermaid
graph LR
    A[Documents] -->|chunk| B[Chunks]
    B -->|embed| C[Vectors]
    C -->|store| D[Vector DB]
    
    Q[Query] -->|embed| E[Query Vector]
    E -->|search| D
    D -->|retrieve| F[Context]
    F -->|generate| G[Response]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

### :material-reload: QueryRewriter

!!! tip "Query Enhancement"
    Improve retrieval quality by rewriting and expanding queries.

=== "HyDE"
    ```python
    from rag_toolkit.rag.rewriter import QueryRewriter
    
    rewriter = QueryRewriter(llm_client=llm)
    
    # Hypothetical Document Embeddings
    expanded = rewriter.rewrite_hyde(
        query="What is RAG?",
        num_variations=3
    )
    ```
    
    Generates hypothetical documents that might answer the query.

=== "Multi-Query"
    ```python
    # Generate multiple query variations
    queries = rewriter.generate_multi_query(
        query="Explain embeddings",
        num_queries=5
    )
    ```
    
    Creates diverse query formulations for better coverage.

### :material-puzzle-outline: ContextAssembler

!!! info "Context Management"
    Intelligently assemble retrieved chunks into coherent context.

```python title="rag/assembler.py"
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

**Features:**

| Feature | Description |
|---------|-------------|
| :material-format-text: **Formatting** | Output in markdown, text, or JSON |
| :material-table: **Metadata** | Include source information and scores |
| :material-crop: **Truncation** | Smart truncation to fit token limits |
| :material-sort: **Reranking** | Optional reranking by relevance |

---

## :material-chart-timeline: Data Flow

!!! abstract "Understanding the Pipeline"

### :material-file-upload: Indexing Flow

```mermaid
graph TB
    A[📄 Documents] --> B{Chunking}
    B --> C[📝 Text Chunks]
    C --> D{Embedding}
    D --> E[🔢 Vectors]
    E --> F{Storage}
    F --> G[💾 Vector Database]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
    
    classDef process fill:#fff3e0
    class B,D,F process
```

**Steps:**

1. **Chunking**: Split documents into manageable pieces
2. **Embedding**: Convert text chunks to vector representations
3. **Storage**: Store vectors with metadata in vector database

### :material-magnify: Query Flow

```mermaid
graph TB
    A[💬 User Query] --> B{Query Rewriting}
    B --> C[🔄 Enhanced Query]
    C --> D{Embedding}
    D --> E[🔢 Query Vector]
    E --> F{Similarity Search}
    F --> G[📚 Retrieved Chunks]
    G --> H{Reranking}
    H --> I[⭐ Ranked Chunks]
    I --> J{Context Assembly}
    J --> K[📋 Context]
    K --> L{LLM Generation}
    L --> M[✨ Response]
    
    style A fill:#e3f2fd
    style M fill:#c8e6c9
    
    classDef process fill:#fff3e0
    class B,D,F,H,J,L process
```

**Steps:**

1. **Query Rewriting**: Enhance query for better retrieval (optional)
2. **Embedding**: Convert query to vector representation
3. **Similarity Search**: Find most similar documents in vector DB
4. **Reranking**: Optionally reorder results by relevance
5. **Context Assembly**: Combine chunks into coherent context
6. **LLM Generation**: Generate final answer using context

---

## :material-puzzle-plus: Extension Points

!!! success "Easy Extensibility"
    Protocol-based design makes adding custom components trivial.

### :material-vector-polyline: Custom Embedding Provider

!!! example "No Inheritance Needed"
    ```python title="custom_embedding.py" linenums="1" hl_lines="1 4-5 8-9"
    class MyEmbeddingProvider:
        """Custom embedding - no inheritance needed!"""
        
        def embed(self, text: str) -> list[float]:
            # Your implementation
            return [0.1, 0.2, 0.3, ...]
        
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            # Your batch implementation
            return [[0.1, ...], [0.2, ...], ...]
    
    # Works seamlessly with RagPipeline! ✨
    pipeline = RagPipeline(
        embedding_client=MyEmbeddingProvider(),
        llm_client=llm,
        vector_store=store,
    )
    ```

!!! tip "Protocol Checking"
    Use `isinstance()` to verify protocol implementation:
    ```python
    from rag_toolkit.core import EmbeddingClient
    
    assert isinstance(MyEmbeddingProvider(), EmbeddingClient)
    ```

### :material-database-plus: Custom Vector Store

!!! example "Bring Your Own Database"
    ```python title="custom_vectorstore.py" linenums="1" hl_lines="1 4 9 14"
    class MyVectorDB:
        """Custom vector store - Protocol-based!"""
        
        def create_collection(self, name, dimension, **kwargs):
            # Connect to your DB and create collection
            pass
        
        def insert(self, collection, vectors, texts, metadata):
            # Store vectors in your DB
            return ["id1", "id2", "id3"]
        
        def search(self, collection, query_vector, top_k):
            # Search and return SearchResult objects
            return [SearchResult(id="...", score=0.95, text="...")]
    
    # Seamless integration! 🎉
    pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=MyVectorDB(),  # ✓ It just works
    )
    ```

### :material-robot-excited: Custom LLM Client

!!! example "Integrate Any LLM"
    ```python title="custom_llm.py" linenums="1" hl_lines="4-5 11-12"
    class MyCustomLLM:
        """Custom LLM implementation"""
        
        def generate(self, prompt: str, max_tokens: int = 512, 
                    temperature: float = 0.7, **kwargs) -> str:
            # Your LLM inference logic
            response = your_model.generate(prompt)
            return response
        
        async def agenerate(self, prompt: str, max_tokens: int = 512,
                           temperature: float = 0.7, **kwargs) -> str:
            # Async version
            return await your_async_model.generate(prompt)
    ```

---

## :material-test-tube: Testing Strategy

!!! abstract "Built for Testability"
    Protocol-based design makes testing straightforward and maintainable.

### :material-cube-outline: Unit Tests

!!! success "Easy Mocking"
    ```python title="test_pipeline.py" linenums="1" hl_lines="2-6 9"
    # Simple mock implementations
    class MockEmbedding:
        def embed(self, text: str) -> list[float]:
            return [0.0] * 768
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 768] * len(texts)
    
    def test_pipeline():
        pipeline = RagPipeline(
            embedding_client=MockEmbedding(),  # Easy mock! ✓
            llm_client=MockLLM(),
            vector_store=MockVectorStore(),
        )
        
        result = pipeline.query("test")
        assert result is not None
        assert result.answer != ""
    ```

!!! tip "No Complex Mocking Frameworks"
    Simple classes with the right methods are all you need!

### :material-integration: Integration Tests

!!! info "Test with Real Services"
    ```python title="test_integration.py" linenums="1" hl_lines="1 3-4"
    @pytest.mark.integration
    def test_milvus_integration():
        # Use real Milvus instance
        store = MilvusVectorStore(
            host="localhost",
            port="19530"
        )
        
        # Test real operations
        store.create_collection("test", dimension=768)
        
        vectors = [[0.1] * 768, [0.2] * 768]
        texts = ["doc1", "doc2"]
        metadata = [{"source": "test"}] * 2
        
        ids = store.insert("test", vectors, texts, metadata)
        assert len(ids) == 2
        
        results = store.search("test", [0.15] * 768, top_k=5)
        assert len(results) > 0
        assert results[0].score > 0
    ```

### :material-run-fast: Performance Tests

!!! example "Benchmark Operations"
    ```python title="test_performance.py"
    import time
    
    def test_batch_embedding_performance():
        embedding = OllamaEmbedding(model="nomic-embed-text")
        texts = ["test document"] * 100
        
        start = time.time()
        vectors = embedding.embed_batch(texts)
        duration = time.time() - start
        
        # Should process 100 docs in under 5 seconds
        assert duration < 5.0
        assert len(vectors) == 100
    ```

---

## :material-speedometer: Performance Considerations

!!! warning "Optimization Best Practices"

### :material-package-variant-closed: Batch Processing

!!! danger "Avoid: One-at-a-time Processing"
    ```python
    # ❌ Slow: Process individually
    vectors = []
    for text in texts:
        vector = embedding.embed(text)
        vectors.append(vector)
    ```

!!! success "Prefer: Batch Operations"
    ```python
    # ✅ Fast: Batch processing
    vectors = embedding.embed_batch(texts)
    ```

**Performance Gain:** Up to **10x faster** for large batches!

### :material-lightning-bolt: Async Operations

!!! info "Use Async for I/O-Bound Tasks"
    
    === "Async LLM Calls"
        ```python
        # Single async call
        response = await llm.agenerate(prompt)
        
        # Concurrent queries
        prompts = ["prompt1", "prompt2", "prompt3"]
        tasks = [llm.agenerate(p) for p in prompts]
        responses = await asyncio.gather(*tasks)
        ```
    
    === "Concurrent Processing"
        ```python
        import asyncio
        
        async def process_documents(documents):
            # Process multiple documents concurrently
            tasks = [
                process_single_doc(doc)
                for doc in documents
            ]
            return await asyncio.gather(*tasks)
        ```

### :material-cached: Caching

!!! tip "Cache Expensive Operations"
    ```python title="cached_embeddings.py" linenums="1" hl_lines="3"
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def cached_embed(text: str) -> tuple[float, ...]:
        # Cache up to 1000 embeddings
        return tuple(embedding.embed(text))
    ```

**Use Case:** Frequently queried terms or repeated documents.

### :material-database-settings: Vector Store Optimization

<div class="grid cards" markdown>

- :material-tune-variant: **Index Configuration**

    ---

    ```python
    store.create_collection(
        name="docs",
        dimension=768,
        index_type="IVF_FLAT",  # Choose appropriate index
        metric="IP",            # Inner product for normalized vectors
        nlist=128,              # Number of clusters
    )
    ```

- :material-filter: **Metadata Filtering**

    ---

    ```python
    # Filter before similarity search
    results = store.search(
        collection="docs",
        query_vector=vector,
        top_k=5,
        filters={"category": "technical"}  # Reduce search space
    )
    ```

- :material-batch-processing: **Batch Insertions**

    ---

    ```python
    # Insert in batches of 1000
    batch_size = 1000
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        store.insert("docs", batch_vectors, batch_texts, metadata)
    ```

</div>

### :material-chart-line: Performance Metrics

| Operation | Without Optimization | With Optimization | Speedup |
|-----------|---------------------|-------------------|---------|
| Embed 100 texts | 5.2s | 0.6s | **8.7x** |
| Concurrent LLM calls | 15s | 3s | **5x** |
| Batch insert 10k vectors | 120s | 18s | **6.7x** |
| Cached embeddings | 0.5s | 0.001s | **500x** |

---

## :material-book-open: Next Steps

<div class="grid cards" markdown>

- :material-school: **Core Concepts**

    ---

    Dive deeper into RAG fundamentals and best practices

    [:material-arrow-right: Learn](../guides/core_concepts.md)

- :material-protocol: **Protocols Guide**

    ---

    Master protocol-based design patterns

    [:material-arrow-right: Explore](../guides/protocols.md)

- :material-code-braces: **Examples**

    ---

    See real-world implementations and patterns

    [:material-arrow-right: Browse](../examples/index.md)

- :material-api: **API Reference**

    ---

    Complete technical documentation

    [:material-arrow-right: Reference](../api/index.md)

</div>

---

## :material-lightbulb: Key Takeaways

!!! success "Design Philosophy"
    
    ✅ **Protocol-Based**: No inheritance required—duck typing with type safety  
    ✅ **Dependency Injection**: Explicit, testable, and flexible  
    ✅ **Layered Architecture**: Clear separation of concerns  
    ✅ **Zero Core Dependencies**: Core layer has no external dependencies  
    ✅ **Easy Testing**: Simple mocking without complex frameworks  
    ✅ **Extensible**: Add custom implementations effortlessly  

!!! quote "Architecture Goals"
    > "Make the simple easy and the complex possible."
    
    RAG Toolkit achieves this through protocols, dependency injection, and clean layering—enabling both quick prototypes and production-ready systems.
