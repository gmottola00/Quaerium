---
hide:
  - navigation
  - toc
---

<div align="center" markdown="1">

# :rocket: RAG Toolkit

### Production-ready RAG library with multi-vectorstore support

[Get Started](getting_started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/gmottola00/rag-toolkit){ .md-button }

</div>

---

## :sparkles: Key Features

<div class="grid cards" markdown>

-   :material-protocol: **Protocol-Based Architecture**

    ---

    Clean abstractions using Python Protocols (PEP 544). No inheritance required, just implement the interface.

    ```python
    class MyEmbedding(EmbeddingClient):
        def embed(self, texts: List[str]) -> List[List[float]]:
            return your_implementation()
    ```

-   :material-database: **Multi-VectorStore Support**

    ---

    Unified interface for Milvus, Qdrant, and ChromaDB. Switch stores with zero code changes.

    ```python
    store = get_qdrant_service()  # Or Milvus, ChromaDB
    store.add_vectors(vectors, texts, metadatas)
    ```

-   :material-robot: **Multiple LLM Providers**

    ---

    Built-in support for Ollama and OpenAI with easy extensibility for custom providers.

    ```python
    llm = get_ollama_llm(model="llama3.2")
    response = llm.generate(prompt)
    ```

-   :material-package-variant: **Modular Installation**

    ---

    Optional dependencies let you install only what you need. Keep your environment lean.

    ```bash
    # Minimal install
    pip install rag-toolkit
    
    # With Qdrant support
    pip install rag-toolkit[qdrant]
    ```

-   :material-test-tube: **Production Ready**

    ---

    Type hints, comprehensive tests, and professional code quality. Battle-tested in production.

    === "Type Safety"
        ```python
        def process(chunks: List[ChunkLike]) -> RagResponse:
            # Full type checking support
            ...
        ```
    
    === "Testing"
        ```python
        # 60+ tests with 100% pass rate
        pytest tests/
        ```

-   :material-transit-transfer: **Migration Tools**

    ---

    Seamlessly migrate vector data between stores with validation, retry logic, and progress tracking.

    ```python
    migrator = VectorStoreMigrator(source, target)
    result = migrator.migrate(
        source_collection="docs",
        filter={"status": "published"},
        dry_run=True  # Test first!
    )
    ```

</div>

---

## :zap: Quick Example

```python title="basic_rag.py" linenums="1"
from rag_toolkit import RagPipeline
from rag_toolkit.infra import get_ollama_embedding, get_ollama_llm
from rag_toolkit.infra.vectorstores import get_qdrant_service

# Initialize components
embedding = get_ollama_embedding(model="nomic-embed-text")
llm = get_ollama_llm(model="llama3.2")
vector_store = get_qdrant_service(
    host="localhost",
    port=6333,
    collection_name="my_docs"
)

# Create RAG pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
)

# Index your documents
pipeline.index_documents([
    "RAG combines retrieval and generation for better answers.",
    "Vector stores enable semantic search over embeddings.",
    "Chunking strategies impact retrieval quality.",
])

# Query with context
response = pipeline.query("What is RAG?")
print(response.answer)
# Output: "RAG (Retrieval-Augmented Generation) is a technique that combines..."

print(f"Sources: {len(response.sources)}")
# Output: Sources: 2
```

---

## :chart_with_upwards_trend: Performance

Real-world benchmarks across vector stores:

<div class="grid" markdown>

=== "Insert Performance"

    | Store | Single Insert | Batch (100) | Batch (1K) |
    |-------|--------------|-------------|------------|
    | **Qdrant** | 1.2ms | 25.5ms | 260ms |
    | **ChromaDB** | 1.6ms | 6.5ms | 46.5ms |
    | **Milvus** | 11086ms | 11082ms | 22110ms |

    !!! success "Winner: Qdrant for single inserts, ChromaDB for batches"

=== "Search Performance"

    | Store | Top-1 | Top-10 | Top-100 |
    |-------|-------|--------|---------|
    | **ChromaDB** | 0.64ms | 0.90ms | 2.99ms |
    | **Qdrant** | 1.41ms | 2.75ms | 8.78ms |
    | **Milvus** | 2.27ms | 2.30ms | 2.16ms |

    !!! success "Winner: ChromaDB for low latency searches"

=== "Scale (10K vectors)"

    | Store | Insert Time | Search Latency |
    |-------|------------|----------------|
    | **ChromaDB** | 601ms | 1.02ms |
    | **Qdrant** | 3178ms | 3.29ms |
    | **Milvus** | 222734ms | 2.40ms |

    !!! success "Winner: ChromaDB for rapid prototyping"

</div>

[View Full Benchmarks :material-chart-bar:](tools/benchmarks.md){ .md-button }

---

## :building_construction: Architecture

```mermaid
graph LR
    A[Documents] --> B[Parser]
    B --> C[Chunker]
    C --> D[Embeddings]
    D --> E[Vector Store]
    
    F[Query] --> G[Rewriter]
    G --> H[Retriever]
    E --> H
    H --> I[Reranker]
    I --> J[LLM]
    J --> K[Response]
    
    style A fill:#e1f5ff
    style E fill:#fff3e0
    style K fill:#e8f5e9
```

**Protocol-based design** means any component can be swapped:

- **Embeddings**: Ollama, OpenAI, HuggingFace, Custom
- **Vector Stores**: Milvus, Qdrant, ChromaDB, Pinecone, Weaviate
- **LLMs**: Ollama, OpenAI, Anthropic, Custom
- **Parsers**: PDF, DOCX, TXT, HTML, Custom

---

## :rocket: Why RAG Toolkit?

<div class="grid" markdown>

!!! tip "Clean Architecture"
    Protocol-based design means no inheritance requirements. Any class matching the protocol signature works seamlessly.

!!! success "Production Ready"
    Built with best practices: type hints, comprehensive docstrings, modular design, and extensive testing (60+ tests).

!!! gear "Extensible"
    Add new vector stores, LLM providers, or embedding models without touching core code. Plugin-friendly architecture.

!!! heart "Developer Friendly"
    Clear documentation, working examples, and intuitive APIs make development fast and enjoyable.

</div>

---

## :books: Learn More

<div class="grid cards" markdown>

-   :material-book-open-page-variant: **[Getting Started](getting_started/installation.md)**

    Install RAG Toolkit and build your first application in 5 minutes.

-   :material-school: **[User Guide](guides/index.md)**

    Learn core concepts, protocols, and best practices for production RAG systems.

-   :material-code-braces: **[API Reference](api/index.md)**

    Complete API documentation with all classes, functions, and protocols.

-   :material-lightbulb-on: **[Examples](examples/index.md)**

    Real-world examples from basic RAG to production deployments.

</div>

---

## :handshake: Contributing

We welcome contributions! Check out our [Contributing Guide](development/contributing.md) to get started.

<div align="center" markdown="1">

[![GitHub Stars](https://img.shields.io/github/stars/gmottola00/rag-toolkit?style=social)](https://github.com/gmottola00/rag-toolkit)
[![PyPI Downloads](https://img.shields.io/pypi/dm/rag-toolkit)](https://pypi.org/project/rag-toolkit/)
[![License](https://img.shields.io/github/license/gmottola00/rag-toolkit)](https://github.com/gmottola00/rag-toolkit/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/rag-toolkit)](https://pypi.org/project/rag-toolkit/)

</div>
