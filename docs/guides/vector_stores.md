# :material-database: Vector Stores

Vector stores are the **foundation** of RAG systems, providing efficient storage and retrieval of high-dimensional embeddings.

---

## :material-information: Overview

!!! abstract "What is a Vector Store?"
    A vector store (or vector database) stores embeddings along with metadata and provides **lightning-fast similarity search** capabilities.

```mermaid
graph LR
    A[📄 Documents] --> B[🔢 Embeddings]
    B --> C[💾 Vector Store]
    D[💬 Query] --> E[🔢 Query Embedding]
    E --> C
    C --> F[📚 Similar Documents]
    
    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

**How it works:**

1. Documents are converted to embeddings and stored
2. Query is converted to an embedding
3. Vector store finds the most similar stored embeddings
4. Returns the corresponding documents

---

## :material-server: Supported Vector Stores

### :material-database-settings: Milvus (Primary Implementation)

!!! success "Production-Ready Performance"
    Milvus is the default vector store in RAG Toolkit, offering enterprise-grade features.

**Features:**

<div class="grid cards" markdown>

- :material-scale-balance: **Distributed Architecture**

    ---

    Horizontal scalability for massive datasets

- :material-speedometer: **Multiple Index Types**

    ---

    HNSW, IVF_FLAT, IVF_PQ for different use cases

- :material-gpu: **GPU Acceleration**

    ---

    Optional GPU support for faster search

- :material-filter: **Hybrid Search**

    ---

    Vector + metadata filtering combined

</div>

**Installation:**

```bash
# Install with Milvus support (included by default)
pip install rag-toolkit

# Start Milvus (Docker)
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest standalone
```

**Basic Usage:**

```python
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore
from rag_toolkit.infra.embedding import OpenAIEmbedding

# Initialize vector store
vector_store = MilvusVectorStore(
    collection_name="my_documents",
    uri="http://localhost:19530",
    embedding_client=OpenAIEmbedding(),
    dimension=1536,  # OpenAI ada-002 dimension
)

# Insert documents
vector_store.upsert(
    texts=["Document 1 content", "Document 2 content"],
    metadatas=[
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc2.pdf", "page": 1}
    ]
)

# Search
results = vector_store.search(
    query="What is machine learning?",
    limit=5
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")
```

## Configuration

### Connection Settings

```python
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

# Local Milvus
vector_store = MilvusVectorStore(
    collection_name="documents",
    uri="http://localhost:19530",
    token=None,  # No authentication for local
)

# Milvus Cloud (Zilliz)
vector_store = MilvusVectorStore(
    collection_name="documents",
    uri="https://your-instance.zillizcloud.com:19530",
    token="your_api_key",
)

# Milvus with authentication
vector_store = MilvusVectorStore(
    collection_name="documents",
    uri="http://production-milvus:19530",
    token="user:password",
)
```

### Index Configuration

Different index types offer trade-offs between speed, accuracy, and memory:

```python
from rag_toolkit.infra.vectorstores.milvus.config import IndexConfig

# HNSW - Best for high accuracy (default)
index_config = IndexConfig(
    index_type="HNSW",
    metric_type="COSINE",  # or "L2", "IP"
    params={"M": 16, "efConstruction": 200}
)

# IVF_FLAT - Balanced speed/accuracy
index_config = IndexConfig(
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 1024}
)

# IVF_PQ - Memory efficient for large datasets
index_config = IndexConfig(
    index_type="IVF_PQ",
    metric_type="COSINE",
    params={"nlist": 1024, "m": 8, "nbits": 8}
)

vector_store = MilvusVectorStore(
    collection_name="documents",
    index_config=index_config,
)
```

### Metric Types

Choose the right distance metric for your embeddings:

| Metric | Use Case | Range | Best For |
|--------|----------|-------|----------|
| `COSINE` | Default, most common | [-1, 1] | Text embeddings (OpenAI, Ollama) |
| `L2` | Euclidean distance | [0, ∞) | Image embeddings |
| `IP` | Inner product | [-∞, ∞) | Pre-normalized vectors |

```python
# Cosine similarity (recommended for most use cases)
vector_store = MilvusVectorStore(
    collection_name="documents",
    metric_type="COSINE",
)
```

## Collection Management

### Creating Collections

```python
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

# Auto-create collection on first insert
vector_store = MilvusVectorStore(
    collection_name="my_docs",
    dimension=1536,
    auto_create=True,  # Default
)

# Explicitly create collection
await vector_store.create_collection(
    dimension=1536,
    description="My document collection",
)

# Check if collection exists
exists = await vector_store.collection_exists()
print(f"Collection exists: {exists}")
```

### Listing Collections

```python
# List all collections
collections = await vector_store.list_collections()
for collection in collections:
    print(f"Name: {collection.name}")
    print(f"Count: {collection.count}")
    print(f"Dimension: {collection.dimension}")
```

### Dropping Collections

```python
# Delete collection and all data
await vector_store.drop_collection()
```

## Data Operations

### Inserting Documents

```python
# Simple insert
ids = vector_store.upsert(
    texts=["First document", "Second document"],
    metadatas=[{"tag": "intro"}, {"tag": "advanced"}]
)

# Insert with explicit IDs
ids = vector_store.upsert(
    ids=["doc-1", "doc-2"],
    texts=["Content 1", "Content 2"],
    metadatas=[{"source": "pdf"}, {"source": "web"}]
)

# Insert with pre-computed embeddings
ids = vector_store.upsert(
    ids=["doc-1"],
    embeddings=[[0.1, 0.2, ..., 0.5]],  # 1536-dim vector
    texts=["Document text"],
    metadatas=[{"source": "custom"}]
)
```

### Updating Documents

```python
# Update existing document (same ID)
vector_store.upsert(
    ids=["doc-1"],
    texts=["Updated content"],
    metadatas=[{"source": "pdf", "updated": True}]
)
```

### Deleting Documents

```python
# Delete by ID
vector_store.delete(ids=["doc-1", "doc-2"])

# Delete by filter
vector_store.delete(filter={"source": "old_data"})

# Delete all
vector_store.delete(filter={})  # ⚠️ Careful!
```

### Retrieving Documents

```python
# Get by ID
documents = vector_store.get(ids=["doc-1", "doc-2"])

# Get all documents
all_docs = vector_store.get()

# Get with metadata filter
filtered = vector_store.get(
    filter={"source": "research_papers"}
)
```

## Search

### Basic Vector Search

```python
# Semantic search
results = vector_store.search(
    query="What is machine learning?",
    limit=5
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text[:100]}...")
    print(f"Metadata: {result.metadata}\n")
```

### Search with Metadata Filtering

```python
# Filter by metadata
results = vector_store.search(
    query="neural networks",
    limit=10,
    filter={
        "source": "textbook.pdf",
        "chapter": 3
    }
)

# Complex filters
results = vector_store.search(
    query="deep learning",
    limit=5,
    filter={
        "year": {"$gte": 2020},  # >= 2020
        "category": {"$in": ["AI", "ML"]},  # In list
        "verified": True
    }
)
```

### Search with Custom Parameters

```python
# Adjust search parameters for speed/accuracy trade-off
results = vector_store.search(
    query="transformer models",
    limit=10,
    search_params={
        "ef": 64,  # HNSW: higher = more accurate but slower
    }
)
```

### Batch Search

```python
# Search multiple queries at once
queries = [
    "What is supervised learning?",
    "Explain neural networks",
    "Describe gradient descent"
]

all_results = vector_store.batch_search(
    queries=queries,
    limit=5
)

for i, results in enumerate(all_results):
    print(f"\nQuery {i+1}: {queries[i]}")
    for result in results:
        print(f"  - {result.text[:50]}... (score: {result.score:.3f})")
```

## Advanced Features

### Hybrid Search (Vector + Keyword)

Combine semantic search with keyword matching:

```python
# Vector search with keyword boost
results = vector_store.hybrid_search(
    query="machine learning algorithms",
    keywords=["neural network", "deep learning"],
    limit=10,
    alpha=0.7,  # 0.7 vector + 0.3 keyword
)
```

### Reranking

Improve result quality with reranking:

```python
from rag_toolkit.rag.rerankers import CrossEncoderReranker

# Initial search with more results
results = vector_store.search(
    query="transformer architecture",
    limit=50  # Retrieve more candidates
)

# Rerank to get best 10
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(
    query="transformer architecture",
    documents=results,
    top_k=10
)
```

### Dynamic Schema

Store arbitrary metadata without predefined schema:

```python
# Each document can have different metadata fields
vector_store.upsert(
    texts=[
        "Document 1",
        "Document 2",
        "Document 3"
    ],
    metadatas=[
        {"source": "pdf", "pages": 10, "author": "Smith"},
        {"source": "web", "url": "example.com"},  # Different fields
        {"source": "book", "isbn": "123", "chapter": 5}  # More fields
    ]
)

# Filter by any field
results = vector_store.search(
    query="important topic",
    filter={"author": "Smith"}
)
```

### Partitions

Organize data into logical partitions:

```python
# Create partitions for different data types
vector_store.create_partition("research_papers")
vector_store.create_partition("blog_posts")

# Insert into specific partition
vector_store.upsert(
    texts=["Research paper content"],
    partition="research_papers"
)

# Search in specific partition
results = vector_store.search(
    query="latest findings",
    partition="research_papers",
    limit=5
)
```

## Performance Optimization

### Batch Operations

Process documents in batches for better performance:

```python
# Batch insert (recommended for large datasets)
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vector_store.upsert(
        texts=[doc.text for doc in batch],
        metadatas=[doc.metadata for doc in batch]
    )
```

### Index Building

Build index after bulk inserts:

```python
# Insert without building index
vector_store.upsert(
    texts=large_document_list,
    build_index=False  # Skip index building
)

# Build index once after all inserts
await vector_store.build_index()
```

### Connection Pooling

Reuse connections for better performance:

```python
# Connection pool automatically managed
vector_store = MilvusVectorStore(
    collection_name="docs",
    pool_size=10,  # Connection pool size
)
```

## Monitoring and Maintenance

### Collection Statistics

```python
# Get collection info
stats = await vector_store.get_collection_stats()
print(f"Document count: {stats.count}")
print(f"Dimension: {stats.dimension}")
print(f"Metric type: {stats.metric}")

# Get collection size
size_mb = await vector_store.get_collection_size()
print(f"Collection size: {size_mb:.2f} MB")
```

### Compaction

Optimize storage by compacting deleted data:

```python
# Compact collection (reclaim space from deleted documents)
await vector_store.compact()
```

### Index Statistics

```python
# Get index information
index_info = await vector_store.get_index_info()
print(f"Index type: {index_info.type}")
print(f"Index params: {index_info.params}")
```

## Implementing Custom Vector Stores

You can implement your own vector store by following the `VectorStoreClient` protocol:

```python
from typing import Protocol, runtime_checkable
from rag_toolkit.core.vectorstore import VectorStoreClient
from rag_toolkit.core.types import SearchResult

@runtime_checkable
class CustomVectorStore(VectorStoreClient):
    """Custom vector store implementation."""
    
    async def upsert(
        self,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Insert or update documents."""
        # Your implementation
        pass
    
    async def search(
        self,
        query: str | list[float],
        limit: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Your implementation
        pass
    
    async def delete(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> None:
        """Delete documents."""
        # Your implementation
        pass
    
    async def get(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents."""
        # Your implementation
        pass
```

See [Custom Vector Store Example](../examples/custom_vectorstore.md) for a complete implementation.

## Troubleshooting

### Connection Issues

```python
# Check connection
try:
    await vector_store.health_check()
    print("✅ Connected to Milvus")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Collection Not Found

```python
# Check if collection exists before operations
if not await vector_store.collection_exists():
    await vector_store.create_collection(dimension=1536)
```

### Dimension Mismatch

```python
# Ensure embedding dimension matches collection dimension
embedding_dim = vector_store.embedding_client.dimension
collection_dim = vector_store.dimension

if embedding_dim != collection_dim:
    raise ValueError(
        f"Dimension mismatch: embeddings={embedding_dim}, "
        f"collection={collection_dim}"
    )
```

### Out of Memory

```python
# For large datasets, use batch processing
batch_size = 100  # Reduce if still OOM

# Or use memory-efficient index
from rag_toolkit.infra.vectorstores.milvus.config import IndexConfig

index_config = IndexConfig(
    index_type="IVF_PQ",  # More memory efficient
    params={"nlist": 1024, "m": 8, "nbits": 8}
)
```

## Best Practices

1. **Choose the Right Metric**
   - Use `COSINE` for text embeddings (default)
   - Use `L2` for image embeddings
   - Use `IP` if embeddings are pre-normalized

2. **Index Selection**
   - `HNSW`: Best accuracy, more memory
   - `IVF_FLAT`: Balanced
   - `IVF_PQ`: Memory efficient for large datasets

3. **Batch Operations**
   - Insert documents in batches (100-1000)
   - Build index after bulk inserts
   - Use async operations for better performance

4. **Metadata Design**
   - Keep metadata small and relevant
   - Use consistent field names
   - Index frequently filtered fields

5. **Collection Management**
   - One collection per logical dataset
   - Use partitions for data organization
   - Regular compaction for space efficiency

6. **Monitoring**
   - Check collection stats regularly
   - Monitor query latency
   - Set up alerts for errors

## Next Steps

- [Embeddings Guide](embeddings.md) - Learn about embedding clients
- [RAG Pipeline](rag_pipeline.md) - Build complete RAG systems
- [Custom Vector Store Example](../examples/custom_vectorstore.md) - Implement your own
- [Production Setup](../examples/production_setup.md) - Deploy to production

## See Also

- [Milvus Documentation](https://milvus.io/docs)
- [Vector Store Protocol](protocols.md#vectorstoreclient)
- [Search Strategies](core_concepts.md#search-strategies)
