# Embeddings

Embeddings are the cornerstone of semantic search and RAG systems. This guide covers everything you need to know about working with embeddings in rag-toolkit.

## What are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search:

```python
embed("dog") ≈ embed("puppy")      # High similarity
embed("dog") ≠ embed("computer")    # Low similarity
```

**Key Properties:**
- **Dense vectors**: Typically 384-4096 dimensions
- **Semantic similarity**: Similar meaning → similar vectors
- **Language understanding**: Capture context, synonyms, relationships
- **Efficient search**: Fast vector similarity operations

## Supported Embedding Providers

### OpenAI (Recommended)

OpenAI provides state-of-the-art embedding models with excellent quality and speed.

**Models:**
- `text-embedding-3-small`: 1536 dimensions, fast, cost-effective
- `text-embedding-3-large`: 3072 dimensions, highest quality
- `text-embedding-ada-002`: 1536 dimensions, previous generation (still good)

**Installation:**

```bash
pip install rag-toolkit[openai]
export OPENAI_API_KEY="your-api-key"
```

**Usage:**

```python
from rag_toolkit.infra.embedding import OpenAIEmbedding

# Initialize
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    api_key="your-api-key",  # Or set OPENAI_API_KEY env var
)

# Embed single text
vector = await embedding.embed("Hello world")
print(f"Dimension: {len(vector)}")  # 1536

# Embed multiple texts (batched)
vectors = await embedding.embed_batch([
    "First document",
    "Second document",
    "Third document"
])
print(f"Embedded {len(vectors)} documents")
```

**Pricing** (as of Dec 2024):
- `text-embedding-3-small`: $0.02 / 1M tokens
- `text-embedding-3-large`: $0.13 / 1M tokens
- `text-embedding-ada-002`: $0.10 / 1M tokens

### Ollama (Local, Free)

Run embedding models locally with Ollama for privacy and zero API costs.

**Popular Models:**
- `nomic-embed-text`: 768 dimensions, excellent quality
- `mxbai-embed-large`: 1024 dimensions, strong performance
- `all-minilm`: 384 dimensions, fast and lightweight

**Installation:**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Install rag-toolkit with Ollama support
pip install rag-toolkit[ollama]
```

**Usage:**

```python
from rag_toolkit.infra.embedding import OllamaEmbedding

# Initialize
embedding = OllamaEmbedding(
    model="nomic-embed-text",
    base_url="http://localhost:11434",  # Default Ollama URL
)

# Embed text
vector = await embedding.embed("Hello world")
print(f"Dimension: {len(vector)}")  # 768

# Batch embedding
vectors = await embedding.embed_batch([
    "First document",
    "Second document",
    "Third document"
])
```

**Model Comparison:**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `nomic-embed-text` | 768 | Fast | Excellent | General purpose |
| `mxbai-embed-large` | 1024 | Medium | Very good | High quality needs |
| `all-minilm` | 384 | Very fast | Good | Speed critical |

## Configuration

### Batch Size

Control how many texts are embedded at once:

```python
# OpenAI (handles batching automatically)
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",
    batch_size=100,  # Default: 100
)

# Ollama
embedding = OllamaEmbedding(
    model="nomic-embed-text",
    batch_size=32,  # Smaller batches for local GPU
)
```

### Timeouts

Set timeouts for embedding requests:

```python
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",
    timeout=30.0,  # 30 seconds (default: 60)
)
```

### Retry Logic

Handle transient failures with retries:

```python
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",
    max_retries=3,  # Default: 3
    retry_delay=1.0,  # Seconds between retries
)
```

## Advanced Usage

### Dimension Reduction

Reduce embedding dimensions for memory efficiency (OpenAI only):

```python
# text-embedding-3-* models support dimension reduction
embedding = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1024,  # Reduce from 3072 to 1024
)

# Maintains ~98% of quality at ~33% of dimensions
```

### Custom Prefixes

Add prefixes for retrieval vs document embeddings:

```python
# For asymmetric search (query ≠ documents)
embedding = OllamaEmbedding(
    model="nomic-embed-text",
    query_prefix="search_query: ",
    document_prefix="search_document: ",
)

# Query embedding (with prefix)
query_vector = await embedding.embed(
    "What is machine learning?",
    is_query=True
)

# Document embedding (with prefix)
doc_vector = await embedding.embed(
    "Machine learning is a subset of AI...",
    is_query=False
)
```

### Normalize Embeddings

Normalize vectors for cosine similarity:

```python
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",
    normalize=True,  # L2 normalization (default: True)
)

# With normalization: cosine similarity = dot product
# Without: need to compute cosine explicitly
```

## Batch Processing

### Basic Batch Embedding

```python
# Embed multiple documents efficiently
documents = [
    "First document text",
    "Second document text",
    # ... thousands more
]

# Automatically batched
embeddings = await embedding.embed_batch(documents)
print(f"Embedded {len(embeddings)} documents")
```

### Progress Tracking

```python
from tqdm import tqdm

# With progress bar
batch_size = 100
all_embeddings = []

for i in tqdm(range(0, len(documents), batch_size)):
    batch = documents[i:i+batch_size]
    batch_embeddings = await embedding.embed_batch(batch)
    all_embeddings.extend(batch_embeddings)
```

### Error Handling

```python
from rag_toolkit.core.embedding import EmbeddingError

try:
    embeddings = await embedding.embed_batch(documents)
except EmbeddingError as e:
    print(f"Embedding failed: {e}")
    # Handle error (retry, skip, etc.)
```

## Model Selection Guide

### By Quality

**Best Quality (OpenAI):**
```python
embedding = OpenAIEmbedding(model="text-embedding-3-large")
# 3072 dimensions, highest quality
# Use for: Production systems, critical applications
```

**Balanced Quality (OpenAI):**
```python
embedding = OpenAIEmbedding(model="text-embedding-3-small")
# 1536 dimensions, excellent quality/cost ratio
# Use for: Most applications (recommended default)
```

**Good Quality (Ollama, Free):**
```python
embedding = OllamaEmbedding(model="nomic-embed-text")
# 768 dimensions, no API costs
# Use for: Privacy-sensitive, development, offline
```

### By Speed

**Fastest (Ollama, Local):**
```python
embedding = OllamaEmbedding(model="all-minilm")
# 384 dimensions, very fast
# Use for: Real-time applications, large datasets
```

**Fast (OpenAI):**
```python
embedding = OpenAIEmbedding(model="text-embedding-3-small")
# 1536 dimensions, fast API
# Use for: Most applications
```

### By Cost

**Free (Ollama):**
```python
embedding = OllamaEmbedding(model="nomic-embed-text")
# Zero API costs, requires local compute
# Cost: GPU/CPU time only
```

**Cost-Effective (OpenAI):**
```python
embedding = OpenAIEmbedding(
    model="text-embedding-3-small",
    dimensions=512,  # Reduce dimensions = lower cost
)
# $0.02 / 1M tokens
# Use for: Budget-conscious applications
```

## Integration with RAG

### Basic RAG Pipeline

```python
from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding import OpenAIEmbedding
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore
from rag_toolkit.infra.llm import OpenAILLM

# Setup embedding
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# Create RAG pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=MilvusVectorStore(
        collection_name="documents",
        embedding_client=embedding,
        dimension=1536,  # Match embedding dimension
    ),
    llm_client=OpenAILLM(model="gpt-4"),
)

# Index documents
await pipeline.index(
    texts=["Document 1", "Document 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# Query
result = await pipeline.query("What is in the documents?")
print(result.answer)
```

### Hybrid Embedding Strategy

Use different embeddings for different purposes:

```python
# Fast embedding for initial retrieval
fast_embedding = OllamaEmbedding(model="all-minilm")

# High-quality embedding for reranking
quality_embedding = OpenAIEmbedding(model="text-embedding-3-large")

# Two-stage retrieval
# Stage 1: Fast search with all-minilm (1000 candidates)
fast_vector_store = MilvusVectorStore(
    collection_name="fast_search",
    embedding_client=fast_embedding,
)

# Stage 2: Rerank with text-embedding-3-large (top 10)
quality_vector_store = MilvusVectorStore(
    collection_name="quality_rerank",
    embedding_client=quality_embedding,
)
```

## Custom Embedding Clients

Implement your own embedding provider following the protocol:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...
    
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (batched)."""
        ...
```

### Example: HuggingFace Embeddings

```python
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedding:
    """HuggingFace sentence-transformers embedding client."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        return embeddings.tolist()

# Usage
embedding = HuggingFaceEmbedding("all-MiniLM-L6-v2")
vector = await embedding.embed("Hello world")
```

See [Custom Vector Store Example](../examples/custom_vectorstore.md#custom-embeddings) for more details.

## Performance Optimization

### Caching Embeddings

Cache embeddings to avoid recomputing:

```python
from functools import lru_cache

class CachedEmbedding:
    """Embedding client with caching."""
    
    def __init__(self, embedding_client):
        self.client = embedding_client
        self._embed_cached = lru_cache(maxsize=10000)(self._embed_single)
    
    async def _embed_single(self, text: str) -> tuple[float, ...]:
        """Cached embedding (must return tuple for hashability)."""
        vector = await self.client.embed(text)
        return tuple(vector)
    
    async def embed(self, text: str) -> list[float]:
        """Embed with caching."""
        return list(await self._embed_cached(text))
    
    @property
    def dimension(self) -> int:
        return self.client.dimension

# Usage
base_embedding = OpenAIEmbedding(model="text-embedding-3-small")
cached_embedding = CachedEmbedding(base_embedding)

# First call: computes embedding
v1 = await cached_embedding.embed("Hello")  # API call

# Second call: uses cache
v2 = await cached_embedding.embed("Hello")  # No API call
```

### Parallel Processing

Process documents in parallel:

```python
import asyncio

async def embed_documents_parallel(
    documents: list[str],
    embedding_client,
    max_concurrent: int = 10
):
    """Embed documents with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_with_limit(text: str):
        async with semaphore:
            return await embedding_client.embed(text)
    
    tasks = [embed_with_limit(doc) for doc in documents]
    return await asyncio.gather(*tasks)

# Usage
embeddings = await embed_documents_parallel(
    documents=large_document_list,
    embedding_client=embedding,
    max_concurrent=20  # 20 concurrent requests
)
```

### Batch Size Tuning

Find optimal batch size for your use case:

```python
import time

async def benchmark_batch_size(
    documents: list[str],
    embedding_client,
    batch_sizes: list[int] = [10, 50, 100, 200]
):
    """Benchmark different batch sizes."""
    for batch_size in batch_sizes:
        embedding_client.batch_size = batch_size
        
        start = time.time()
        await embedding_client.embed_batch(documents)
        duration = time.time() - start
        
        docs_per_sec = len(documents) / duration
        print(f"Batch size {batch_size}: {docs_per_sec:.1f} docs/sec")

# Find best batch size
await benchmark_batch_size(test_documents, embedding)
```

## Monitoring and Debugging

### Token Usage Tracking

Track embedding costs:

```python
class TokenTrackedEmbedding:
    """Wrapper to track token usage."""
    
    def __init__(self, embedding_client):
        self.client = embedding_client
        self.total_tokens = 0
    
    async def embed(self, text: str) -> list[float]:
        # Rough estimate: 1 token ≈ 4 characters
        tokens = len(text) // 4
        self.total_tokens += tokens
        return await self.client.embed(text)
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        tokens = sum(len(text) // 4 for text in texts)
        self.total_tokens += tokens
        return await self.client.embed_batch(texts)
    
    @property
    def dimension(self) -> int:
        return self.client.dimension
    
    def get_cost(self, model: str = "text-embedding-3-small") -> float:
        """Estimate cost in USD."""
        # Pricing per 1M tokens
        prices = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }
        price_per_million = prices.get(model, 0.02)
        return (self.total_tokens / 1_000_000) * price_per_million

# Usage
tracked = TokenTrackedEmbedding(OpenAIEmbedding())
await tracked.embed_batch(documents)
print(f"Tokens used: {tracked.total_tokens:,}")
print(f"Estimated cost: ${tracked.get_cost():.4f}")
```

### Embedding Quality Check

Verify embedding similarity:

```python
import numpy as np

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test similarity
v1 = await embedding.embed("dog")
v2 = await embedding.embed("puppy")
v3 = await embedding.embed("computer")

print(f"dog <-> puppy: {cosine_similarity(v1, v2):.3f}")  # ~0.8-0.9
print(f"dog <-> computer: {cosine_similarity(v1, v3):.3f}")  # ~0.2-0.3
```

## Troubleshooting

### API Key Issues

```python
from rag_toolkit.infra.embedding import OpenAIEmbedding

try:
    embedding = OpenAIEmbedding()
    await embedding.embed("test")
except Exception as e:
    if "api_key" in str(e).lower():
        print("❌ Invalid or missing API key")
        print("Set OPENAI_API_KEY environment variable")
```

### Ollama Not Running

```python
from rag_toolkit.infra.embedding import OllamaEmbedding

try:
    embedding = OllamaEmbedding()
    await embedding.embed("test")
except Exception as e:
    if "connection" in str(e).lower():
        print("❌ Ollama not running")
        print("Start Ollama: ollama serve")
```

### Model Not Found

```python
try:
    embedding = OllamaEmbedding(model="nomic-embed-text")
    await embedding.embed("test")
except Exception as e:
    if "not found" in str(e).lower():
        print("❌ Model not found")
        print("Pull model: ollama pull nomic-embed-text")
```

### Dimension Mismatch

```python
# Ensure embedding and vector store dimensions match
embedding = OpenAIEmbedding(model="text-embedding-3-small")
print(f"Embedding dimension: {embedding.dimension}")  # 1536

vector_store = MilvusVectorStore(
    collection_name="docs",
    dimension=embedding.dimension,  # Must match!
)
```

## Best Practices

1. **Choose the Right Model**
   - Production: `text-embedding-3-small` (balanced quality/cost)
   - High quality: `text-embedding-3-large`
   - Privacy/offline: `nomic-embed-text` (Ollama)

2. **Batch Processing**
   - Always use `embed_batch()` for multiple texts
   - Tune batch size based on your infrastructure
   - Use async for parallel requests

3. **Error Handling**
   - Implement retries for transient failures
   - Log failed embeddings for debugging
   - Have fallback embedding strategy

4. **Cost Optimization**
   - Cache frequently embedded texts
   - Use dimension reduction when possible
   - Consider Ollama for development

5. **Quality Assurance**
   - Test embeddings with known similar/dissimilar texts
   - Monitor embedding quality over time
   - Validate dimension consistency

6. **Performance**
   - Use appropriate batch sizes
   - Implement connection pooling
   - Consider caching layer

## Next Steps

- [Vector Stores Guide](vector_stores.md) - Store and search embeddings
- [RAG Pipeline](rag_pipeline.md) - Build complete RAG systems
- [Custom Vector Store Example](../examples/custom_vectorstore.md#custom-embeddings)
- [Production Setup](../examples/production_setup.md) - Deploy to production

## See Also

- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Ollama Documentation](https://ollama.com/docs)
- [Embedding Protocol](protocols.md#embeddingclient)
