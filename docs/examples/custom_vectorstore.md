# Custom Vector Store Implementation

Learn how to implement your own vector store by following the `VectorStoreClient` protocol. This example shows a complete implementation using ChromaDB.

## Why Custom Vector Stores?

While rag-toolkit includes Milvus by default, you might want to:
- Use a different vector database (Pinecone, Qdrant, ChromaDB, Weaviate)
- Integrate with existing infrastructure
- Implement custom search logic
- Add specialized features

## The VectorStoreClient Protocol

All vector stores must implement this protocol:

```python
from typing import Protocol, runtime_checkable
from rag_toolkit.core.types import SearchResult

@runtime_checkable
class VectorStoreClient(Protocol):
    """Protocol for vector store implementations."""
    
    async def upsert(
        self,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Insert or update documents."""
        ...
    
    async def search(
        self,
        query: str | list[float],
        limit: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...
    
    async def delete(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> None:
        """Delete documents."""
        ...
    
    async def get(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents."""
        ...
```

## Complete ChromaDB Implementation

Here's a full-featured implementation using ChromaDB:

```python
import uuid
from typing import Optional
import chromadb
from chromadb.config import Settings
from rag_toolkit.core.vectorstore import VectorStoreClient
from rag_toolkit.core.embedding import EmbeddingClient
from rag_toolkit.core.types import SearchResult

class ChromaVectorStore:
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_client: EmbeddingClient,
        persist_directory: str = "./chroma_db",
        distance_metric: str = "cosine",  # or "l2", "ip"
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_client: Embedding client for text→vector
            persist_directory: Where to store the database
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        self.distance_metric = distance_metric
        
        # Initialize Chroma client
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )
    
    async def upsert(
        self,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Insert or update documents."""
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts or embeddings))]
        
        # Compute embeddings if not provided
        if embeddings is None and texts is not None:
            embeddings = await self.embedding_client.embed_batch(texts)
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        # ChromaDB requires documents (texts)
        if texts is None:
            texts = ["" for _ in ids]
        
        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    async def search(
        self,
        query: str | list[float],
        limit: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Embed query if string
        if isinstance(query, str):
            query_vector = await self.embedding_client.embed(query)
        else:
            query_vector = query
        
        # Convert filter to ChromaDB format
        where = self._convert_filter(filter) if filter else None
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Convert to SearchResult
        search_results = []
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity score
            distance = results['distances'][0][i]
            score = self._distance_to_score(distance)
            
            search_results.append(
                SearchResult(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=score,
                    vector=results['embeddings'][0][i] if results['embeddings'] else None
                )
            )
        
        return search_results
    
    async def delete(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> None:
        """Delete documents."""
        if ids:
            # Delete by IDs
            self.collection.delete(ids=ids)
        elif filter:
            # Delete by filter
            where = self._convert_filter(filter)
            self.collection.delete(where=where)
        else:
            # Delete all (careful!)
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
    
    async def get(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents."""
        where = self._convert_filter(filter) if filter else None
        
        results = self.collection.get(
            ids=ids,
            where=where,
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Convert to SearchResult
        search_results = []
        for i in range(len(results['ids'])):
            search_results.append(
                SearchResult(
                    id=results['ids'][i],
                    text=results['documents'][i],
                    metadata=results['metadatas'][i],
                    score=1.0,  # No score for get operation
                    vector=results['embeddings'][i] if results['embeddings'] else None
                )
            )
        
        return search_results
    
    def _convert_filter(self, filter: dict) -> dict:
        """Convert rag-toolkit filter to ChromaDB where clause."""
        where = {}
        
        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle operators like $gte, $in
                for op, val in value.items():
                    if op == "$gte":
                        where[key] = {"$gte": val}
                    elif op == "$lte":
                        where[key] = {"$lte": val}
                    elif op == "$in":
                        where[key] = {"$in": val}
                    elif op == "$ne":
                        where[key] = {"$ne": val}
            else:
                # Simple equality
                where[key] = {"$eq": value}
        
        return where
    
    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score [0, 1]."""
        if self.distance_metric == "cosine":
            # Cosine distance is 1 - cosine_similarity
            # So score = 1 - distance
            return max(0.0, 1.0 - distance)
        elif self.distance_metric == "l2":
            # L2 distance: smaller is better
            # Convert to similarity score
            return 1.0 / (1.0 + distance)
        elif self.distance_metric == "ip":
            # Inner product: larger is better
            # Already a similarity score
            return distance
        return distance
```

## Using the Custom Vector Store

### Basic Usage

```python
from rag_toolkit.infra.embedding import OpenAIEmbedding

# Create embedding client
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# Create ChromaDB vector store
vector_store = ChromaVectorStore(
    collection_name="my_documents",
    embedding_client=embedding,
    persist_directory="./my_chroma_db",
)

# Insert documents
ids = await vector_store.upsert(
    texts=["Document 1", "Document 2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# Search
results = await vector_store.search(
    query="relevant query",
    limit=5
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text}")
```

### Integration with RAG Pipeline

```python
from rag_toolkit import RagPipeline
from rag_toolkit.infra.llm import OpenAILLM

# Create RAG pipeline with custom vector store
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,  # Your custom implementation!
    llm_client=OpenAILLM(model="gpt-4-turbo"),
)

# Use as normal
await pipeline.index(texts=["Doc 1", "Doc 2"])
result = await pipeline.query("What is in the documents?")
print(result.answer)
```

## Advanced Features

### Custom Embeddings

You can also implement custom embedding clients:

```python
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedding:
    """HuggingFace sentence-transformers embedding."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        return embeddings.tolist()

# Use together
embedding = HuggingFaceEmbedding("all-MiniLM-L6-v2")
vector_store = ChromaVectorStore(
    collection_name="docs",
    embedding_client=embedding
)
```

### Hybrid Search

Add keyword search to your vector store:

```python
class ChromaVectorStoreWithHybrid(ChromaVectorStore):
    """ChromaDB with hybrid search support."""
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.7,  # 0.7 vector + 0.3 keyword
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Hybrid vector + keyword search."""
        # Vector search
        vector_results = await self.search(
            query=query,
            limit=limit * 2,  # Get more candidates
            filter=filter
        )
        
        # Keyword search (simple text matching)
        keyword_results = await self._keyword_search(
            query=query,
            limit=limit * 2,
            filter=filter
        )
        
        # Combine scores
        combined = self._combine_results(
            vector_results,
            keyword_results,
            alpha=alpha
        )
        
        # Return top K
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:limit]
    
    async def _keyword_search(
        self,
        query: str,
        limit: int,
        filter: dict | None = None
    ) -> list[SearchResult]:
        """Simple keyword-based search."""
        where = self._convert_filter(filter) if filter else None
        
        # Get all documents (or filtered subset)
        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )
        
        # Score by keyword presence
        query_lower = query.lower()
        scored_results = []
        
        for i in range(len(results['ids'])):
            doc_lower = results['documents'][i].lower()
            
            # Simple TF score
            score = sum(
                doc_lower.count(word.lower())
                for word in query_lower.split()
            )
            
            if score > 0:
                scored_results.append(
                    SearchResult(
                        id=results['ids'][i],
                        text=results['documents'][i],
                        metadata=results['metadatas'][i],
                        score=score,
                        vector=None
                    )
                )
        
        # Normalize scores
        if scored_results:
            max_score = max(r.score for r in scored_results)
            for result in scored_results:
                result.score = result.score / max_score
        
        # Sort and return top K
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:limit]
    
    def _combine_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        alpha: float
    ) -> list[SearchResult]:
        """Combine vector and keyword results."""
        # Create score map
        scores = {}
        
        # Add vector scores
        for result in vector_results:
            scores[result.id] = {
                'vector': result.score * alpha,
                'keyword': 0.0,
                'result': result
            }
        
        # Add keyword scores
        for result in keyword_results:
            if result.id in scores:
                scores[result.id]['keyword'] = result.score * (1 - alpha)
            else:
                scores[result.id] = {
                    'vector': 0.0,
                    'keyword': result.score * (1 - alpha),
                    'result': result
                }
        
        # Combine scores
        combined = []
        for doc_id, data in scores.items():
            result = data['result']
            result.score = data['vector'] + data['keyword']
            combined.append(result)
        
        return combined

# Usage
vector_store = ChromaVectorStoreWithHybrid(
    collection_name="docs",
    embedding_client=embedding
)

results = await vector_store.hybrid_search(
    query="machine learning algorithms",
    limit=10,
    alpha=0.7  # Prefer vector search
)
```

## Testing Your Implementation

### Unit Tests

```python
import pytest

@pytest.mark.asyncio
async def test_upsert():
    """Test document insertion."""
    vector_store = ChromaVectorStore(
        collection_name="test",
        embedding_client=embedding
    )
    
    ids = await vector_store.upsert(
        texts=["Test document"],
        metadatas=[{"source": "test"}]
    )
    
    assert len(ids) == 1
    assert ids[0] is not None

@pytest.mark.asyncio
async def test_search():
    """Test search functionality."""
    vector_store = ChromaVectorStore(
        collection_name="test",
        embedding_client=embedding
    )
    
    # Insert documents
    await vector_store.upsert(
        texts=["Machine learning is great", "Python is amazing"],
        metadatas=[{"topic": "ML"}, {"topic": "Programming"}]
    )
    
    # Search
    results = await vector_store.search(
        query="artificial intelligence",
        limit=2
    )
    
    assert len(results) <= 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.score >= 0 for r in results)

@pytest.mark.asyncio
async def test_metadata_filter():
    """Test metadata filtering."""
    vector_store = ChromaVectorStore(
        collection_name="test",
        embedding_client=embedding
    )
    
    # Insert with metadata
    await vector_store.upsert(
        texts=["Doc 1", "Doc 2"],
        metadatas=[{"category": "A"}, {"category": "B"}]
    )
    
    # Search with filter
    results = await vector_store.search(
        query="document",
        filter={"category": "A"}
    )
    
    assert all(r.metadata["category"] == "A" for r in results)
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_rag_pipeline_integration():
    """Test integration with RAG pipeline."""
    from rag_toolkit import RagPipeline
    from rag_toolkit.infra.llm import OpenAILLM
    
    pipeline = RagPipeline(
        embedding_client=embedding,
        vector_store=vector_store,
        llm_client=OpenAILLM()
    )
    
    # Index documents
    await pipeline.index(
        texts=["Machine learning is a subset of AI."],
        metadatas=[{"source": "textbook"}]
    )
    
    # Query
    result = await pipeline.query("What is ML?")
    
    assert result.answer is not None
    assert len(result.sources) > 0
```

## Other Vector Store Examples

### Pinecone Implementation

```python
import pinecone

class PineconeVectorStore:
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        index_name: str,
        embedding_client: EmbeddingClient,
        api_key: str,
        environment: str,
    ):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
        self.embedding_client = embedding_client
    
    async def upsert(
        self,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        if embeddings is None:
            embeddings = await self.embedding_client.embed_batch(texts)
        
        # Prepare vectors
        vectors = [
            (ids[i], embeddings[i], metadatas[i] if metadatas else {})
            for i in range(len(ids))
        ]
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        return ids
    
    # ... implement other methods
```

### Qdrant Implementation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantVectorStore:
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_client: EmbeddingClient,
        url: str = "localhost",
        port: int = 6333,
    ):
        self.client = QdrantClient(url=url, port=port)
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        
        # Create collection if not exists
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_client.dimension,
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass  # Collection already exists
    
    async def upsert(
        self,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        if embeddings is None:
            embeddings = await self.embedding_client.embed_batch(texts)
        
        # Prepare points
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "text": texts[i] if texts else "",
                    **metadatas[i] if metadatas else {}
                }
            )
            for i in range(len(ids))
        ]
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    # ... implement other methods
```

## Best Practices

1. **Follow the Protocol Strictly**
   - Implement all required methods
   - Match signatures exactly
   - Return correct types

2. **Handle Errors Gracefully**
   - Wrap external library exceptions
   - Provide helpful error messages
   - Implement retries for transient failures

3. **Optimize Performance**
   - Batch operations when possible
   - Use connection pooling
   - Cache embeddings

4. **Test Thoroughly**
   - Unit tests for each method
   - Integration tests with RAG pipeline
   - Load tests for performance

5. **Document Your Implementation**
   - Clear docstrings
   - Usage examples
   - Configuration options

## Next Steps

- [Hybrid Search Example](hybrid_search.md) - Advanced search techniques
- [Vector Stores Guide](../user_guide/vector_stores.md) - Deep dive
- [Production Setup](production_setup.md) - Deploy to production

## See Also

- [VectorStoreClient Protocol](../user_guide/protocols.md#vectorstoreclient)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
