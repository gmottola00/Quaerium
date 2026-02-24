# :material-magnify-expand: Hybrid Search Implementation

!!! success "Best of Both Worlds"
    Learn how to combine vector search with keyword search for more accurate and robust retrieval. Hybrid search leverages the strengths of both approaches to improve search quality.

---

## :material-help-circle: Why Hybrid Search?

!!! info "Complementary Strengths"
    Vector search and keyword search each have unique advantages.

=== "🔍 Vector Search"

    **Advantages:**
    
    - ✅ Semantic understanding
    - ✅ Finds conceptually similar content
    - ✅ Handles synonyms and paraphrases
    
    **Limitations:**
    
    - ❌ May miss exact keyword matches
    - ❌ Less effective for proper nouns/codes

=== "📝 Keyword Search"

    **Advantages:**
    
    - ✅ Exact match precision
    - ✅ Great for proper nouns, codes, IDs
    - ✅ Fast and deterministic
    
    **Limitations:**
    
    - ❌ No semantic understanding
    - ❌ Misses synonyms

=== "⚡ Hybrid Search"

    **Benefits:**
    
    - ✅ Best of both worlds
    - ✅ Semantic + exact matching
    - ✅ More robust retrieval
    - ✅ Better overall accuracy

## Basic Hybrid Search

### Simple Implementation

```python
from quaerium.core.vectorstore import MilvusVectorStore
from quaerium.core.embedding import OpenAIEmbedding
from quaerium.core.types import SearchResult

class HybridSearcher:
    """Simple hybrid search combining vector + keyword."""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_client: OpenAIEmbedding,
        alpha: float = 0.7,  # Weight for vector search
    ):
        """Initialize hybrid searcher.
        
        Args:
            vector_store: Vector store for semantic search
            embedding_client: Embedding client
            alpha: Weight for vector search (1-alpha for keyword)
        """
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.alpha = alpha
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query
            limit: Number of results
            filter: Metadata filter
            
        Returns:
            Ranked search results
        """
        # 1. Vector search
        vector_results = await self.vector_store.search(
            query=query,
            limit=limit * 2,  # Get more candidates
            filter=filter
        )
        
        # 2. Keyword search (BM25 or simple text matching)
        keyword_results = await self._keyword_search(
            query=query,
            limit=limit * 2,
            filter=filter
        )
        
        # 3. Combine scores
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60  # RRF constant
        )
        
        # 4. Return top K
        return combined[:limit]
    
    async def _keyword_search(
        self,
        query: str,
        limit: int,
        filter: dict | None = None
    ) -> list[SearchResult]:
        """Simple BM25-style keyword search."""
        # Get all documents (in production, use Elasticsearch or similar)
        all_docs = await self.vector_store.get(filter=filter)
        
        # Score documents
        query_terms = query.lower().split()
        scored_docs = []
        
        for doc in all_docs:
            score = self._bm25_score(
                doc.text.lower(),
                query_terms
            )
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to SearchResult
        results = []
        for doc, score in scored_docs[:limit]:
            results.append(
                SearchResult(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=score,
                    vector=doc.vector
                )
            )
        
        return results
    
    def _bm25_score(
        self,
        text: str,
        query_terms: list[str],
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """Simplified BM25 scoring."""
        score = 0.0
        
        for term in query_terms:
            # Term frequency
            tf = text.count(term)
            
            # Document length normalization
            doc_len = len(text.split())
            avg_doc_len = 100  # Approximate
            
            # BM25 formula (simplified)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            
            score += numerator / denominator
        
        return score
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        k: int = 60
    ) -> list[SearchResult]:
        """Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score(doc) = sum(1 / (k + rank))
        """
        # Create score map
        scores = {}
        
        # Add vector scores (rank-based)
        for rank, result in enumerate(vector_results):
            if result.id not in scores:
                scores[result.id] = {'doc': result, 'score': 0.0}
            scores[result.id]['score'] += 1.0 / (k + rank + 1)
        
        # Add keyword scores (rank-based)
        for rank, result in enumerate(keyword_results):
            if result.id not in scores:
                scores[result.id] = {'doc': result, 'score': 0.0}
            scores[result.id]['score'] += 1.0 / (k + rank + 1)
        
        # Sort by combined score
        ranked = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Convert back to SearchResult
        results = []
        for item in ranked:
            doc = item['doc']
            doc.score = item['score']
            results.append(doc)
        
        return results

# Usage
searcher = HybridSearcher(
    vector_store=vector_store,
    embedding_client=embedding,
    alpha=0.7
)

results = await searcher.search(
    query="machine learning algorithms",
    limit=10
)
```

## Advanced Hybrid Search

### Milvus Native Hybrid Search

Milvus 2.4+ supports native hybrid search with multiple vector fields:

```python
from pymilvus import Collection, connections

class MilvusHybridSearch:
    """Milvus native hybrid search with sparse + dense vectors."""
    
    def __init__(
        self,
        collection_name: str,
        dense_embedding: OpenAIEmbedding,
        sparse_embedding: SparseEmbedding,  # BM25 or SPLADE
    ):
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        dense_weight: float = 0.7,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Milvus native hybrid search."""
        # Generate both embeddings
        dense_vector = await self.dense_embedding.embed(query)
        sparse_vector = await self.sparse_embedding.embed(query)
        
        # Hybrid search
        results = self.collection.hybrid_search(
            data=[
                [dense_vector],  # Dense search
                [sparse_vector]  # Sparse search
            ],
            anns_field=["dense_vector", "sparse_vector"],
            param=[
                {"metric_type": "COSINE", "params": {"ef": 64}},
                {"metric_type": "IP", "params": {}}
            ],
            limit=limit,
            expr=filter_expr,
            output_fields=["text", "metadata"],
            rerank=RRFReranker(k=60, weights=[dense_weight, 1 - dense_weight])
        )
        
        # Convert to SearchResult
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(
                    SearchResult(
                        id=hit.id,
                        text=hit.entity.get("text"),
                        metadata=hit.entity.get("metadata", {}),
                        score=hit.score,
                        vector=None
                    )
                )
        
        return search_results
```

### Elasticsearch + Milvus Hybrid

Use Elasticsearch for keyword search and Milvus for vector search:

```python
from elasticsearch import Elasticsearch

class ElasticsearchMilvusHybrid:
    """Hybrid search with Elasticsearch + Milvus."""
    
    def __init__(
        self,
        es_client: Elasticsearch,
        es_index: str,
        milvus_store: MilvusVectorStore,
        embedding: OpenAIEmbedding,
    ):
        self.es = es_client
        self.es_index = es_index
        self.milvus = milvus_store
        self.embedding = embedding
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        """Hybrid search with Elasticsearch + Milvus."""
        # 1. Elasticsearch BM25 search
        es_results = self.es.search(
            index=self.es_index,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "title"],
                        "type": "best_fields"
                    }
                },
                "size": limit * 2
            }
        )
        
        # 2. Milvus vector search
        milvus_results = await self.milvus.search(
            query=query,
            limit=limit * 2
        )
        
        # 3. Combine scores (weighted average)
        combined = self._weighted_fusion(
            es_results['hits']['hits'],
            milvus_results,
            alpha=alpha
        )
        
        return combined[:limit]
    
    def _weighted_fusion(
        self,
        es_hits: list,
        milvus_hits: list[SearchResult],
        alpha: float
    ) -> list[SearchResult]:
        """Weighted score fusion."""
        # Normalize scores
        es_scores = {}
        if es_hits:
            max_es = max(hit['_score'] for hit in es_hits)
            for hit in es_hits:
                es_scores[hit['_id']] = hit['_score'] / max_es
        
        milvus_scores = {}
        if milvus_hits:
            max_milvus = max(hit.score for hit in milvus_hits)
            for hit in milvus_hits:
                milvus_scores[hit.id] = hit.score / max_milvus
        
        # Combine scores
        combined = {}
        all_ids = set(es_scores.keys()) | set(milvus_scores.keys())
        
        for doc_id in all_ids:
            es_score = es_scores.get(doc_id, 0.0)
            milvus_score = milvus_scores.get(doc_id, 0.0)
            
            combined_score = alpha * milvus_score + (1 - alpha) * es_score
            combined[doc_id] = combined_score
        
        # Sort and return
        ranked = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to SearchResult (simplified)
        results = []
        for doc_id, score in ranked:
            # Find original document
            doc = next(
                (h for h in milvus_hits if h.id == doc_id),
                None
            )
            if doc:
                doc.score = score
                results.append(doc)
        
        return results
```

## Adaptive Hybrid Search

Adjust weights based on query type:

```python
import re

class AdaptiveHybridSearch:
    """Adaptive hybrid search with query-specific weighting."""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding: OpenAIEmbedding,
    ):
        self.vector_store = vector_store
        self.embedding = embedding
    
    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Adaptive hybrid search."""
        # Determine query type
        alpha = self._determine_alpha(query)
        
        # Perform hybrid search with adaptive weight
        searcher = HybridSearcher(
            vector_store=self.vector_store,
            embedding_client=self.embedding,
            alpha=alpha
        )
        
        return await searcher.search(query, limit)
    
    def _determine_alpha(self, query: str) -> float:
        """Determine vector search weight based on query type."""
        # Proper nouns or codes → prefer keyword search
        if self._has_proper_nouns(query) or self._has_codes(query):
            return 0.3  # 30% vector, 70% keyword
        
        # Short queries → prefer keyword search
        if len(query.split()) <= 3:
            return 0.4  # 40% vector, 60% keyword
        
        # Questions or conceptual queries → prefer vector search
        if query.startswith(("what", "how", "why", "explain")):
            return 0.9  # 90% vector, 10% keyword
        
        # Default: balanced
        return 0.7  # 70% vector, 30% keyword
    
    def _has_proper_nouns(self, query: str) -> bool:
        """Check if query contains proper nouns."""
        # Simple heuristic: capitalized words mid-sentence
        words = query.split()
        if len(words) > 1:
            return any(w[0].isupper() for w in words[1:])
        return False
    
    def _has_codes(self, query: str) -> bool:
        """Check if query contains codes or IDs."""
        # Pattern: numbers, hyphens, underscores
        return bool(re.search(r'[A-Z0-9]{2,}[-_][A-Z0-9]+', query))

# Usage
adaptive = AdaptiveHybridSearch(
    vector_store=vector_store,
    embedding=embedding
)

# Conceptual query → high vector weight
results1 = await adaptive.search("What is machine learning?")

# Proper noun query → high keyword weight
results2 = await adaptive.search("Find documents about PyTorch")

# Code query → high keyword weight
results3 = await adaptive.search("error code ERR_123")
```

## Integration with RAG Pipeline

### Custom Hybrid RAG

```python
from quaerium import RagPipeline
from quaerium.infra.llm import OpenAILLM

class HybridRagPipeline(RagPipeline):
    """RAG pipeline with hybrid search."""
    
    def __init__(
        self,
        embedding_client: OpenAIEmbedding,
        vector_store: MilvusVectorStore,
        llm_client: OpenAILLM,
        alpha: float = 0.7,
    ):
        super().__init__(
            embedding_client=embedding_client,
            vector_store=vector_store,
            llm_client=llm_client
        )
        self.hybrid_searcher = HybridSearcher(
            vector_store=vector_store,
            embedding_client=embedding_client,
            alpha=alpha
        )
    
    async def query(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True,
        **kwargs
    ):
        """Query with optional hybrid search."""
        if use_hybrid:
            # Use hybrid search
            retrieved = await self.hybrid_searcher.search(
                query=query,
                limit=limit
            )
        else:
            # Use standard vector search
            retrieved = await self.vector_store.search(
                query=query,
                limit=limit
            )
        
        # Assemble context
        context = "\n\n".join(
            f"[{i+1}] {doc.text}"
            for i, doc in enumerate(retrieved)
        )
        
        # Generate answer
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = await self.llm_client.generate(prompt)
        
        return {
            "answer": answer,
            "sources": retrieved,
            "context": context
        }

# Usage
pipeline = HybridRagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=OpenAILLM(),
    alpha=0.7
)

# Query with hybrid search
result = await pipeline.query(
    "What are the key features of Python?",
    use_hybrid=True
)

print(result["answer"])
```

## Query-Specific Hybrid Strategies

### Multi-Strategy Hybrid

```python
class MultiStrategyHybrid:
    """Hybrid search with multiple strategies."""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding: OpenAIEmbedding,
    ):
        self.vector_store = vector_store
        self.embedding = embedding
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        strategy: str = "auto",
    ) -> list[SearchResult]:
        """Multi-strategy hybrid search.
        
        Args:
            query: Search query
            limit: Number of results
            strategy: "vector", "keyword", "balanced", "auto"
        """
        if strategy == "auto":
            strategy = self._detect_strategy(query)
        
        if strategy == "vector":
            # Pure vector search
            return await self.vector_store.search(query, limit)
        
        elif strategy == "keyword":
            # Pure keyword search
            return await self._keyword_search(query, limit)
        
        elif strategy == "balanced":
            # Balanced hybrid
            searcher = HybridSearcher(
                vector_store=self.vector_store,
                embedding_client=self.embedding,
                alpha=0.5
            )
            return await searcher.search(query, limit)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _detect_strategy(self, query: str) -> str:
        """Auto-detect best strategy for query."""
        # Question words → vector
        if any(query.lower().startswith(w) for w in ["what", "how", "why"]):
            return "vector"
        
        # Exact phrases → keyword
        if '"' in query:
            return "keyword"
        
        # IDs or codes → keyword
        if re.search(r'[A-Z0-9]{3,}[-_]', query):
            return "keyword"
        
        # Default: balanced
        return "balanced"
    
    async def _keyword_search(
        self,
        query: str,
        limit: int
    ) -> list[SearchResult]:
        """Pure keyword search implementation."""
        # Implementation details...
        pass
```

## Performance Optimization

### Parallel Search

Execute vector and keyword searches in parallel:

```python
import asyncio

class ParallelHybridSearch:
    """Hybrid search with parallel execution."""
    
    async def search(
        self,
        query: str,
        limit: int = 10
    ) -> list[SearchResult]:
        """Parallel hybrid search."""
        # Execute searches in parallel
        vector_task = self.vector_store.search(query, limit * 2)
        keyword_task = self._keyword_search(query, limit * 2)
        
        vector_results, keyword_results = await asyncio.gather(
            vector_task,
            keyword_task
        )
        
        # Combine results
        return self._reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )[:limit]
```

### Caching

Cache hybrid search results:

```python
from functools import lru_cache
import hashlib

class CachedHybridSearch:
    """Hybrid search with result caching."""
    
    def __init__(self, vector_store, embedding):
        self.vector_store = vector_store
        self.embedding = embedding
        self._cache = {}
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        use_cache: bool = True
    ) -> list[SearchResult]:
        """Cached hybrid search."""
        # Generate cache key
        cache_key = self._make_cache_key(query, limit)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Perform search
        results = await self._hybrid_search(query, limit)
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = results
        
        return results
    
    def _make_cache_key(self, query: str, limit: int) -> str:
        """Generate cache key."""
        content = f"{query}:{limit}"
        return hashlib.md5(content.encode()).hexdigest()
```

## Evaluation

### Compare Strategies

```python
async def evaluate_hybrid_search(
    queries: list[str],
    ground_truth: dict[str, list[str]],
    searcher: HybridSearcher,
):
    """Evaluate hybrid search performance."""
    metrics = {
        "precision@5": [],
        "recall@5": [],
        "mrr": []
    }
    
    for query in queries:
        # Search
        results = await searcher.search(query, limit=5)
        retrieved_ids = [r.id for r in results]
        
        # Ground truth
        relevant_ids = ground_truth.get(query, [])
        
        if not relevant_ids:
            continue
        
        # Precision@5
        relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
        precision = len(relevant_retrieved) / len(retrieved_ids)
        metrics["precision@5"].append(precision)
        
        # Recall@5
        recall = len(relevant_retrieved) / len(relevant_ids)
        metrics["recall@5"].append(recall)
        
        # MRR
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                metrics["mrr"].append(1.0 / (i + 1))
                break
        else:
            metrics["mrr"].append(0.0)
    
    # Average metrics
    return {
        key: sum(values) / len(values)
        for key, values in metrics.items()
    }

# Compare strategies
results_vector = await evaluate_hybrid_search(
    queries, ground_truth, 
    HybridSearcher(alpha=1.0)  # Pure vector
)

results_keyword = await evaluate_hybrid_search(
    queries, ground_truth,
    HybridSearcher(alpha=0.0)  # Pure keyword
)

results_hybrid = await evaluate_hybrid_search(
    queries, ground_truth,
    HybridSearcher(alpha=0.7)  # Hybrid
)

print("Vector:", results_vector)
print("Keyword:", results_keyword)
print("Hybrid:", results_hybrid)
```

## Best Practices

1. **Choose the Right Alpha**
   - Start with 0.7 (70% vector, 30% keyword)
   - Adjust based on your data and queries
   - Use adaptive weighting for different query types

2. **Optimize Both Search Paths**
   - Tune vector index (HNSW, IVF)
   - Optimize keyword index (BM25 parameters)
   - Consider caching frequently used results

3. **Use Reciprocal Rank Fusion**
   - More robust than weighted averaging
   - Less sensitive to score distribution
   - Works well with different score ranges

4. **Test on Real Queries**
   - Evaluate on your actual use case
   - Compare pure vs. hybrid approaches
   - A/B test different alpha values

5. **Monitor Performance**
   - Track search latency
   - Measure retrieval quality
   - Log query patterns for tuning

## Next Steps

- [Advanced Pipeline](advanced_pipeline.md) - Complete RAG pipeline
- [Reranking Guide](../guides/reranking.md) - Improve ranking quality
- [Vector Stores](../guides/vector_stores.md) - Vector search deep dive

## See Also

- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Milvus Hybrid Search](https://milvus.io/docs/hybrid_search.md)
