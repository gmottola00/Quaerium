# Reranking

Reranking is a two-stage retrieval technique that dramatically improves result quality. This guide covers everything you need to know about reranking in rag-toolkit.

## Why Reranking?

**Problem with single-stage retrieval:**
- Embedding-based search is fast but imperfect
- May retrieve semantically similar but irrelevant documents
- Limited by embedding quality

**Solution: Two-stage retrieval:**
1. **Stage 1 (Fast)**: Vector search retrieves many candidates (e.g., 50-100)
2. **Stage 2 (Accurate)**: Reranker scores candidates and selects best (e.g., top 5)

**Benefits:**
- ✅ 10-30% improvement in retrieval quality
- ✅ Better context for LLM
- ✅ More relevant answers
- ✅ Minimal latency increase

```{mermaid}
graph LR
    A[Query] --> B[Vector Search]
    B --> C[50 Candidates]
    C --> D[Reranker]
    D --> E[Top 5 Best]
    E --> F[LLM]
```

## Reranker Types

### 1. Cross-Encoder Reranker

Uses deep learning to score query-document pairs.

**Best for**: High accuracy, production systems

```python
from rag_toolkit.rag.rerankers import CrossEncoderReranker

# Create reranker
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Rerank results
candidates = await vector_store.search(query, limit=50)
reranked = await reranker.rerank(
    query="What is machine learning?",
    documents=candidates,
    top_k=5  # Return top 5
)

for doc in reranked:
    print(f"Score: {doc.score:.4f}")
    print(f"Text: {doc.text[:100]}...")
```

**Popular Models:**

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `ms-marco-MiniLM-L-6-v2` | Fast | Good | General purpose |
| `ms-marco-MiniLM-L-12-v2` | Medium | Better | Higher quality |
| `ms-marco-electra-base` | Slower | Best | Maximum quality |

### 2. LLM-Based Reranker

Uses LLM to judge relevance.

**Best for**: Maximum quality, complex queries

```python
from rag_toolkit.rag.rerankers import LLMReranker

# Create LLM reranker
reranker = LLMReranker(
    llm_client=OpenAILLM(model="gpt-4-turbo")
)

# Rerank with LLM
candidates = await vector_store.search(query, limit=20)
reranked = await reranker.rerank(
    query="Explain quantum computing for beginners",
    documents=candidates,
    top_k=5
)
```

**Prompt:**
```
Query: {query}
Document: {document}

Rate the relevance of this document to the query on a scale of 0-1.
Only return the numeric score.
```

### 3. Reciprocal Rank Fusion (RRF)

Combines rankings from multiple retrievers.

**Best for**: Hybrid search, multi-source retrieval

```python
from rag_toolkit.rag.rerankers import ReciprocalRankFusion

# Create RRF reranker
reranker = ReciprocalRankFusion(k=60)

# Get results from multiple sources
vector_results = await vector_store.search(query, limit=20)
keyword_results = await keyword_search(query, limit=20)

# Fuse rankings
fused = await reranker.fuse(
    results_lists=[vector_results, keyword_results]
)
```

**Formula:**
```
score(doc) = Σ (1 / (k + rank_i))
```

### 4. Similarity Reranker

Rerank by semantic similarity to query.

**Best for**: Simple reranking, no external dependencies

```python
from rag_toolkit.rag.rerankers import SimilarityReranker

# Create similarity reranker
reranker = SimilarityReranker(
    embedding_client=OpenAIEmbedding()
)

# Rerank by similarity
reranked = await reranker.rerank(
    query="machine learning",
    documents=candidates,
    top_k=5
)
```

### 5. Diversity Reranker

Maximize diversity in results.

**Best for**: Exploratory search, varied perspectives

```python
from rag_toolkit.rag.rerankers import DiversityReranker

# Create diversity reranker
reranker = DiversityReranker(
    embedding_client=OpenAIEmbedding(),
    lambda_param=0.5  # 0.5 relevance + 0.5 diversity
)

# Get diverse results
reranked = await reranker.rerank(
    query="AI applications",
    documents=candidates,
    top_k=10
)
```

## Integration with RAG

### Basic Integration

```python
from rag_toolkit import RagPipeline
from rag_toolkit.rag.rerankers import CrossEncoderReranker
from rag_toolkit.rag.models import RagConfig

# Create reranker
reranker = CrossEncoderReranker()

# Configure pipeline with reranking
config = RagConfig(
    retrieval_k=50,  # Retrieve 50 candidates
    rerank=True,  # Enable reranking
    rerank_k=5,  # Rerank to top 5
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    config=config,
    reranker=reranker,
)

# Queries automatically use two-stage retrieval
result = await pipeline.query("What is machine learning?")
# 1. Vector search: 50 candidates
# 2. Rerank: top 5 most relevant
# 3. LLM: generate answer from top 5
```

### Custom Reranking Strategy

```python
# Multi-stage reranking
config = RagConfig(
    retrieval_k=100,  # Stage 1: 100 candidates
    rerank=True,
    rerank_k=20,  # Stage 2: rerank to 20
)

# Then apply diversity
diversity_reranker = DiversityReranker()

# In custom pipeline
candidates = await vector_store.search(query, limit=100)
relevant = await cross_encoder.rerank(query, candidates, top_k=20)
final = await diversity_reranker.rerank(query, relevant, top_k=5)
```

## Advanced Reranking

### Weighted Reranking

Combine multiple signals:

```python
from rag_toolkit.rag.rerankers import WeightedReranker

# Create weighted reranker
reranker = WeightedReranker(
    rerankers=[
        (CrossEncoderReranker(), 0.6),  # 60% weight
        (SimilarityReranker(), 0.3),  # 30% weight
        (RecencyReranker(), 0.1),  # 10% weight
    ]
)

# Combined scoring
reranked = await reranker.rerank(
    query="recent AI developments",
    documents=candidates,
    top_k=5
)
```

### Metadata-Aware Reranking

Boost results based on metadata:

```python
from rag_toolkit.rag.rerankers import MetadataReranker

# Create metadata-aware reranker
reranker = MetadataReranker(
    base_reranker=CrossEncoderReranker(),
    metadata_boosts={
        "source": {"research_paper": 1.2, "blog": 0.8},
        "year": lambda y: 1.0 + (int(y) - 2020) * 0.05,  # Boost recent
    }
)

# Rerank with metadata consideration
reranked = await reranker.rerank(
    query="latest ML techniques",
    documents=candidates,
    top_k=5
)
```

### Query-Specific Reranking

Adapt reranking to query type:

```python
from rag_toolkit.rag.rerankers import AdaptiveReranker

# Create adaptive reranker
reranker = AdaptiveReranker(
    query_classifier=QueryClassifier(),
    rerankers={
        "factual": CrossEncoderReranker(),
        "exploratory": DiversityReranker(),
        "recent": RecencyReranker(),
    }
)

# Automatically selects best reranker
reranked = await reranker.rerank(
    query=query,  # Classified automatically
    documents=candidates,
    top_k=5
)
```

## Reranking Configuration

### Candidate Count

```python
# More candidates = better recall, slower
config = RagConfig(
    retrieval_k=100,  # Retrieve 100
    rerank_k=5,  # Select top 5
)

# Fewer candidates = faster, lower recall
config = RagConfig(
    retrieval_k=20,  # Retrieve 20
    rerank_k=5,  # Select top 5
)
```

**Recommended:**
- Production: `retrieval_k=50`, `rerank_k=5`
- High quality: `retrieval_k=100`, `rerank_k=10`
- Fast: `retrieval_k=20`, `rerank_k=3`

### Score Threshold

```python
# Only keep documents above threshold
reranked = await reranker.rerank(
    query=query,
    documents=candidates,
    top_k=10,
    min_score=0.5,  # Minimum relevance score
)

# May return fewer than top_k if scores too low
```

### Batching

```python
# Rerank multiple queries in batch
queries = ["Q1", "Q2", "Q3"]
candidates_lists = [candidates1, candidates2, candidates3]

# Batch reranking
all_reranked = await reranker.rerank_batch(
    queries=queries,
    documents_lists=candidates_lists,
    top_k=5
)
```

## Model Selection

### Cross-Encoder Models

```python
# Fast and good quality (recommended)
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Better quality, slower
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Best quality, slowest
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-electra-base"
)
```

### Domain-Specific Models

```python
# Biomedical
reranker = CrossEncoderReranker(
    model="cross-encoder/biomed-roberta-base"
)

# Legal
reranker = CrossEncoderReranker(
    model="cross-encoder/legal-bert-base"
)

# Multilingual
reranker = CrossEncoderReranker(
    model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
```

## Performance Optimization

### Caching Scores

```python
from functools import lru_cache

class CachedReranker:
    """Reranker with score caching."""
    
    def __init__(self, reranker):
        self.reranker = reranker
        self._cache = {}
    
    async def rerank(self, query: str, documents, top_k: int):
        """Rerank with caching."""
        # Cache key: query + document IDs
        cache_key = (query, tuple(d.id for d in documents))
        
        if cache_key not in self._cache:
            self._cache[cache_key] = await self.reranker.rerank(
                query, documents, top_k
            )
        
        return self._cache[cache_key]

# Usage
cached_reranker = CachedReranker(reranker)
```

### GPU Acceleration

```python
# Use GPU for faster reranking
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda",  # Use GPU
    batch_size=32,  # Larger batches on GPU
)
```

### Batch Processing

```python
# Process multiple queries in parallel
import asyncio

async def batch_rerank(
    queries: list[str],
    documents_lists: list,
    reranker,
    max_concurrent: int = 5
):
    """Rerank multiple queries in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def rerank_one(query, documents):
        async with semaphore:
            return await reranker.rerank(query, documents, top_k=5)
    
    tasks = [
        rerank_one(q, docs)
        for q, docs in zip(queries, documents_lists)
    ]
    
    return await asyncio.gather(*tasks)

# Usage
results = await batch_rerank(queries, candidates_lists, reranker)
```

## Evaluation

### Reranking Quality

```python
from rag_toolkit.rag.evaluation import RerankingEvaluator

# Create evaluator
evaluator = RerankingEvaluator()

# Evaluate reranker
metrics = await evaluator.evaluate(
    reranker=reranker,
    test_queries=queries,
    candidates_lists=all_candidates,
    relevant_doc_ids=relevant_ids,
)

print(f"NDCG@5: {metrics.ndcg_at_5:.3f}")
print(f"MRR: {metrics.mrr:.3f}")
print(f"Precision@5: {metrics.precision_at_5:.3f}")
```

### A/B Testing

```python
# Compare rerankers
reranker_a = CrossEncoderReranker()
reranker_b = LLMReranker()

metrics_a = await evaluator.evaluate(reranker_a, test_data)
metrics_b = await evaluator.evaluate(reranker_b, test_data)

print(f"Reranker A NDCG: {metrics_a.ndcg_at_5:.3f}")
print(f"Reranker B NDCG: {metrics_b.ndcg_at_5:.3f}")

if metrics_b.ndcg_at_5 > metrics_a.ndcg_at_5:
    print("Reranker B wins!")
```

## Custom Rerankers

Implement your own reranker:

```python
from typing import Protocol

class Reranker(Protocol):
    """Protocol for rerankers."""
    
    async def rerank(
        self,
        query: str,
        documents: list,
        top_k: int
    ) -> list:
        """Rerank documents."""
        ...
```

### Example: BM25 Reranker

```python
from rank_bm25 import BM25Okapi

class BM25Reranker:
    """BM25-based reranker."""
    
    def __init__(self):
        self.bm25 = None
    
    async def rerank(
        self,
        query: str,
        documents: list,
        top_k: int
    ) -> list:
        """Rerank using BM25."""
        # Tokenize
        tokenized_docs = [doc.text.split() for doc in documents]
        tokenized_query = query.split()
        
        # Create BM25
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Score
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return [doc for doc, score in doc_scores[:top_k]]

# Usage
bm25_reranker = BM25Reranker()
reranked = await bm25_reranker.rerank(query, candidates, top_k=5)
```

## Monitoring

### Track Reranking Impact

```python
class MonitoredReranker:
    """Reranker with monitoring."""
    
    def __init__(self, reranker):
        self.reranker = reranker
        self.improvements = []
    
    async def rerank(self, query, documents, top_k):
        """Rerank with monitoring."""
        # Original order scores
        original_scores = [d.score for d in documents]
        
        # Rerank
        reranked = await self.reranker.rerank(query, documents, top_k)
        
        # New scores
        new_scores = [d.score for d in reranked]
        
        # Track improvement
        improvement = sum(new_scores) - sum(original_scores[:top_k])
        self.improvements.append(improvement)
        
        return reranked
    
    def get_avg_improvement(self):
        """Get average score improvement."""
        return sum(self.improvements) / len(self.improvements)

# Usage
monitored = MonitoredReranker(reranker)
await monitored.rerank(query, candidates, top_k=5)
print(f"Avg improvement: {monitored.get_avg_improvement():.3f}")
```

## Best Practices

1. **Always Use Reranking in Production**
   - 10-30% quality improvement
   - Minimal latency cost
   - Worth the complexity

2. **Retrieve More, Rerank Less**
   - Retrieve 50-100 candidates
   - Rerank to top 5-10
   - Maximizes recall and precision

3. **Choose the Right Reranker**
   - Cross-encoder: Default choice
   - LLM: Maximum quality, higher cost
   - RRF: Combining multiple sources

4. **Optimize Configuration**
   - Test different candidate counts
   - Tune score thresholds
   - A/B test models

5. **Cache Aggressively**
   - Cache reranking results
   - Significant speed improvement
   - Low memory cost

6. **Monitor Quality**
   - Track NDCG, MRR metrics
   - Compare with/without reranking
   - Iterate based on results

## Troubleshooting

### Reranking Too Slow

```python
# Use faster model
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast model
)

# Reduce candidates
config = RagConfig(retrieval_k=20, rerank_k=5)

# Use GPU
reranker = CrossEncoderReranker(device="cuda")
```

### Poor Reranking Quality

```python
# Use better model
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-electra-base"  # Better quality
)

# Retrieve more candidates
config = RagConfig(retrieval_k=100, rerank_k=10)

# Use LLM reranker
reranker = LLMReranker(llm_client=OpenAILLM(model="gpt-4-turbo"))
```

### Memory Issues

```python
# Reduce batch size
reranker = CrossEncoderReranker(batch_size=8)

# Process in smaller batches
for i in range(0, len(candidates), 20):
    batch = candidates[i:i+20]
    reranked_batch = await reranker.rerank(query, batch, top_k=5)
```

## Next Steps

- [RAG Pipeline](rag_pipeline.md) - Full pipeline integration
- [Advanced Pipeline Example](../examples/advanced_pipeline.md)
- [Production Setup](../examples/production_setup.md)
- [Vector Stores](vector_stores.md) - Initial retrieval

## See Also

- [Core Concepts](core_concepts.md#retrieval-strategies) - Retrieval fundamentals
- [Architecture](../architecture.md) - System design
- [Cross-Encoder Models](https://www.sbert.net/docs/pretrained_cross-encoders.html)
