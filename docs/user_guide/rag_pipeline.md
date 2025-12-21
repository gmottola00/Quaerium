# RAG Pipeline

The RAG (Retrieval-Augmented Generation) pipeline is the heart of rag-toolkit, orchestrating the entire process from document indexing to answering queries. This guide covers everything you need to build production-ready RAG systems.

## Overview

A RAG pipeline combines three core components:

```{mermaid}
graph LR
    A[Documents] --> B[Embedding Client]
    B --> C[Vector Store]
    D[Query] --> B
    B --> C
    C --> E[Retrieved Context]
    E --> F[LLM Client]
    D --> F
    F --> G[Answer]
```

**Components:**
1. **Embedding Client**: Converts text to vectors
2. **Vector Store**: Stores and searches embeddings
3. **LLM Client**: Generates answers from context

## Quick Start

### Basic Pipeline

```python
from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding import OpenAIEmbedding
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore
from rag_toolkit.infra.llm import OpenAILLM

# Create pipeline
pipeline = RagPipeline(
    embedding_client=OpenAIEmbedding(model="text-embedding-3-small"),
    vector_store=MilvusVectorStore(
        collection_name="my_docs",
        embedding_client=OpenAIEmbedding(),
        dimension=1536,
    ),
    llm_client=OpenAILLM(model="gpt-4-turbo"),
)

# Index documents
await pipeline.index(
    texts=[
        "RAG combines retrieval with generation.",
        "Vector databases enable semantic search."
    ],
    metadatas=[
        {"source": "doc1.txt", "page": 1},
        {"source": "doc2.txt", "page": 1}
    ]
)

# Query
result = await pipeline.query("What is RAG?")
print(f"Answer: {result.answer}")
print(f"Sources: {[s.metadata for s in result.sources]}")
```

## Pipeline Configuration

### Constructor Parameters

```python
from rag_toolkit import RagPipeline
from rag_toolkit.rag.models import RagConfig

# Full configuration
pipeline = RagPipeline(
    # Required components
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    
    # Optional configuration
    config=RagConfig(
        retrieval_k=5,  # Number of documents to retrieve
        rerank=True,  # Enable reranking
        rerank_k=3,  # Final number after reranking
        temperature=0.7,  # LLM temperature
        max_tokens=1000,  # LLM max response length
    ),
    
    # Custom prompt template
    prompt_template="""
    Answer the question based on the context below.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """,
)
```

### Retrieval Configuration

```python
# Configure retrieval behavior
config = RagConfig(
    retrieval_k=10,  # Retrieve 10 documents
    retrieval_threshold=0.7,  # Minimum similarity score
    use_metadata_filter=True,  # Enable metadata filtering
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    config=config,
)
```

### LLM Configuration

```python
# Configure LLM behavior
config = RagConfig(
    temperature=0.0,  # Deterministic responses
    max_tokens=500,  # Shorter responses
    system_message="You are a helpful research assistant.",
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    config=config,
)
```

## Indexing Documents

### Basic Indexing

```python
# Index text documents
ids = await pipeline.index(
    texts=["Document 1 content", "Document 2 content"],
    metadatas=[
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc2.pdf", "page": 1}
    ]
)
print(f"Indexed {len(ids)} documents")
```

### Batch Indexing

```python
# Index large datasets in batches
documents = load_large_dataset()  # Returns list of documents

batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    
    await pipeline.index(
        texts=[doc.text for doc in batch],
        metadatas=[doc.metadata for doc in batch]
    )
    
    print(f"Indexed {i+batch_size}/{len(documents)} documents")
```

### Indexing with Custom IDs

```python
# Provide your own document IDs
await pipeline.index(
    ids=["doc-1", "doc-2", "doc-3"],
    texts=["Text 1", "Text 2", "Text 3"],
    metadatas=[{"source": "file1"}, {"source": "file2"}, {"source": "file3"}]
)
```

### Incremental Indexing

```python
# Add new documents without replacing existing ones
await pipeline.index(
    texts=["New document"],
    metadatas=[{"source": "new.pdf", "date": "2024-12-20"}]
)

# Update existing document (same ID)
await pipeline.index(
    ids=["doc-1"],
    texts=["Updated content for doc-1"],
    metadatas=[{"source": "doc1.pdf", "updated": True}]
)
```

## Querying

### Basic Query

```python
# Simple question answering
result = await pipeline.query("What is machine learning?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources used: {len(result.sources)}")
```

### Query with Metadata Filtering

```python
# Filter by document source
result = await pipeline.query(
    "What are the key findings?",
    metadata_filter={"source": "research_paper.pdf"}
)

# Filter by multiple criteria
result = await pipeline.query(
    "Summarize chapter 3",
    metadata_filter={
        "source": "textbook.pdf",
        "chapter": 3,
        "verified": True
    }
)

# Complex filtering
result = await pipeline.query(
    "Recent developments",
    metadata_filter={
        "date": {"$gte": "2024-01-01"},  # After Jan 1, 2024
        "category": {"$in": ["AI", "ML"]},  # In category list
    }
)
```

### Query with Custom K

```python
# Retrieve more/fewer documents
result = await pipeline.query(
    "Explain neural networks",
    k=10  # Retrieve 10 documents instead of default
)
```

### Streaming Responses

```python
# Stream answer tokens in real-time
async for chunk in pipeline.query_stream(
    "Explain quantum computing in detail"
):
    print(chunk, end="", flush=True)
print()  # New line
```

## Advanced Features

### Query Rewriting

Improve retrieval by rewriting queries:

```python
from rag_toolkit.rag.rewriter import QueryRewriter

# Create rewriter
rewriter = QueryRewriter(llm_client=llm)

# Rewrite query
original = "What's ML?"
rewritten = await rewriter.rewrite(original)
print(f"Original: {original}")
print(f"Rewritten: {rewritten}")  # "What is machine learning?"

# Use in pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    query_rewriter=rewriter,  # Enable rewriting
)

# Queries automatically rewritten
result = await pipeline.query("What's ML?")
```

### Reranking

Improve result quality with reranking:

```python
from rag_toolkit.rag.rerankers import CrossEncoderReranker

# Create reranker
reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Configure pipeline with reranking
config = RagConfig(
    retrieval_k=20,  # Retrieve 20 candidates
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

# Queries use two-stage retrieval
result = await pipeline.query("Important information")
# 1. Vector search: 20 candidates
# 2. Rerank: top 5 most relevant
# 3. LLM generation with top 5
```

### Hybrid Search

Combine vector and keyword search:

```python
from rag_toolkit.rag.models import RagConfig

# Enable hybrid search
config = RagConfig(
    retrieval_k=10,
    use_hybrid_search=True,  # Enable hybrid search
    hybrid_alpha=0.7,  # 0.7 vector + 0.3 keyword
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    config=config,
)

# Queries use both vector and keyword matching
result = await pipeline.query("machine learning algorithms")
```

### Context Assembly

Control how context is assembled for the LLM:

```python
from rag_toolkit.rag.assembler import ContextAssembler

# Custom assembler
assembler = ContextAssembler(
    max_tokens=2000,  # Maximum context tokens
    deduplication=True,  # Remove duplicate chunks
    sorting="relevance",  # Sort by relevance (or "source")
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    context_assembler=assembler,
)
```

### Multi-Query

Generate multiple query variations for better retrieval:

```python
# Enable multi-query
config = RagConfig(
    multi_query=True,  # Generate query variations
    multi_query_count=3,  # Number of variations
)

pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    config=config,
)

# Single query becomes multiple queries internally
result = await pipeline.query("What is AI?")
# Internally queries:
# - "What is artificial intelligence?"
# - "Define AI and its applications"
# - "Explain the concept of AI"
```

## Conversational RAG

Maintain conversation history for multi-turn interactions:

```python
from rag_toolkit.rag.models import ConversationHistory

# Initialize conversation history
history = ConversationHistory()

# First question
result1 = await pipeline.query(
    "What is machine learning?",
    conversation_history=history
)
print(result1.answer)

# Follow-up (uses history for context)
result2 = await pipeline.query(
    "What are its applications?",  # "its" refers to ML from previous question
    conversation_history=history
)
print(result2.answer)

# Another follow-up
result3 = await pipeline.query(
    "Give me an example",  # Context from previous turns
    conversation_history=history
)
print(result3.answer)
```

## Custom Prompt Templates

### Template Structure

```python
# Define custom template
custom_template = """
You are an expert {role}.

Context information:
{context}

User question: {question}

Instructions:
- Provide detailed explanations
- Use examples when helpful
- Cite sources using [Source: X]

Your answer:
"""

# Use in pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    prompt_template=custom_template,
)

# Query with custom role
result = await pipeline.query(
    "Explain neural networks",
    template_variables={"role": "AI researcher"}
)
```

### Multi-Language Templates

```python
# English template
en_template = """
Context: {context}
Question: {question}
Answer:
"""

# Italian template
it_template = """
Contesto: {context}
Domanda: {question}
Risposta:
"""

# Create pipelines for each language
en_pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    prompt_template=en_template,
)

it_pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    prompt_template=it_template,
)
```

## Pipeline Evaluation

### Basic Metrics

```python
from rag_toolkit.rag.evaluation import RagEvaluator

# Create evaluator
evaluator = RagEvaluator(pipeline=pipeline)

# Evaluate on test set
test_questions = [
    "What is machine learning?",
    "Explain neural networks",
    "What is gradient descent?"
]

test_answers = [
    "Machine learning is...",
    "Neural networks are...",
    "Gradient descent is..."
]

# Run evaluation
metrics = await evaluator.evaluate(
    questions=test_questions,
    expected_answers=test_answers
)

print(f"Accuracy: {metrics.accuracy:.2f}")
print(f"Relevance: {metrics.relevance:.2f}")
print(f"Faithfulness: {metrics.faithfulness:.2f}")
```

### Retrieval Quality

```python
# Evaluate retrieval only
retrieval_metrics = await evaluator.evaluate_retrieval(
    questions=test_questions,
    relevant_doc_ids=[
        ["doc-1", "doc-2"],  # Relevant for Q1
        ["doc-3", "doc-4"],  # Relevant for Q2
        ["doc-5"],  # Relevant for Q3
    ]
)

print(f"Precision@5: {retrieval_metrics.precision_at_5:.2f}")
print(f"Recall@5: {retrieval_metrics.recall_at_5:.2f}")
print(f"MRR: {retrieval_metrics.mrr:.2f}")
```

## Performance Optimization

### Caching

```python
from rag_toolkit.rag.cache import QueryCache

# Create cache
cache = QueryCache(max_size=1000)

# Wrap pipeline
cached_pipeline = cache.wrap(pipeline)

# First query: normal execution
result1 = await cached_pipeline.query("What is AI?")  # Executes query

# Second query: from cache
result2 = await cached_pipeline.query("What is AI?")  # Returns cached result
```

### Batch Processing

```python
# Process multiple queries in parallel
queries = [
    "What is ML?",
    "Explain neural networks",
    "Define gradient descent"
]

# Parallel execution
results = await pipeline.query_batch(queries)

for query, result in zip(queries, results):
    print(f"Q: {query}")
    print(f"A: {result.answer}\n")
```

### Connection Pooling

```python
# Pipeline automatically manages connections
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=MilvusVectorStore(
        collection_name="docs",
        pool_size=10,  # Connection pool
    ),
    llm_client=llm,
)
```

## Error Handling

### Graceful Degradation

```python
try:
    result = await pipeline.query("What is AI?")
except Exception as e:
    # Log error
    logger.error(f"Query failed: {e}")
    
    # Fallback response
    result = RagResult(
        answer="I apologize, but I'm unable to answer that question right now.",
        sources=[],
        confidence=0.0
    )
```

### Retry Logic

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(3)
)
async def robust_query(query: str):
    """Query with automatic retries."""
    return await pipeline.query(query)

# Usage
result = await robust_query("What is machine learning?")
```

## Monitoring

### Query Logging

```python
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log queries
class LoggedPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    async def query(self, query: str, **kwargs):
        logger.info(f"Query received: {query}")
        
        start = time.time()
        result = await self.pipeline.query(query, **kwargs)
        duration = time.time() - start
        
        logger.info(f"Query completed in {duration:.2f}s")
        logger.info(f"Sources used: {len(result.sources)}")
        
        return result

# Usage
logged_pipeline = LoggedPipeline(pipeline)
result = await logged_pipeline.query("What is AI?")
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')

# Instrument pipeline
class MetricsPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    async def query(self, query: str, **kwargs):
        query_counter.inc()
        
        with query_duration.time():
            result = await self.pipeline.query(query, **kwargs)
        
        return result

# Usage
metrics_pipeline = MetricsPipeline(pipeline)
```

## Best Practices

1. **Component Selection**
   - Use OpenAI embeddings for best quality
   - Use GPT-4-turbo for complex reasoning
   - Use Ollama for privacy/offline needs

2. **Configuration**
   - Start with `retrieval_k=5`
   - Enable reranking for critical applications
   - Use temperature=0.0 for consistency

3. **Indexing**
   - Batch large datasets
   - Include rich metadata
   - Update documents incrementally

4. **Querying**
   - Use metadata filters to narrow scope
   - Enable query rewriting for better retrieval
   - Implement conversation history for multi-turn

5. **Performance**
   - Cache common queries
   - Use connection pooling
   - Monitor query latency

6. **Quality**
   - Evaluate regularly with test sets
   - A/B test different configurations
   - Collect user feedback

## Next Steps

- [Vector Stores Guide](vector_stores.md) - Deep dive into vector stores
- [Embeddings Guide](embeddings.md) - Learn about embeddings
- [LLMs Guide](llms.md) - Master LLM configuration
- [Advanced Pipeline Example](../examples/advanced_pipeline.md)
- [Production Setup](../examples/production_setup.md)

## See Also

- [Core Concepts](core_concepts.md) - RAG fundamentals
- [Protocols](protocols.md) - Understand the protocol system
- [Architecture](../architecture.md) - System design
