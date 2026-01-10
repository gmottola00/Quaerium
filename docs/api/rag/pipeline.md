# RAG Pipeline

Complete RAG pipeline implementation with query rewriting, retrieval, reranking, and generation.

## RagPipeline

::: rag_toolkit.rag.pipeline.RagPipeline
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - query
        - index_documents
      heading_level: 3

## Response Models

::: rag_toolkit.rag.models.RagResponse
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: rag_toolkit.rag.models.RetrievedChunk
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Pipeline

```python
from rag_toolkit.rag import RagPipeline
from rag_toolkit.infra import get_ollama_embedding, get_ollama_llm
from rag_toolkit.infra.vectorstores import get_qdrant_service

# Initialize components
embedding = get_ollama_embedding(model="nomic-embed-text")
llm = get_ollama_llm(model="llama3.2")
vector_store = get_qdrant_service(
    host="localhost",
    collection_name="docs"
)

# Create pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
)

# Index documents
pipeline.index_documents([
    "RAG combines retrieval and generation.",
    "Vector stores enable semantic search.",
])

# Query
response = pipeline.query("What is RAG?")
print(response.answer)
print(f"Sources: {len(response.sources)}")
```

### Advanced Pipeline with Reranking

```python
from rag_toolkit.rag import RagPipeline
from rag_toolkit.rag.rerankers import LLMReranker

# Create reranker
reranker = LLMReranker(llm_client=llm)

# Create pipeline with reranking
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
    reranker=reranker,  # Add reranking
    top_k=20,           # Retrieve more candidates
    rerank_top_k=5,     # Rerank to top 5
)

response = pipeline.query("Complex query requiring reranking")
```

### Custom Query Processing

```python
# Query with metadata filtering
response = pipeline.query(
    query="What is RAG?",
    filter_dict={"category": "tutorial", "level": "beginner"}
)

# Access response details
for i, source in enumerate(response.sources, 1):
    print(f"Source {i}:")
    print(f"  Text: {source.text[:100]}...")
    print(f"  Score: {source.score:.3f}")
    print(f"  Metadata: {source.metadata}")
```

## See Also

- [User Guide: RAG Pipeline](../../guides/rag_pipeline.md) - Detailed explanation
- [Examples: Basic RAG](../../examples/basic_rag.md) - Working example
- [Examples: Advanced Pipeline](../../examples/advanced_pipeline.md) - Production setup
