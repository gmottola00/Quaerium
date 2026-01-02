# Core Concepts

Understanding these core concepts will help you make the most of rag-toolkit.

## RAG (Retrieval-Augmented Generation)

RAG combines information retrieval with text generation to create more accurate and contextual responses.

### Traditional LLM

```
User Query ──► LLM ──► Response
```

**Limitations**:
- Limited to training data
- Can hallucinate facts
- No access to private data

### RAG-Enhanced LLM

```
User Query ──► [Retrieval] ──► Context + Query ──► LLM ──► Response
                    ▲
                    │
              [Vector Store]
```

**Benefits**:
- ✅ Access to current information
- ✅ Reduced hallucinations
- ✅ Citations and sources
- ✅ Private data integration

## Key Components

### 1. Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning.

```python
text = "Hello world"
vector = embedding.embed(text)
# [0.23, -0.45, 0.12, ...]  # 768 dimensions
```

**Similar texts have similar vectors**:
```python
embed("dog") ≈ embed("puppy")  # High similarity
embed("dog") ≠ embed("computer")  # Low similarity
```

### 2. Vector Stores

Vector stores enable efficient similarity search over large collections of embeddings.

```python
# Store documents as vectors
store.insert(
    collection="docs",
    vectors=[vector1, vector2, ...],
    texts=["text1", "text2", ...],
)

# Find similar documents
results = store.search(
    collection="docs",
    query_vector=query_vector,
    top_k=5,
)
```

**Popular vector stores**:
- **Milvus**: High-performance, scalable
- **Pinecone**: Managed service
- **Qdrant**: Developer-friendly
- **Weaviate**: GraphQL interface

### 3. LLMs (Large Language Models)

LLMs generate human-like text based on input prompts.

```python
prompt = """
Context: {retrieved_context}

Question: {user_question}

Answer:"""

response = llm.generate(prompt)
```

**Popular LLMs**:
- **OpenAI GPT-4**: Most capable
- **Anthropic Claude**: Long context
- **Ollama/Llama**: Local/private
- **Mistral**: Efficient

### 4. Chunking

Splitting documents into manageable pieces for embedding and retrieval.

```python
document = "Very long document..."

chunks = chunker.chunk(
    document,
    chunk_size=512,
    chunk_overlap=50,
)
# ["chunk1...", "chunk2...", ...]
```

**Chunking strategies**:
- **Fixed size**: Simple, consistent
- **Sentence-based**: Semantic boundaries
- **Recursive**: Hierarchical splitting
- **Token-aware**: LLM token limits

## The RAG Workflow

### Indexing Phase

```python
# 1. Load documents
documents = load_documents("./data/")

# 2. Chunk documents
chunks = chunker.chunk_documents(documents)

# 3. Generate embeddings
vectors = embedding.embed_batch(chunks)

# 4. Store in vector database
store.insert(
    collection="knowledge_base",
    vectors=vectors,
    texts=chunks,
    metadata=[{"source": doc.source} for doc in documents],
)
```

### Query Phase

```python
# 1. Embed user query
query = "What is RAG?"
query_vector = embedding.embed(query)

# 2. Retrieve similar chunks
results = store.search(
    collection="knowledge_base",
    query_vector=query_vector,
    top_k=5,
)

# 3. Assemble context
context = "\n\n".join([r.text for r in results])

# 4. Generate response
prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

answer = llm.generate(prompt)
```

## Semantic Search vs Keyword Search

### Keyword Search

```python
query = "dog care"
results = keyword_search(query)
# Matches: "dog", "care", "dogs"
```

**Limitations**:
- Exact word matching
- Misses synonyms
- No context understanding

### Semantic Search (Embeddings)

```python
query = "dog care"
query_vector = embed(query)
results = vector_search(query_vector)
# Matches: "pet grooming", "canine health", "puppy training"
```

**Advantages**:
- ✅ Understands meaning
- ✅ Finds related concepts
- ✅ Language-agnostic
- ✅ Context-aware

## Protocols vs Inheritance

### Traditional Approach (Inheritance)

```python
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

class MyEmbedding(BaseEmbedding):  # Must inherit!
    def embed(self, text: str) -> list[float]:
        return [0.0] * 768
```

### rag-toolkit Approach (Protocols)

```python
from typing import Protocol

class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...

class MyEmbedding:  # No inheritance needed!
    def embed(self, text: str) -> list[float]:
        return [0.0] * 768

# Works seamlessly
pipeline = RagPipeline(embedding_client=MyEmbedding())
```

**Benefits**:
- ✅ No inheritance required
- ✅ Duck typing with type safety
- ✅ Easy testing with mocks
- ✅ More flexibility

## Dependency Injection

Components receive their dependencies explicitly:

```python
# ❌ Bad: Hidden dependencies
class Pipeline:
    def __init__(self):
        self.embedding = OllamaEmbedding()  # Hardcoded!
        self.llm = OllamaLLM()              # Can't test!

# ✅ Good: Explicit dependencies
class RagPipeline:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        vector_store: VectorStoreClient,
    ):
        self.embedding = embedding_client
        self.llm = llm_client
        self.store = vector_store
```

**Benefits**:
- ✅ Easy to test (inject mocks)
- ✅ Flexible (swap implementations)
- ✅ Clear dependencies
- ✅ No global state

## Metadata and Filtering

Attach metadata to chunks for filtering:

```python
# Index with metadata
store.insert(
    collection="docs",
    vectors=vectors,
    texts=chunks,
    metadata=[
        {"source": "manual.pdf", "page": 1, "section": "intro"},
        {"source": "manual.pdf", "page": 2, "section": "setup"},
        {"source": "faq.txt", "topic": "installation"},
    ],
)

# Query with filters
results = store.search(
    query_vector=query_vector,
    top_k=5,
    filters={"source": "manual.pdf"},  # Only from manual
)
```

## Retrieval Strategies

### Simple Retrieval

```python
results = store.search(query_vector, top_k=5)
```

### Hybrid Search (Vector + Keyword)

```python
results = store.hybrid_search(
    query_vector=query_vector,
    query_text=query_text,
    alpha=0.5,  # 50% vector, 50% keyword
)
```

### Multi-Query Retrieval

```python
# Generate multiple query variations
queries = [
    "What is RAG?",
    "How does RAG work?",
    "Explain retrieval augmented generation",
]

all_results = []
for q in queries:
    vector = embed(q)
    results = store.search(vector, top_k=3)
    all_results.extend(results)

# Deduplicate and rerank
final_results = deduplicate_and_rerank(all_results)
```

## Context Window Management

LLMs have token limits. Manage context carefully:

```python
# Check token count
def count_tokens(text: str) -> int:
    # Approximate: 1 token ≈ 4 characters
    return len(text) // 4

# Fit within context window
max_context_tokens = 4096
prompt_tokens = 100
answer_tokens = 512

available_tokens = max_context_tokens - prompt_tokens - answer_tokens

# Select chunks that fit
selected_chunks = []
total_tokens = 0

for chunk in ranked_chunks:
    chunk_tokens = count_tokens(chunk.text)
    if total_tokens + chunk_tokens <= available_tokens:
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
```

## Evaluation Metrics

### Retrieval Metrics

**Precision@K**: Relevant results in top K
```python
relevant_in_topk = sum(1 for r in results[:k] if r.is_relevant)
precision = relevant_in_topk / k
```

**Recall@K**: Relevant results found
```python
recall = relevant_in_topk / total_relevant
```

**Mean Reciprocal Rank (MRR)**: Rank of first relevant result
```python
for i, result in enumerate(results, 1):
    if result.is_relevant:
        mrr = 1 / i
        break
```

### Generation Metrics

**BLEU**: N-gram overlap
**ROUGE**: Recall of n-grams
**BERTScore**: Semantic similarity

## Best Practices

### 1. Chunk Size

```python
# ❌ Too small: Loses context
chunk_size = 50

# ❌ Too large: Too much irrelevant info
chunk_size = 5000

# ✅ Just right: Balanced
chunk_size = 512  # Adjust based on your needs
```

### 2. Overlap

```python
# ✅ Add overlap to preserve context
chunk_overlap = chunk_size // 10  # 10% overlap
```

### 3. Metadata

```python
# ✅ Include rich metadata
metadata = {
    "source": doc.filename,
    "page": page_num,
    "section": section_title,
    "timestamp": doc.created_at,
    "author": doc.author,
}
```

### 4. Error Handling

```python
# ✅ Handle errors gracefully
try:
    results = store.search(query_vector, top_k=5)
except Exception as e:
    logger.error(f"Search failed: {e}")
    results = []  # Fallback to empty results
```

## Next Steps

- Learn about [Protocols](protocols.md)
- Explore [Vector Stores](vector_stores.md)
- Read about [Embeddings](embeddings.md)
- See [RAG Pipeline](rag_pipeline.md) for complete workflows
