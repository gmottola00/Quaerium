# :material-lightbulb: Core Concepts

Master these fundamental concepts to unlock the full potential of RAG Toolkit.

---

## :material-brain: RAG (Retrieval-Augmented Generation)

!!! abstract "What is RAG?"
    RAG combines **information retrieval** with **text generation** to create more accurate, factual, and contextual responses.

### :material-robot-outline: Traditional LLM

```mermaid
graph LR
    A[👤 User Query] --> B[🤖 LLM] --> C[📝 Response]
    style A fill:#e3f2fd
    style C fill:#ffebee
```

!!! failure "Limitations"
    - :material-close-circle: Limited to training data cutoff
    - :material-alert: Can hallucinate facts
    - :material-lock-off: No access to private/recent data
    - :material-source-branch-remove: No source citations

### :material-brain-circuit: RAG-Enhanced LLM

```mermaid
graph LR
    A[👤 User Query] --> B[🔍 Retrieval]
    B --> C[💾 Vector Store]
    C --> D[📚 Relevant Docs]
    D --> E[📋 Context + Query]
    E --> F[🤖 LLM]
    F --> G[✨ Response]
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

!!! success "Benefits"
    - :material-check-circle: Access to current information
    - :material-shield-check: Reduced hallucinations
    - :material-source-branch-check: Citations and sources
    - :material-database-lock: Private data integration
    - :material-update: Always up-to-date knowledge

---

## :material-puzzle: Key Components

!!! info "The Building Blocks of RAG"

### 1. :material-vector-polyline: Embeddings

!!! abstract "Semantic Vector Representations"
    Embeddings convert text into numerical vectors that capture semantic meaning.

```python title="embedding_example.py" hl_lines="2 3"
text = "Hello world"
vector = embedding.embed(text)
# [0.23, -0.45, 0.12, ...]  # 768 dimensions
```

**Semantic Similarity:**

```python
embed("dog") ≈ embed("puppy")       # High similarity ✓
embed("dog") ≠ embed("computer")    # Low similarity ✗
```

!!! tip "Key Property"
    **Similar texts produce similar vectors** — enabling semantic search beyond keyword matching!

<div class="grid cards" markdown>

- :material-ruler: **Dense Vectors**

    ---

    Typically 384-4096 dimensions per text

- :material-semantic-web: **Semantic Understanding**

    ---

    Captures context, synonyms, relationships

- :material-speedometer: **Efficient Search**

    ---

    Fast vector similarity operations

- :material-language: **Language Agnostic**

    ---

    Works across multiple languages

</div>

### 2. :material-database: Vector Stores

!!! abstract "Efficient Similarity Search"
    Vector stores enable lightning-fast similarity search over millions of embeddings.

```python title="vector_store_example.py" hl_lines="3-7 10-14"
# Store documents as vectors
store.insert(
    collection="docs",
    vectors=[vector1, vector2, vector3],
    texts=["text1", "text2", "text3"],
    metadata=[{"page": 1}, {"page": 2}, {"page": 3}],
)

# Find similar documents
results = store.search(
    collection="docs",
    query_vector=query_vector,
    top_k=5,
)
```

**Popular Vector Stores:**

=== "Milvus"
    ```python
    # High-performance, scalable
    store = MilvusVectorStore(
        host="localhost",
        port="19530",
        collection_name="docs"
    )
    ```
    
    - :material-speedometer: Production-ready performance
    - :material-scale-balance: Horizontal scalability
    - :material-gpu: GPU acceleration support

=== "Pinecone"
    ```python
    # Managed cloud service
    store = PineconeVectorStore(
        api_key="your-key",
        environment="us-west1-gcp"
    )
    ```
    
    - :material-cloud: Fully managed
    - :material-auto-fix: Auto-scaling
    - :material-shield-check: Enterprise security

=== "Qdrant"
    ```python
    # Developer-friendly
    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="docs"
    )
    ```
    
    - :material-api: Rich filtering capabilities
    - :material-docker: Easy Docker setup
    - :material-code-json: JSON-based schema

### 3. :material-robot: LLMs (Large Language Models)

!!! abstract "Natural Language Generation"
    LLMs generate human-like text based on input prompts and retrieved context.

```python title="llm_example.py" hl_lines="2-6"
prompt = """
Context: {retrieved_context}

Question: {user_question}

Answer:"""

response = llm.generate(prompt)
```

**Popular LLMs:**

| Model | Provider | Context | Quality | Cost |
|-------|----------|---------|---------|------|
| **GPT-4** | OpenAI | 128K | :material-star::material-star::material-star::material-star::material-star: | $$$ |
| **Claude 3** | Anthropic | 200K | :material-star::material-star::material-star::material-star::material-star: | $$$ |
| **Llama 3** | Meta (Ollama) | 8K | :material-star::material-star::material-star::material-star: | Free |
| **Mistral** | Mistral (Ollama) | 32K | :material-star::material-star::material-star::material-star: | Free |

!!! tip "Choosing an LLM"
    - **Production**: GPT-4 or Claude 3 for best quality
    - **Development**: Ollama models for free local testing
    - **Cost-sensitive**: GPT-3.5 Turbo for balanced performance

### 4. :material-content-cut: Chunking

!!! abstract "Document Segmentation"
    Splitting documents into manageable pieces for optimal embedding and retrieval.

```python title="chunking_example.py" hl_lines="3-6"
document = "Very long document with lots of content..."

chunks = chunker.chunk(
    document,
    chunk_size=512,
    chunk_overlap=50,
)
# ["chunk1...", "chunk2...", "chunk3..."]
```

**Chunking Strategies:**

<div class="grid cards" markdown>

- :material-format-size: **Fixed Size**

    ---

    ```python
    chunks = fixed_chunker.chunk(
        text, chunk_size=512
    )
    ```
    
    Simple and consistent chunk sizes

- :material-format-text-variant: **Sentence-Based**

    ---

    ```python
    chunks = sentence_chunker.chunk(
        text, max_sentences=10
    )
    ```
    
    Respects semantic boundaries

- :material-file-tree: **Recursive**

    ---

    ```python
    chunks = recursive_chunker.chunk(
        text, separators=["\n\n", "\n", ". "]
    )
    ```
    
    Hierarchical splitting strategy

- :material-code-tags: **Token-Aware**

    ---

    ```python
    chunks = token_chunker.chunk(
        text, max_tokens=800
    )
    ```
    
    Respects LLM token limits

</div>

!!! warning "Chunking Best Practices"
    - **Chunk size**: 300-800 tokens optimal for most use cases
    - **Overlap**: 10-20% overlap preserves context at boundaries
    - **Strategy**: Match chunking to document structure (headings, paragraphs)

---

## :material-workflow: The RAG Workflow

!!! info "End-to-End Process"

### :material-upload: Indexing Phase

```mermaid
graph TB
    A[📁 Documents] --> B[✂️ Chunking]
    B --> C[📝 Text Chunks]
    C --> D[🔢 Embedding]
    D --> E[📊 Vectors]
    E --> F[💾 Vector Store]
    
    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

```python title="indexing_pipeline.py" linenums="1" hl_lines="2 5 8 11-17"
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
    metadata=[{"source": doc.source, "page": doc.page} 
              for doc in documents],
)
```

### :material-magnify: Query Phase

```mermaid
graph TB
    A[💬 User Query] --> B[🔢 Embed Query]
    B --> C[📊 Query Vector]
    C --> D[🔍 Similarity Search]
    D --> E[💾 Vector Store]
    E --> F[📚 Retrieved Docs]
    F --> G[📋 Build Context]
    G --> H[🤖 LLM Generate]
    H --> I[✨ Final Answer]
    
    style A fill:#e3f2fd
    style I fill:#c8e6c9
```

```python title="query_pipeline.py" linenums="1" hl_lines="2 5 8 11 14"
# 1. Embed user query
query = "What is RAG?"
query_vector = embedding.embed(query)

# 2. Search for similar documents
results = store.search(
    collection="knowledge_base",
    query_vector=query_vector,
    top_k=5,
)

# 3. Build context from results
context = "\n\n".join([r.text for r in results])

# 4. Generate answer with LLM
prompt = f"""
Context: {context}

Question: {query}

Answer:"""

answer = llm.generate(prompt)
print(answer)
```

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
