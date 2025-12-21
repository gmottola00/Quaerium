# Chunking Strategies

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval. Good chunking is critical for RAG quality. This guide covers everything you need to know about chunking in rag-toolkit.

## Why Chunking Matters

**Problems with large documents:**
- Embeddings lose semantic focus
- Context windows overflow
- Poor retrieval precision

**Benefits of chunking:**
- ✅ Focused semantic embeddings
- ✅ Precise retrieval
- ✅ Better LLM context utilization
- ✅ Efficient processing

## Chunking Strategies

### 1. Fixed-Size Chunking

Split text into fixed-size chunks with optional overlap.

**Best for**: General documents, simple use cases

```python
from rag_toolkit.core.chunking import FixedSizeChunker

# Create chunker
chunker = FixedSizeChunker(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks
)

# Chunk document
document = "Long document text..." * 100
chunks = await chunker.chunk(document)

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"Chunk {chunk.id}: {len(chunk.text)} chars")
```

**Configuration:**

```python
# Small chunks for precise retrieval
chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=30)

# Large chunks for more context
chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=100)

# No overlap (faster but may miss context)
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=0)
```

### 2. Token-Aware Chunking

Split by token count instead of characters, respecting LLM limits.

**Best for**: Production systems, token budget control

```python
from rag_toolkit.core.chunking import TokenChunker

# Create token-aware chunker
chunker = TokenChunker(
    chunk_size=512,  # Tokens per chunk
    chunk_overlap=50,  # Token overlap
    model="gpt-4",  # Model for tokenization
)

# Chunk with token precision
chunks = await chunker.chunk(document)

for chunk in chunks:
    print(f"Tokens: {chunk.token_count}")
```

**Benefits:**
- Precise token control
- Optimal context window usage
- Model-specific tokenization

### 3. Semantic Chunking

Split at natural semantic boundaries (sentences, paragraphs).

**Best for**: Maintaining context integrity

```python
from rag_toolkit.core.chunking import SemanticChunker

# Create semantic chunker
chunker = SemanticChunker(
    mode="sentence",  # or "paragraph"
    max_chunk_size=500,
    similarity_threshold=0.7,  # Merge similar sentences
)

# Chunk at semantic boundaries
chunks = await chunker.chunk(document)
```

**Modes:**
- `sentence`: Split at sentence boundaries
- `paragraph`: Split at paragraph boundaries
- `section`: Split at section headers

### 4. Recursive Chunking

Hierarchical chunking with fallback strategies.

**Best for**: Complex documents, robust processing

```python
from rag_toolkit.core.chunking import RecursiveChunker

# Create recursive chunker
chunker = RecursiveChunker(
    separators=["\n\n\n", "\n\n", "\n", ". ", " "],  # Try in order
    chunk_size=500,
    chunk_overlap=50,
)

# Chunks with intelligent splitting
chunks = await chunker.chunk(document)
```

**How it works:**
1. Try splitting by first separator (`\n\n\n`)
2. If chunks too large, try next separator (`\n\n`)
3. Continue until target size reached

### 5. Markdown-Aware Chunking

Respect Markdown structure (headers, lists, code blocks).

**Best for**: Documentation, technical content

```python
from rag_toolkit.core.chunking import MarkdownChunker

# Create Markdown chunker
chunker = MarkdownChunker(
    chunk_size=500,
    preserve_code_blocks=True,  # Keep code blocks intact
    header_hierarchy=True,  # Include parent headers
)

# Chunk Markdown
markdown_doc = """
# Chapter 1
## Section 1.1
Content here...

## Section 1.2
More content...
"""

chunks = await chunker.chunk(markdown_doc)

for chunk in chunks:
    print(f"Headers: {chunk.metadata['headers']}")
    print(f"Content: {chunk.text[:50]}...")
```

### 6. Dynamic Chunking

Adapt chunk size based on content characteristics.

**Best for**: Mixed content types, optimal retrieval

```python
from rag_toolkit.core.chunking import DynamicChunker

# Create dynamic chunker
chunker = DynamicChunker(
    min_chunk_size=200,
    max_chunk_size=800,
    target_chunk_size=500,
    adapt_to_content=True,  # Adjust based on content
)

# Automatically optimizes chunk sizes
chunks = await chunker.chunk(document)
```

## Advanced Chunking

### Metadata Enrichment

Add metadata to chunks for better filtering:

```python
from rag_toolkit.core.chunking import MetadataEnricher

# Enrich chunks with metadata
enricher = MetadataEnricher()

chunks = await chunker.chunk(document)
enriched_chunks = await enricher.enrich(
    chunks,
    metadata={
        "source": "research_paper.pdf",
        "author": "John Doe",
        "date": "2024-12-20",
        "category": "AI"
    }
)

# Each chunk now has metadata
for chunk in enriched_chunks:
    print(chunk.metadata)
```

### Hierarchical Chunking

Create parent-child relationships:

```python
from rag_toolkit.core.chunking import HierarchicalChunker

# Create hierarchical chunks
chunker = HierarchicalChunker(
    parent_chunk_size=1000,
    child_chunk_size=200,
    overlap=50,
)

# Returns parents and children
parents, children = await chunker.chunk(document)

# Children reference parents
for child in children:
    print(f"Child: {child.id}")
    print(f"Parent: {child.parent_id}")
```

### Context-Preserving Chunking

Include surrounding context in chunks:

```python
from rag_toolkit.core.chunking import ContextChunker

# Add context to chunks
chunker = ContextChunker(
    chunk_size=500,
    context_before=100,  # Characters before
    context_after=100,  # Characters after
)

chunks = await chunker.chunk(document)

for chunk in chunks:
    print(f"Main: {chunk.text}")
    print(f"Context before: {chunk.context_before}")
    print(f"Context after: {chunk.context_after}")
```

## Chunk Size Selection

### Guidelines

| Document Type | Recommended Size | Reasoning |
|---------------|------------------|-----------|
| Short Q&A | 200-300 chars | Precise answers |
| General docs | 500-800 chars | Balanced |
| Long-form content | 1000-1500 chars | More context |
| Code | 300-500 chars | Function-level |
| Academic papers | 800-1200 chars | Paragraph-level |

### Finding Optimal Size

```python
from rag_toolkit.core.chunking import ChunkSizeOptimizer

# Optimize chunk size for your data
optimizer = ChunkSizeOptimizer(
    pipeline=rag_pipeline,
    test_queries=["Q1", "Q2", "Q3"],
    test_documents=documents,
)

# Test different sizes
results = await optimizer.optimize(
    chunk_sizes=[200, 500, 800, 1000],
    metric="retrieval_precision"
)

best_size = results.best_chunk_size
print(f"Optimal chunk size: {best_size}")
```

## Overlap Configuration

### Why Overlap?

Overlap prevents information loss at chunk boundaries:

```
Without overlap:
[Chunk 1: "...end of sentence."] [Chunk 2: "Start of new..."]
❌ Context break

With overlap:
[Chunk 1: "...end of sentence. Start of"] [Chunk 2: "sentence. Start of new..."]
✅ Context preserved
```

### Recommended Overlap

```python
# General rule: 10-20% of chunk size
chunk_size = 500
overlap = int(chunk_size * 0.15)  # 75 characters

chunker = FixedSizeChunker(
    chunk_size=chunk_size,
    chunk_overlap=overlap
)
```

## Document Type-Specific Chunking

### PDF Documents

```python
from rag_toolkit.infra.parsers.pdf import PDFParser
from rag_toolkit.core.chunking import SemanticChunker

# Parse PDF
parser = PDFParser()
document = await parser.parse("document.pdf")

# Chunk with page awareness
chunker = SemanticChunker(
    mode="paragraph",
    preserve_page_numbers=True,
)

chunks = await chunker.chunk(document)

# Chunks include page numbers
for chunk in chunks:
    print(f"Pages: {chunk.page_numbers}")
```

### Code Files

```python
from rag_toolkit.core.chunking import CodeChunker

# Chunk code by functions/classes
chunker = CodeChunker(
    language="python",
    chunk_by="function",  # or "class", "method"
    include_docstrings=True,
)

code = """
def function1():
    '''Docstring'''
    pass

def function2():
    '''Another docstring'''
    pass
"""

chunks = await chunker.chunk(code)

# Each chunk is a function
for chunk in chunks:
    print(f"Function: {chunk.metadata['function_name']}")
```

### Structured Data

```python
from rag_toolkit.core.chunking import StructuredChunker

# Chunk JSON/CSV
chunker = StructuredChunker(
    format="json",
    chunk_by="record",  # Each record is a chunk
)

json_data = [
    {"id": 1, "text": "Record 1"},
    {"id": 2, "text": "Record 2"},
]

chunks = await chunker.chunk(json_data)
```

## Chunking Pipeline Integration

### With RAG Pipeline

```python
from rag_toolkit import RagPipeline
from rag_toolkit.core.chunking import TokenChunker

# Create chunker
chunker = TokenChunker(chunk_size=512, chunk_overlap=50)

# Create pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    vector_store=vector_store,
    llm_client=llm,
    chunker=chunker,  # Add chunker
)

# Automatically chunks before indexing
await pipeline.index(
    texts=[long_document],  # Single long document
    # Automatically chunked into smaller pieces
)
```

### Custom Preprocessing

```python
from rag_toolkit.core.chunking import Preprocessor

# Create preprocessor
preprocessor = Preprocessor(
    lowercase=False,
    remove_extra_whitespace=True,
    remove_urls=True,
    remove_emails=True,
)

# Preprocess before chunking
cleaned_text = await preprocessor.process(raw_text)
chunks = await chunker.chunk(cleaned_text)
```

## Quality Evaluation

### Chunk Quality Metrics

```python
from rag_toolkit.core.chunking import ChunkQualityEvaluator

# Evaluate chunk quality
evaluator = ChunkQualityEvaluator()

metrics = await evaluator.evaluate(chunks)

print(f"Average chunk size: {metrics.avg_size}")
print(f"Size variance: {metrics.size_variance}")
print(f"Semantic coherence: {metrics.coherence:.2f}")
print(f"Information density: {metrics.density:.2f}")
```

### A/B Testing

```python
# Compare two chunking strategies
chunker_a = FixedSizeChunker(chunk_size=500)
chunker_b = TokenChunker(chunk_size=512)

# Test both
results_a = await test_retrieval(chunker_a, test_queries)
results_b = await test_retrieval(chunker_b, test_queries)

print(f"Strategy A precision: {results_a.precision:.2f}")
print(f"Strategy B precision: {results_b.precision:.2f}")
```

## Performance Optimization

### Parallel Chunking

```python
import asyncio

async def chunk_documents_parallel(
    documents: list[str],
    chunker,
    max_concurrent: int = 10
):
    """Chunk multiple documents in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def chunk_one(doc: str):
        async with semaphore:
            return await chunker.chunk(doc)
    
    tasks = [chunk_one(doc) for doc in documents]
    return await asyncio.gather(*tasks)

# Usage
all_chunks = await chunk_documents_parallel(documents, chunker)
```

### Caching

```python
from functools import lru_cache

class CachedChunker:
    """Chunker with caching."""
    
    def __init__(self, chunker):
        self.chunker = chunker
        self._cache = {}
    
    async def chunk(self, text: str):
        """Chunk with caching."""
        # Use hash as cache key
        key = hash(text)
        
        if key not in self._cache:
            self._cache[key] = await self.chunker.chunk(text)
        
        return self._cache[key]

# Usage
cached_chunker = CachedChunker(chunker)
```

## Best Practices

1. **Choose the Right Strategy**
   - Start with TokenChunker for production
   - Use SemanticChunker for maintaining context
   - Use specialized chunkers for specific content

2. **Optimize Chunk Size**
   - Test different sizes with your data
   - Consider your embedding model's optimal input
   - Balance precision vs context

3. **Use Overlap**
   - Always use 10-20% overlap
   - Increases retrieval quality significantly
   - Small performance cost, big quality gain

4. **Enrich with Metadata**
   - Add source, page, section metadata
   - Enables powerful filtering
   - Improves traceability

5. **Preprocess Text**
   - Remove noise (extra whitespace, etc.)
   - Normalize text encoding
   - Handle special characters

6. **Test and Evaluate**
   - A/B test different strategies
   - Measure retrieval quality
   - Iterate based on results

## Troubleshooting

### Chunks Too Large

```python
# Reduce chunk size
chunker = TokenChunker(
    chunk_size=256,  # Smaller chunks
    chunk_overlap=25
)
```

### Chunks Too Small

```python
# Increase chunk size
chunker = TokenChunker(
    chunk_size=1024,  # Larger chunks
    chunk_overlap=100
)
```

### Context Loss at Boundaries

```python
# Increase overlap
chunker = FixedSizeChunker(
    chunk_size=500,
    chunk_overlap=100  # 20% overlap
)
```

### Poor Semantic Coherence

```python
# Use semantic chunker
chunker = SemanticChunker(
    mode="paragraph",
    max_chunk_size=800
)
```

## Next Steps

- [RAG Pipeline](rag_pipeline.md) - Integrate chunking
- [Embeddings Guide](embeddings.md) - Optimal embedding sizes
- [Vector Stores](vector_stores.md) - Store chunks
- [Production Setup](../examples/production_setup.md)

## See Also

- [Core Concepts](core_concepts.md#chunking) - Chunking fundamentals
- [Architecture](../architecture.md) - System design
- [Token Limits](https://platform.openai.com/docs/models) - Model context windows
