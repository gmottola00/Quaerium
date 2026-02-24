# :material-content-cut: Chunking Strategies

Chunking is **critical** for RAG quality. Learn how to split documents effectively for optimal embedding and retrieval.

---

## :material-help-circle: Why Chunking Matters

!!! warning "The Problem with Large Documents"
    
    **Issues:**
    
    - :material-close-circle: Embeddings lose semantic focus
    - :material-alert: Context windows overflow
    - :material-target-off: Poor retrieval precision
    - :material-speedometer-slow: Slow processing

!!! success "Benefits of Proper Chunking"
    
    - :material-check-circle: Focused semantic embeddings
    - :material-bullseye-arrow: Precise retrieval
    - :material-text-box-check: Better LLM context utilization
    - :material-speedometer: Efficient processing

```mermaid
graph LR
    A[📄 Large Document] --> B[✂️ Chunking]
    B --> C[📝 Chunk 1]
    B --> D[📝 Chunk 2]
    B --> E[📝 Chunk 3]
    
    C --> F[🔢 Focused Embedding]
    D --> G[🔢 Focused Embedding]
    E --> H[🔢 Focused Embedding]
    
    style A fill:#ffebee
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
```

---

## :material-strategy: Available Chunking Strategies

!!! info "Two Complementary Approaches"
    RAG Toolkit provides two chunking strategies that work together for optimal results.

### 1. :material-file-tree: DynamicChunker - Structural Chunking

!!! abstract "Document Structure-Based"
    Creates chunks based on document structure using heading hierarchy.

**Best for**: Structured documents with clear sections (PDFs, documentation, reports)

```python title="dynamic_chunking.py" linenums="1" hl_lines="4-7 11"
from quaerium.core.chunking import DynamicChunker

# Create dynamic chunker
chunker = DynamicChunker(
    include_tables=True,        # Include table blocks
    max_heading_level=6,        # Maximum heading depth
    allow_preamble=False        # Handle content before first heading
)

# Build chunks from parsed pages
# Input: pages from document parser with blocks
chunks = chunker.build_chunks(pages)

for chunk in chunks:
    print(f"Title: {chunk.title}")
    print(f"Level: {chunk.heading_level}")
    print(f"Pages: {chunk.page_numbers}")
    print(f"Text length: {len(chunk.text)}")
```

**How it works:**

```mermaid
graph TB
    A[📄 Document] --> B{Level-1 Headings}
    B --> C[📑 Section 1]
    B --> D[📑 Section 2]
    B --> E[📑 Section 3]
    
    C --> F[h2, h3, paragraphs, tables]
    D --> G[h2, h3, paragraphs, tables]
    E --> H[h2, h3, paragraphs, tables]
    
    style A fill:#e3f2fd
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
```

1. Splits document at level-1 headings (h1)
2. Each chunk includes:
    - Level-1 heading as title
    - All sub-headings (h2-h6) under that section
    - Paragraphs, lists, and optionally tables
    - Continues until the next level-1 heading

**Configuration options:**

=== "Include Tables"
    ```python
    chunker = DynamicChunker(include_tables=True)
    ```
    
    Tables are included as structured text within chunks.

=== "Limit Heading Depth"
    ```python
    chunker = DynamicChunker(max_heading_level=3)  # Only h1-h3
    ```
    
    Ignore deeper heading levels for simpler structure.

=== "Handle Preamble"
    ```python
    chunker = DynamicChunker(allow_preamble=True)
    ```
    
    Creates a "Preamble" chunk for content before the first heading.

### 2. :material-code-tags: TokenChunker - Token-Based Chunking

!!! abstract "Token Budget Control"
    Splits larger chunks into smaller token-based pieces with intelligent overlap.

**Best for**: Controlling token budget, optimal embedding sizes, LLM context limits

```python title="token_chunking.py" linenums="1" hl_lines="4-7 11"
from quaerium.core.chunking import TokenChunker

# Create token chunker
chunker = TokenChunker(
    max_tokens=800,         # Maximum tokens per chunk
    min_tokens=400,         # Minimum tokens per chunk
    overlap_tokens=120,     # Overlap between chunks
)

# Chunk structured chunks into token chunks
# Input: Chunk objects from DynamicChunker
token_chunks = chunker.chunk(structured_chunks)

for chunk in token_chunks:
    print(f"ID: {chunk.id}")
    print(f"Section: {chunk.section_path}")
    print(f"Text: {chunk.text[:100]}...")
    print(f"Metadata: {chunk.metadata}")
```

**How it works:**

```mermaid
graph LR
    A[📝 Large Chunk<br/>2400 tokens] --> B{TokenChunker}
    B --> C[📄 Chunk 1<br/>tokens 0-800]
    B --> D[📄 Chunk 2<br/>tokens 680-1480]
    B --> E[📄 Chunk 3<br/>tokens 1360-2160]
    
    C -.120 overlap.-> D
    D -.120 overlap.-> E
    
    style A fill:#ffebee
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
```

1. Takes chunks from DynamicChunker
2. Splits them by token count (whitespace tokenizer)
3. Applies overlap to preserve context at boundaries
4. Maintains section hierarchy and metadata

**Token overlap visualization:**

```
Chunk 1: tokens[0:800]
         ████████████████████░░░░
Chunk 2:           ░░░░████████████████████░░░░  # 120 overlap
Chunk 3:                      ░░░░████████████████████  # 120 overlap
```

**Configuration:**

=== "Small Chunks (Precise)"
    ```python
    chunker = TokenChunker(
        max_tokens=512,
        min_tokens=200,
        overlap_tokens=50
    )
    ```
    
    Best for precise retrieval, shorter contexts.

=== "Large Chunks (Context)"
    ```python
    chunker = TokenChunker(
        max_tokens=1500,
        min_tokens=800,
        overlap_tokens=200
    )
    ```
    
    Best for more context, longer documents.

=== "Custom Tokenizer"
    ```python
    def my_tokenizer(text: str) -> list[str]:
        # Your tokenization logic
        return text.split()
    
    chunker = TokenChunker(
        max_tokens=800,
        tokenizer=my_tokenizer
    )
    ```
    
    Use your own tokenization strategy.

## Two-Stage Chunking Pipeline

The recommended approach is to use both chunkers together:

```python
from quaerium.core.chunking import DynamicChunker, TokenChunker

# Stage 1: Structural chunking
dynamic_chunker = DynamicChunker(
    include_tables=True,
    max_heading_level=6,
    allow_preamble=False
)

# Stage 2: Token-based chunking
token_chunker = TokenChunker(
    max_tokens=800,
    min_tokens=400,
    overlap_tokens=120
)

# Process document
# 1. Parse document (PDF, DOCX, etc.)
pages = parser.parse("document.pdf")

# 2. Create structural chunks
structural_chunks = dynamic_chunker.build_chunks(pages)

# 3. Split into token chunks
final_chunks = token_chunker.chunk(structural_chunks)

# Now ready for embedding and indexing
for chunk in final_chunks:
    embedding = await embed(chunk.text)
    await vector_store.insert(embedding, chunk.text, chunk.metadata)
```

## Chunk Types

### Chunk (from DynamicChunker)

Basic structural chunk with heading information:

```python
@dataclass
class Chunk:
    id: str                           # Unique identifier
    title: str                        # Section heading
    heading_level: int                # Heading depth (1-6)
    text: str                         # Full text content
    blocks: List[Dict[str, Any]]     # Structured blocks
    page_numbers: List[int]          # Source pages
```

### TokenChunk (from TokenChunker)

Token-based chunk with metadata:

```python
@dataclass
class TokenChunk:
    id: str                           # Unique ID (parent:start-end)
    text: str                         # Chunk text
    section_path: str                 # Full section hierarchy
    metadata: Dict[str, str]         # Extracted metadata
    page_numbers: List[int]          # Source pages
    source_chunk_id: str             # Parent chunk ID
```

## Metadata Extraction

TokenChunker automatically extracts metadata from text:

```python
# Metadata patterns (configurable in source)
TENDER_CODE_PATTERN = r"\b\d{6}-\d{4}\b"
LOT_ID_PATTERN = r"\bLOT[-_\s]?\w+\b"

# Document type keywords
DOC_TYPE_KEYWORDS = {
    "bando": "tender_notice",
    "avviso": "notice",
    "rettifica": "corrigendum",
    "capitolato": "specs",
    "disciplinare": "disciplinare",
}

# Example extracted metadata
{
    "tender_code": "123456-2024",
    "lot_id": "lot_1",
    "document_type": "tender_notice",
    "clause_type": "article"
}
```

## Section Path Hierarchy

TokenChunker builds a hierarchical section path:

```python
# Example document structure:
# H1: Chapter 1
#   H2: Section 1.1
#     H3: Subsection 1.1.1

# Resulting section_path:
"Chapter 1 > Section 1.1 > Subsection 1.1.1"
```

This helps maintain context when chunks are retrieved.

## Chunk Size Guidelines

| Document Type | max_tokens | min_tokens | overlap | Reasoning |
|---------------|------------|------------|---------|-----------|
| Technical docs | 800 | 400 | 120 | Balanced context |
| Short Q&A | 512 | 200 | 50 | Precise answers |
| Long reports | 1500 | 800 | 200 | More context |
| Code files | 600 | 300 | 60 | Function-level |

## Best Practices

### 1. Use Two-Stage Pipeline

Always use DynamicChunker → TokenChunker for best results:

```python
# ✅ Good: Two-stage chunking
structural = dynamic_chunker.build_chunks(pages)
final = token_chunker.chunk(structural)

# ❌ Bad: Only one chunker
final = token_chunker.chunk(raw_text)  # Loses structure
```

### 2. Configure Overlap

Always use overlap (15-20% of max_tokens):

```python
# ✅ Good: 15% overlap
TokenChunker(max_tokens=800, overlap_tokens=120)

# ❌ Bad: No overlap
TokenChunker(max_tokens=800, overlap_tokens=0)
```

### 3. Preserve Tables

Include tables for complete information:

```python
# ✅ Good: Include tables
DynamicChunker(include_tables=True)

# ⚠️  Careful: May lose important data
DynamicChunker(include_tables=False)
```

### 4. Use Metadata for Filtering

Leverage extracted metadata:

```python
# Search with metadata filters
results = await vector_store.search(
    query_embedding,
    filter={
        "document_type": "tender_notice",
        "lot_id": "lot_1"
    }
)
```

## Integration with RAG Pipeline

Complete example with quaerium:

```python
from quaerium.core.chunking import DynamicChunker, TokenChunker
from quaerium.infra.embedding import OpenAIEmbedding
from quaerium.core.vectorstore import MilvusVectorStore

# Initialize components
dynamic_chunker = DynamicChunker()
token_chunker = TokenChunker(max_tokens=800)
embedding = OpenAIEmbedding(model="text-embedding-3-small")
vector_store = MilvusVectorStore(collection_name="documents")

# Process and index document
async def index_document(document_path: str):
    # 1. Parse
    pages = parser.parse(document_path)
    
    # 2. Structural chunking
    structural_chunks = dynamic_chunker.build_chunks(pages)
    
    # 3. Token chunking
    token_chunks = token_chunker.chunk(structural_chunks)
    
    # 4. Embed and index
    for chunk in token_chunks:
        embedding_vector = await embedding.embed(chunk.text)
        
        await vector_store.insert(
            id=chunk.id,
            vector=embedding_vector,
            text=chunk.text,
            metadata={
                **chunk.metadata,
                "section_path": chunk.section_path,
                "pages": chunk.page_numbers
            }
        )
    
    print(f"Indexed {len(token_chunks)} chunks")

# Index
await index_document("report.pdf")
```

## Customization

### Custom Tokenizer

```python
import tiktoken

# Use OpenAI tokenizer
def openai_tokenizer(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return [enc.decode([t]) for t in tokens]

chunker = TokenChunker(
    max_tokens=800,
    tokenizer=openai_tokenizer
)
```

### Domain-Specific Metadata

Extend metadata extraction by modifying patterns:

```python
# In chunking.py, add your patterns:
CUSTOM_PATTERN = re.compile(r"YOUR_REGEX")

def _extract_metadata(text: str, title: str | None = None) -> Dict[str, str]:
    metadata = {}
    
    # Add your extraction logic
    if CUSTOM_PATTERN.search(text):
        metadata["custom_field"] = ...
    
    return metadata
```

## Troubleshooting

### Chunks Too Large

```python
# Reduce max_tokens
TokenChunker(max_tokens=512, min_tokens=200)
```

### Chunks Too Small

```python
# Increase min_tokens
TokenChunker(max_tokens=1200, min_tokens=600)
```

### Missing Context at Boundaries

```python
# Increase overlap
TokenChunker(overlap_tokens=200)  # 25% overlap
```

### Lost Document Structure

```python
# Ensure DynamicChunker is used first
structural = dynamic_chunker.build_chunks(pages)
final = token_chunker.chunk(structural)
```

## Next Steps

- [Embeddings Guide](embeddings.md) - Embed your chunks
- [Vector Stores](vector_stores.md) - Store and retrieve chunks
- [RAG Pipeline](rag_pipeline.md) - Complete RAG setup
- [Production Setup](../examples/production_setup.md) - Deploy to production

## See Also

- [Core Concepts](core_concepts.md) - Chunking fundamentals
- [Architecture](../getting_started/architecture.md) - System design
- [API Reference](../api/index.md) - Complete API docs

Split text into fixed-size chunks with optional overlap.

**Best for**: General documents, simple use cases

```python
from quaerium.core.chunking import FixedSizeChunker

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
from quaerium.core.chunking import TokenChunker

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
from quaerium.core.chunking import SemanticChunker

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
from quaerium.core.chunking import RecursiveChunker

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
from quaerium.core.chunking import MarkdownChunker

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
from quaerium.core.chunking import DynamicChunker

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
from quaerium.core.chunking import MetadataEnricher

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
from quaerium.core.chunking import HierarchicalChunker

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
from quaerium.core.chunking import ContextChunker

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
from quaerium.core.chunking import ChunkSizeOptimizer

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
from quaerium.infra.parsers.pdf import PDFParser
from quaerium.core.chunking import SemanticChunker

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
from quaerium.core.chunking import CodeChunker

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
from quaerium.core.chunking import StructuredChunker

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
from quaerium import RagPipeline
from quaerium.core.chunking import TokenChunker

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
from quaerium.core.chunking import Preprocessor

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
from quaerium.core.chunking import ChunkQualityEvaluator

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

- [Core Concepts](core_concepts.md) - Chunking fundamentals
- [Architecture](../getting_started/architecture.md) - System design
- [Token Limits](https://platform.openai.com/docs/models) - Model context windows
