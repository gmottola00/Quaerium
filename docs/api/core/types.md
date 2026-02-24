# Core Types

Type definitions and data models used throughout RAG Toolkit.

## Data Models

::: quaerium.core.chunking.models.Chunk
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: quaerium.core.chunking.models.TokenChunk
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Type Aliases

Common type aliases used in the codebase:

```python
from typing import List, Dict, Any, Optional

# Vector types
Vector = List[float]
Vectors = List[Vector]

# Metadata types
Metadata = Dict[str, Any]
Metadatas = List[Metadata]

# Document types
Document = str
Documents = List[Document]
```

## Usage Examples

### Creating Chunks

```python
from quaerium.core.chunking.models import Chunk

chunk = Chunk(
    text="RAG combines retrieval and generation.",
    metadata={
        "source": "docs/intro.md",
        "page": 1,
        "section": "Overview"
    }
)

print(chunk.text)       # Access text
print(chunk.metadata)   # Access metadata
```

### Working with TokenChunks

```python
from quaerium.core.chunking.models import TokenChunk

token_chunk = TokenChunk(
    text="Long document text...",
    metadata={"doc_id": "123"},
    token_count=150,
    char_count=750
)

# Check token limits
if token_chunk.token_count > 512:
    print("Chunk exceeds token limit")
```
