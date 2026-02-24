# API Reference

Complete API documentation for RAG Toolkit.

## Overview

The API is organized into several modules:

<div class="grid cards" markdown>

-   :material-protocol: **[Core Protocols](core/protocols.md)**

    Base protocols for embeddings, LLMs, and vector stores

-   :material-shape: **[Core Types](core/types.md)**

    Type definitions and data models

-   :material-pipeline: **[RAG Pipeline](rag/pipeline.md)**

    Complete RAG pipeline implementation

-   :material-swap-horizontal: **[Migration](migration/migrator.md)**

    Vector store migration tools

</div>

## Using the API Reference

Each page includes:

- **Class/Function signatures** with type hints
- **Parameters** with descriptions and types
- **Return values** and exceptions
- **Examples** showing usage
- **Source code** links

## Quick Navigation

### Core Module

```python
from quaerium.core import (
    EmbeddingClient,  # Protocol for embeddings
    LLMClient,        # Protocol for LLMs
    VectorStoreClient # Protocol for vector stores
)
```

### RAG Module

```python
from quaerium.rag import (
    RagPipeline,      # Main RAG pipeline
    RagResponse,      # Response model
    RetrievedChunk    # Retrieved document chunk
)
```

### Migration Module

```python
from quaerium.migration import (
    VectorStoreMigrator,  # Migration engine
    MigrationResult,      # Result model
    MigrationError        # Exception types
)
```

## Convention Notes

- **Protocols**: Duck-typed interfaces (no inheritance needed)
- **Type Hints**: All public APIs are fully typed
- **Async Support**: Currently synchronous (async coming in v0.2.0)
- **Exceptions**: All raise descriptive exceptions with context
