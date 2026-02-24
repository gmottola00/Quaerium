# Index Layer Overview

Generic vector indexing infrastructure following Clean Architecture principles.

## 🏗️ Architecture

This layer contains **generic, reusable abstractions** for vector indexing and search:

- **`base.py`**: Protocol definitions (pure abstractions, zero dependencies)
- **`service.py`**: `IndexService` generic implementation with dependency injection
- **`search_strategies.py`**: Generic search strategies (VectorSearch, KeywordSearch, HybridSearch)
- **`vector/`**: Milvus-specific implementations (connection, service, data operations)

### Tender-Specific Components (Moved to Domain Layer)

Tender-specific indexing and search have been moved to:
- **`src/domain/tender/indexing/`**: TenderMilvusIndexer (domain-specific indexer)
- **`src/domain/tender/search/`**: VectorSearcher, KeywordSearcher, HybridSearcher (domain-specific search)

## 🚀 Quick Usage

### Generic IndexService (Recommended)

```python
from src.core.index.service import IndexService
from src.core.index.search_strategies import VectorSearch, HybridSearch
from src.core.embedding import EmbeddingClient

# Create index service
index_service = IndexService(
    collection_name="my_collection",
    milvus_uri="http://localhost:19530"
)

# Vector search
vector_search = VectorSearch(index_service, embed_fn=embed_client.embed)
hits = vector_search.search("query", top_k=5)
```

### Tender-Specific Usage

For tender document indexing, use the domain layer:

```python
from src.domain.tender.indexing import TenderMilvusIndexer
from src.domain.tender.search import TenderSearcher
from src.core.embedding import EmbeddingClient

# Setup
indexer = TenderMilvusIndexer(...)
searcher = TenderSearcher(indexer, embed_client)

# Search
vector_results = searcher.vector_search("query", top_k=5)
keyword_results = searcher.keyword_search("keywords", top_k=5)
hybrid_results = searcher.hybrid_search("query", top_k=5)
```

### Generic Search Strategies

For reusable search patterns across any domain:

```python
from src.core.index.search_strategies import VectorSearch, KeywordSearch, HybridSearch
from src.core.index.service import IndexService

# Vector Search
vector_search = VectorSearch(index_service, embed_fn=embed_client.embed)
hits = vector_search.search("query", top_k=5)

# Keyword Search
keyword_search = KeywordSearch(index_service)
hits = keyword_search.search("keywords", top_k=5)

# Hybrid Search (70% vector + 30% keyword)
hybrid = HybridSearch(
    vector_search=vector_search,
    keyword_search=keyword_search,
    alpha=0.7
)
hits = hybrid.search("query", top_k=5)
```

## 🎯 Architecture Benefits

1. **🎯 Clean Separation**: Generic abstractions in core, domain logic in domain layer
2. **🧪 Testable**: Protocol-based design allows easy mocking
3. **🔄 Swappable**: Change vector stores without touching domain code
4. **📦 Reusable**: Core strategies work for any domain
5. **📚 Maintainable**: Clear boundaries between layers

## 📚 Related Documentation

- **Core Layer**: See `../README.md` for core abstractions overview
- **Domain Layer**: See `../../domain/tender/` for tender-specific implementations
- **Examples**: See `../../examples/index_usage.py` for complete usage examples

## Schema Tender Chunks

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `id` | VARCHAR(64) | Primary key |
| `text` | VARCHAR(65535) | Testo del chunk |
| `section_path` | VARCHAR(2048) | Percorso della sezione |
| `tender_id` | VARCHAR(2048) | ID della gara |
| `metadata` | JSON | Metadati aggiuntivi |
| `page_numbers` | JSON | Numeri di pagina |
| `source_chunk_id` | VARCHAR(64) | ID chunk sorgente |
| `embedding` | FLOAT_VECTOR | Vettore embedding |

## Configurazione (Environment Variables)

```bash
# Milvus connection
MILVUS_URI=http://localhost:19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB=default

# Collection config
MILVUS_COLLECTION=tender_chunks
MILVUS_METRIC=IP
MILVUS_INDEX_TYPE=HNSW

# HNSW parameters
MILVUS_HNSW_M=24
MILVUS_HNSW_EF=200
```

## Design Rationale

### Clean Architecture
- **Core (`src/core/index/`)**: Protocol-based abstractions, no vendor dependencies
- **Domain (`src/domain/tender/`)**: Tender-specific indexing and search logic
- **Infra (`src/infra/`)**: Concrete vendor implementations (Milvus, etc.)

### Key Principles
- **Protocol-Based**: Use Python Protocols instead of ABC for loose coupling
- **Dependency Injection**: All dependencies injected, making code testable
- **Layer Separation**: Clear boundaries between core, domain, and infra
- **Swappable Implementations**: Change vendors without touching domain code

---
[Torna al README core](../README.md)
