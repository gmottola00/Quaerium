# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-22

### 🎉 Initial Release

This is the first official release of `rag-toolkit`, a production-ready library for building Retrieval-Augmented Generation (RAG) systems with Protocol-based architecture.

### Added

#### Core Components
- **Protocol-based Architecture**: Clean interface definitions using Python Protocols (PEP 544)
  - `EmbeddingClient`: Protocol for embedding providers
  - `LLMClient`: Protocol for language model providers
  - `VectorStoreClient`: Protocol for vector database operations
  - `ChunkLike` & `TokenChunkLike`: Protocols for document chunks

#### Chunking System
- **DynamicChunker**: Structure-aware chunking based on document heading hierarchy
  - Splits documents at level-1 headings
  - Preserves nested structure (subsections, paragraphs, lists, tables)
  - Configurable heading levels and table inclusion
  - Preamble handling for content before first heading

- **TokenChunker**: Token-based chunking with overlap
  - Configurable token limits (max, min, overlap)
  - Pluggable tokenizer support
  - Metadata extraction (tender codes, lot IDs, document types)
  - Two-stage pipeline compatibility (DynamicChunker → TokenChunker)

- **Concrete Implementations**: Dataclass implementations of chunking protocols
  - `Chunk`: Standard document chunk
  - `TokenChunk`: Token-optimized chunk with metadata

#### RAG Pipeline
- **RagPipeline**: End-to-end RAG workflow
  - Query rewriting for better retrieval
  - Vector-based search integration
  - LLM-based reranking
  - Context assembly
  - Answer generation with citations

- **Models**: Structured data models
  - `RagResponse`: Generated answer with source citations
  - `RetrievedChunk`: Retrieved document chunk with metadata and scores

#### Infrastructure
- **Multi-provider Support** (via lazy loading):
  - Ollama (embeddings and LLM)
  - OpenAI (embeddings and LLM)
  - Extensible for custom providers

- **Vector Store Abstraction**:
  - Milvus integration (built-in)
  - Protocol-based design for easy provider switching
  - Support for metadata filtering and hybrid search

#### Documentation
- **Comprehensive Sphinx Documentation**:
  - User guides for all core concepts
  - API reference with auto-generated docs
  - Practical examples and tutorials
  - Production deployment guides
  - Published at: https://gmottola00.github.io/rag-toolkit/

#### Development Tools
- **Test Suite**: 28 tests with 19% initial coverage
  - Core protocol compliance tests
  - Chunking strategy tests
  - RAG pipeline integration tests
  - Mock implementations for testing

- **CI/CD**: GitHub Actions workflows
  - Automated testing across Python 3.11, 3.12, 3.13
  - Documentation builds and deployment
  - Code quality checks (ruff, black, isort, mypy)

### Supported Platforms
- **Python**: 3.11, 3.12, 3.13
- **Operating Systems**: Linux, macOS, Windows (via WSL)
- **Vector Databases**: Milvus 2.3+
- **LLM Providers**: Ollama, OpenAI

### Dependencies
- Core: `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`, `pymilvus>=2.3.0`
- Optional: `ollama`, `openai`, `pymupdf`, `python-docx`, `easyocr`, `langdetect`

### Known Limitations
- Test coverage at 19% (focused on core components)
- Milvus is currently the only built-in vector store (more coming soon)
- Document parsers (PDF, DOCX) included but minimally tested
- OCR support experimental

### Breaking Changes
None - initial release.

### Migration Guide
Not applicable - initial release.

---

## [Unreleased]

### Added

#### Vector Store Integrations (2025-12-23 to 2025-12-26)
- **Qdrant Integration**: Full implementation with connection management, collection operations, and hybrid search
- **ChromaDB Integration**: Complete support for local and client-server deployments
- **Unified Testing Framework**: Comprehensive test suite for all vector store implementations
- **Docker Compose Setup**: Development environment with Milvus and Qdrant containers
- **Benchmark Framework**: Performance comparison tool across vector stores (Milvus, Qdrant, ChromaDB)
  - Query latency measurement
  - Indexing throughput analysis
  - Memory usage profiling
  - HTML report generation

#### Migration Tools (2026-01-02)
- **VectorStoreMigrator**: Production-ready migration engine for vector data transfers across stores (Milvus, Qdrant, ChromaDB)
- **Advanced Features**: Filtered migration, dry-run mode, retry logic with exponential backoff
- **Models & Exceptions**: Complete migration lifecycle support with validation and error handling
- **Documentation**: Comprehensive guide with production examples and roadmap
- **Test Coverage**: 60 tests (100% pass rate)

### Changed
- **Documentation**: Updated roadmap reflecting completed vector store integrations
- **Infrastructure**: Enhanced Docker setup for local development and testing

### Planned Features (Next Releases)

#### Phase 2 Priority 2
- **Incremental Migration**: Checkpoint-based resume capability for large datasets
- **Schema Mapping**: Field transformation and mapping between different vector stores

#### Phase 3
- **CLI Tool**: Command-line interface with YAML configuration support
- **Parallel Migration**: Multi-threaded/multi-process batch processing
- **Metrics & Observability**: Prometheus metrics and OpenTelemetry tracing

#### Future Enhancements
- Enhanced document parsing with better table extraction
- Query expansion strategies
- Caching layer for embeddings and LLM responses
- Async support for all I/O operations
- Evaluation framework for RAG quality metrics
- Examples for production deployments (Kubernetes, cloud platforms)
- Zero-downtime migration with dual-write pattern
- Data quality validation (embedding drift detection)

---

[0.1.0]: https://github.com/gmottola00/rag-toolkit/releases/tag/v0.1.0
