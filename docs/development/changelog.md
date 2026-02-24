# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.2] - 2026-02-10

### 🎉 First Public Release as "Quaerium"

This release marks the public debut of the package under its new name **Quaerium** (from Latin "quaerere" - to seek, search, inquire). The package includes all features from 0.1.0 plus significant additions in vector stores, migration tools, metadata extraction, and graph RAG capabilities.

### Added

#### Graph RAG with Neo4j
- **GraphStoreClient Protocol**: Protocol interface for graph database operations
  - `create_node()`: Create nodes with labels and properties
  - `create_relationship()`: Create relationships between nodes
  - `query()`: Execute native graph queries (Cypher for Neo4j)
  - `create_constraint()`: Add constraints for data integrity
  - `create_index()`: Add indexes for query performance
  - `get_stats()`: Retrieve graph statistics
  - `clear()`: Clear database (development/testing)

- **Neo4j Implementation**: Production-ready Neo4j 5.x client
  - `Neo4jClient`: Low-level async Neo4j client with connection pooling
  - `Neo4jService`: High-level service implementing GraphStoreClient protocol
  - `Neo4jConfig`: Configuration dataclass for connection settings
  - `create_neo4j_service()`: Factory function with environment variable support
  - Support for local (bolt://) and cloud (neo4j+s://) connections
  - Automatic Neo4j type conversion to JSON-serializable Python types

- **Graph Types**: New dataclasses for graph operations
  - `GraphNode`: Represents graph nodes with labels and properties
  - `GraphRelationship`: Represents relationships between nodes
  - `GraphMetadata`: Type alias for graph metadata

- **Graph Documentation**:
  - Complete Graph RAG guide with examples
  - API reference documentation
  - Working example script (examples/graph_rag_basic.py)
  - Integration with existing documentation

- **Graph Testing**:
  - Unit tests with mocks (no Neo4j required)
  - Integration tests with Docker Neo4j container
  - Makefile targets for graph testing

- **Graph Dependencies**:
  - Optional `neo4j` dependency group: `pip install quaerium[neo4j]`
  - Added to `all` extras group

#### Vector Store Integrations
- **Qdrant Integration**: Full implementation with connection management, collection operations, and hybrid search
- **ChromaDB Integration**: Complete support for local and client-server deployments
- **Unified Testing Framework**: Comprehensive test suite for all vector store implementations
- **Docker Compose Setup**: Development environment with Milvus, Qdrant, and Neo4j containers
- **Benchmark Framework**: Performance comparison tool across vector stores (Milvus, Qdrant, ChromaDB)
  - Query latency measurement
  - Indexing throughput analysis
  - Memory usage profiling
  - HTML report generation

#### Migration Tools
- **VectorStoreMigrator**: Production-ready migration engine for vector data transfers across stores (Milvus, Qdrant, ChromaDB)
- **Advanced Features**: Filtered migration, dry-run mode, retry logic with exponential backoff
- **Models & Exceptions**: Complete migration lifecycle support with validation and error handling
- **Migration Documentation**: Comprehensive guide with production examples and roadmap
- **Test Coverage**: 60 tests (100% pass rate)

#### Metadata Extraction & Enrichment
- **LLMMetadataExtractor**: Generic metadata extractor using LLM prompts (`quaerium.core.metadata`)
  - Customizable system and extraction prompt templates for domain-specific extraction
  - Support for legal, medical, tender, academic, and custom domains
  - Automatic JSON parsing with code block cleanup
  - Graceful error handling (returns empty dict on failure)
  - Custom response parser support for non-JSON formats
  - Test coverage: 13 unit tests (100% coverage)

- **MetadataEnricher**: Chunk text enrichment for improved retrieval (`quaerium.core.chunking`)
  - Adds metadata inline to chunk text for better vector and keyword search
  - Customizable format templates (e.g., `[key: value]`, `(key=value)`)
  - Configurable excluded keys (defaults: file_name, chunk_id, id, source_chunk_id)
  - Batch enrichment support via `enrich_chunks()` method
  - Test coverage: 19 unit tests (100% coverage)

- **Batch Embedding Support**: Enhanced embedding clients with batch operations
  - `OllamaEmbeddingClient.embed_batch()`: Sequential batch processing
  - `OpenAIEmbeddingClient.embed_batch()`: Native batch API for efficiency
  - Improves performance for large-scale indexing operations
  - Backward compatible with existing code

- **Metadata Examples**: Working example demonstrating full metadata extraction pipeline
  - `examples/metadata_extraction_example.py`: Legal document extraction demo
  - Shows complete workflow: extract → chunk → enrich → embed
  - Mock data included for easy testing

### Changed
- **Package Name**: Renamed from "rag-toolkit" to "quaerium"
- **Documentation**: Updated roadmap reflecting completed vector store integrations
- **Infrastructure**: Enhanced Docker setup for local development and testing (Milvus, Qdrant, Neo4j)
- **Package Exports**: Added `LLMMetadataExtractor`, `MetadataEnricher`, and graph components to main package exports
- **Module Docstring**: Updated feature list to include metadata extraction, enrichment, and graph RAG
- **EmbeddingClient Protocol**: Now includes `embed_batch()` method (documented and implemented in all clients)

### Breaking Changes
- Package name changed from `rag-toolkit` to `quaerium`
  - Update all imports: `from rag_toolkit` → `from quaerium`
  - Update installation: `pip install rag-toolkit` → `pip install quaerium`

## [0.1.0] - 2025-12-22

### 🎉 Initial Release

This is the first official release of `quaerium`, a production-ready library for building Retrieval-Augmented Generation (RAG) systems with Protocol-based architecture.

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
  - Published at: https://gmottola00.github.io/quaerium/

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

[0.1.2]: https://github.com/gmottola00/quaerium/releases/tag/v0.1.2
[0.1.0]: https://github.com/gmottola00/quaerium/releases/tag/v0.1.0
