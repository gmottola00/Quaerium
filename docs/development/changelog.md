# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features (Next Releases)

---

## [0.4.1] - 2026-11-02

### Added

- `AgentRunner(on_suspend=...)`: an async callback for a run that stops at the
  gate, e.g. a push notification. It runs in the background, so the stream does
  not wait for it, and its failures are logged. `AgentRunner.aclose()` waits for
  callbacks still in flight.
- `AgentRunner.restore_pending()`: after a restart, marks every run still waiting
  at the gate in the audit log, by asking the checkpointer. Covers runs suspended
  before the upgrade too.
- `AuditLog.load()`, called by `build_audit_log` with Postgres: the most recent
  `capacity` runs, with their tool calls and approvals, are loaded into memory at
  startup. `AuditLog.unfinished()` lists runs with no answer that are not marked
  waiting.
- `AuditLog.suspend_run(..., since=...)`: when the wait began, for restored runs.

### Fixed

- `/api/runs` and `/api/approvals` were empty after every restart: the audit log
  wrote to Postgres but read only from memory.

### Changed

- `{prefix}_tool_calls` has a `summary` column (added with `ALTER TABLE ... IF NOT
  EXISTS` at startup), so a loaded run shows the same one-line summaries.
  Rows written before it get theirs from `output`.

## [0.4.0] - 2026-11-01

### 🤖 Agents kernel, and a lean core install

`quaerium.agents` is now the agent runtime shared by Jarvis (homelab) and Alfred
(finance): a small LangGraph loop around the Anthropic SDK, with a human approval
gate that survives restarts. It replaces the text-based ReAct experiment that was
in this module before, which had no users.

### ⚠️ Breaking changes

- **Core dependencies are now `pydantic` and `pydantic-settings` only.** Backends
  moved to extras: `milvus` (pymilvus), `ollama` (requests), `supabase`, `pdf`
  (pymupdf, pdfplumber, ftfy, charset-normalizer), `docx`, `lang` (lingua).
  `pip install quaerium` no longer pulls in pymilvus — use `quaerium[milvus]`.
- **`quaerium.agents` is a different API.** The ReAct experiment (`AgentRuntime`,
  `ReActExecutor`, `RAGAgent`, …) is removed, and so are its re-exports from the
  top-level `quaerium` package.
- Removed the `ocr` extra (easyocr was never imported) and `aiohttp` (never used).
- Removed `quaerium.core.index.vector`, an unused copy of
  `quaerium.infra.vectorstores.milvus`.

### Added

- **`quaerium.agents`** (extra `agents`):
  - `@tool` — tools as typed async functions; JSON Schema from signature and docstring.
  - `ToolRegistry`, `RiskLevel` (read / write / destructive), `PolicyGate` with
    app-supplied explanation texts.
  - `build_graph` — router → model → gate → act; injectable router heuristic,
    static + per-turn system blocks, effort per route, step ceiling.
  - `AgentRunner` — runs as a flat stream of `RunEvent`s; suspends on destructive
    calls via LangGraph `interrupt` and resumes from the checkpoint.
  - `AnthropicLLM` — adaptive thinking, effort, refusal fallbacks; optional
    LiteLLM gateway for non-Claude models.
  - `checkpointer` (extra `agents-postgres`) with an allowlist serializer and
    `legacy_modules`, so runs suspended before a migration still resume.
  - `MemorySessionStore` / `RedisSessionStore` (extra `agents-redis`) and
    `AuditLog` (Postgres optional), both namespaced per app.
  - `ClientProvider` — real client, optional fake, or a clear configuration error.
  - `rag_tool` / `graph_query_tool` — a `RagPipeline` or a graph store as a tool.
  - `quaerium.agents.testing.ScriptedLLM` — deterministic tests without an API key.
- **`quaerium.agents.server`** (extra `server`): SSE streaming and the shared
  `/api/approvals` and `/api/runs` routers.
- `examples/agent_rag_example.py`, runnable offline.

### Fixed

- The wheel was missing `quaerium.core.index` and `quaerium.infra.evaluation`
  (no `__init__.py` with `namespaces = false`), so `import quaerium.rag` failed
  from an installed package.
- Optional backends are imported lazily: installing one vector store extra no
  longer requires the others.
- `quaerium.__version__` comes from the package metadata.
- Leftover `rag_toolkit` references in tests, Makefile and messages.

### CI

- pytest and ruff (errors) are blocking; a `minimal` job installs with no extras
  and checks the core imports.
- Tests that need live services run with `--run-integration`; vector store unit
  tests that drifted from the code are listed in `tests/known_failures.txt` as
  strict xfail.

---

## [0.1.3] - 2026-02-25

### 🎯 Major Feature: Evaluation & Metrics Framework

Comprehensive evaluation framework for measuring and improving RAG system quality with retrieval metrics, generation quality assessment, and end-to-end performance tracking.

### Added

#### Core Evaluation Protocols
- **MetricCalculator Protocol**: Base protocol for individual metric calculation
- **RetrievalEvaluator Protocol**: Protocol for evaluating retrieval quality
- **GenerationEvaluator Protocol**: Protocol for evaluating generation quality
- **LLMJudge Protocol**: Protocol for LLM-based evaluation
- **PipelineObserver Protocol**: Protocol for non-invasive pipeline monitoring

#### Data Models
- **MetricScore**: Individual metric result with metadata
- **RetrievalMetrics**: Complete retrieval evaluation results (Precision@K, Recall@K, MRR, NDCG, Hit Rate)
- **GenerationMetrics**: Complete generation evaluation results (Relevance, Faithfulness, Hallucination)
- **EndToEndMetrics**: Pipeline performance metrics (Latency, Tokens, Cost, Context Utilization)
- **EvaluationResult**: Unified result container for all metrics
- **EvaluationExample**: Single evaluation example with ground truth
- **DatasetMetadata**: Metadata for evaluation datasets
- **EvaluationDataset Protocol**: Protocol for evaluation dataset management

#### Retrieval Metrics
- **PrecisionAtK**: Precision at K calculator (K=5,10,20,...)
- **RecallAtK**: Recall at K calculator
- **MRRCalculator**: Mean Reciprocal Rank
- **NDCGCalculator**: Normalized Discounted Cumulative Gain
- **HitRateCalculator**: Binary hit rate metric
- **StandardRetrievalEvaluator**: Composite evaluator with all retrieval metrics
- 33 comprehensive tests with 100% coverage

#### Generation Metrics
- **OpenAIJudge**: LLM-as-judge implementation with robust score parsing
- **AnswerRelevanceEvaluator**: Measures if answer addresses the question
- **FaithfulnessEvaluator**: Measures if answer is grounded in context
- **HallucinationDetector**: Detects information not present in context
- **StandardGenerationEvaluator**: Composite evaluator with all generation metrics
- 24 comprehensive tests with mock LLM support

#### Pipeline Integration
- **MetricsObserver**: Non-invasive observer for RAG pipeline with automatic latency tracking, token counting, and cost estimation
- **LatencyTracker**: Stage-by-stage latency tracking
- **TokenTracker**: Token usage and cost estimation for GPT-4, GPT-3.5, GPT-4o models
- Added `observers` parameter to RagPipeline (~20 lines change)
- 20 integration tests with full pipeline simulation

#### Dataset Management
- **JSONLDataset**: JSONL format dataset loader with validation
- Sample dataset with 10 evaluation examples
- 13 dataset loading and validation tests

#### Documentation
- **Evaluation Guide** (500+ lines): Complete guide with examples and best practices
- **API Reference** (600+ lines): Full protocol and implementation documentation
- **Examples Documentation**: Quick start, patterns, and result interpretation
- **Working Example** (327 lines): Complete end-to-end evaluation script

#### Testing
- **90 Total Tests**: All passing in 0.41s
- **100% Coverage**: On all metric calculations
- **Mock Support**: Testing without API keys

### Changed
- **RagPipeline**: Added optional `observers` parameter for metrics collection
- **Package Exports**: Added all evaluation protocols and metrics to `__all__`
- **README.md**: Updated roadmap to mark v0.3.0 as COMPLETED ✅

### Fixed
- **Mermaid Graphs**: Fixed rendering in documentation by adding Mermaid.js library
- **Documentation Build**: Created missing evaluation.md example file
- **LLM Judge**: Fixed negative number parsing in score extraction

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
