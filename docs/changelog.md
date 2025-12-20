# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Pinecone vector store implementation
- Qdrant vector store implementation
- Cross-encoder rerankers (Cohere, Jina)
- Parent-child chunking strategy
- Query decomposition
- Multi-query retrieval
- Conversation memory management

## [0.1.0] - 2024-12-20

### Added
- Initial release of rag-toolkit
- Protocol-based architecture with `EmbeddingClient`, `LLMClient`, `VectorStoreClient`
- Ollama integration for embeddings and LLMs
- OpenAI integration for embeddings and LLMs
- Milvus vector store implementation
- Basic RAG pipeline with document indexing and querying
- Document chunking with fixed-size and token-aware strategies
- PDF and DOCX document parsers
- Query rewriting (HyDE, multi-query)
- Context assembler for retrieved chunks
- Comprehensive type hints throughout
- Full documentation with Sphinx + Furo theme
- GitHub Actions CI/CD workflows
- Example applications and tutorials

### Core Features
- **Protocol-Based Design**: No inheritance required, duck typing with type safety
- **Multi-Provider Support**: Easy integration of different LLM and embedding providers
- **Flexible Vector Stores**: Pluggable vector store interface
- **Modular Installation**: Optional dependencies for different providers
- **Production Ready**: Type hints, tests, and best practices

### Documentation
- Complete user guide covering all core concepts
- Protocol implementation guide
- API reference with sphinx-autoapi
- Multiple working examples
- Installation and quickstart guides
- Architecture overview

### Development
- Professional package structure with pyproject.toml
- Development tools: pytest, mypy, ruff, black, isort
- Pre-commit hooks configuration
- Comprehensive Makefile for common tasks
- Docker setup for Milvus

[Unreleased]: https://github.com/gmottola00/rag-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gmottola00/rag-toolkit/releases/tag/v0.1.0
