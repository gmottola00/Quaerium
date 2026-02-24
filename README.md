# 🏛️ Quaerium

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-27%20passed-success.svg)](https://github.com/gmottola00/quaerium)
[![Coverage](https://img.shields.io/badge/coverage-76%25-brightgreen.svg)](https://github.com/gmottola00/quaerium)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://gmottola00.github.io/quaerium/)

> **Advanced RAG framework with multi-store support (vector + graph) and protocol-based architecture**

*"Quaerere" (Latin): to seek, to search, to inquire - The art of intelligent knowledge retrieval.*

Build production-grade Retrieval-Augmented Generation (RAG) systems with a clean, Protocol-based architecture that seamlessly combines vector search and knowledge graphs.

---

## ✨ Features

### 🧩 Protocol-Based Architecture
- **Zero inheritance required** - duck typing with type safety
- **Swap implementations easily** - change vector stores without rewriting code
- **Test-friendly** - mock any component with simple classes

### 🔌 Multi-Store Support
- **LLMs**: Ollama, OpenAI (more coming soon)
- **Embeddings**: Ollama (nomic-embed-text), OpenAI (text-embedding-3-small/large)
- **Vector Stores**: Milvus, Qdrant, ChromaDB
- **Graph Stores**: Neo4j 5.x (async, production-ready) 🆕

### 🕸️ Graph RAG (NEW!)
- **Knowledge Graphs**: Native Neo4j integration for structured knowledge
- **Hybrid Retrieval**: Combine vector similarity with graph traversal
- **Entity Relationships**: Model documents, chunks, and entities with rich connections
- **Cypher Queries**: Full power of Neo4j's query language
- **APOC Support**: Advanced graph algorithms and procedures

### 📄 Document Processing
- **Parsers**: PDF (PyMuPDF), DOCX, plain text
- **OCR**: Optional EasyOCR integration
- **Language Detection**: Automatic language identification

### ✂️ Smart Chunking
- **Dynamic chunking**: Heading-based document structure preservation
- **Token-based chunking**: Fixed-size chunks with overlap
- **Metadata-rich**: Automatic section paths, page numbers, hierarchy

### 🔍 Advanced RAG Pipeline
- **Query rewriting**: LLM-powered query optimization
- **Hybrid search**: Vector + keyword search combination
- **Reranking**: LLM-based result reordering
- **Context assembly**: Intelligent context window management
- **Citation tracking**: Full provenance of generated answers

---

## 🚀 Quick Start

### Installation

```bash
# From PyPI (coming soon)
pip install quaerium

# From GitHub (current release)
pip install git+https://github.com/gmottola00/quaerium.git@v0.1.0

# For development (editable install)
git clone https://github.com/gmottola00/quaerium.git
cd quaerium
pip install -e ".[dev]"
```

#### Optional Dependencies

```bash
# With Ollama support
pip install quaerium[ollama]

# With OpenAI support
pip install quaerium[openai]

# With document parsing (PDF, DOCX)
pip install quaerium[pdf,docx]

# All features
pip install quaerium[all]
```

### Basic Usage

```python
from quaerium import RagPipeline, get_ollama_embedding, get_ollama_llm
from quaerium.infra.vectorstore.milvus import MilvusVectorStore

# 1. Initialize components
embedding = get_ollama_embedding()(model="nomic-embed-text")
llm = get_ollama_llm()(model="llama3.2")
vectorstore = MilvusVectorStore(host="localhost", port=19530)

# 2. Create collection
vectorstore.create_collection(
    name="my_docs",
    dimension=768,  # nomic-embed-text dimension
    metric="IP"
)

# 3. Index documents (example)
texts = ["Document 1 content...", "Document 2 content..."]
vectors = [embedding.embed(text) for text in texts]
vectorstore.insert(
    collection_name="my_docs",
    vectors=vectors,
    texts=texts,
    metadata=[{"source": "doc1"}, {"source": "doc2"}]
)

# 4. Build RAG pipeline
pipeline = RagPipeline(
    vectorstore=vectorstore,
    embedding=embedding,
    llm=llm,
    collection_name="my_docs"
)

# 5. Ask questions!
response = pipeline.run("What is mentioned in the documents?")
print(response.answer)

# Access citations
for citation in response.citations:
    print(f"- {citation.text[:100]}... (score: {citation.score:.2f})")
```

---

## 📚 Core Concepts

### Protocol-Based Design

All core interfaces are defined as Protocols, not abstract base classes:

```python
from quaerium.core import EmbeddingClient, LLMClient, VectorStoreClient

# Any class implementing these methods works automatically
class MyCustomEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 384  # Your implementation
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]
    
    @property
    def model_name(self) -> str:
        return "my-model"

# Works with any quaerium component!
embedding: EmbeddingClient = MyCustomEmbedding()
```

### Clean Layered Architecture

```
quaerium/
├── core/          # Protocols only (zero dependencies)
│   ├── chunking.py
│   ├── embedding.py
│   ├── llm.py
│   └── vectorstore.py
│
├── infra/         # Concrete implementations
│   ├── embedding/
│   │   ├── ollama.py
│   │   └── openai.py
│   ├── llm/
│   ├── vectorstore/
│   └── parsers/
│
└── rag/           # RAG orchestration
    ├── pipeline.py
    ├── rewriter.py
    ├── reranker.py
    └── assembler.py
```

---

## 🎯 Advanced Usage

### Custom Vector Store

Implement the `VectorStoreClient` Protocol:

```python
from quaerium.core import VectorStoreClient, SearchResult

class PineconeVectorStore:
    def create_collection(self, name: str, dimension: int, **kwargs):
        # Your implementation
        pass
    
    def insert(self, collection_name: str, vectors, texts, metadata, **kwargs):
        # Your implementation
        return ["id1", "id2", ...]
    
    def search(self, collection_name: str, query_vector, top_k=10, **kwargs):
        # Your implementation
        return [SearchResult(...), ...]

# Works seamlessly with RagPipeline!
store: VectorStoreClient = PineconeVectorStore()
```

### Hybrid Search

```python
results = vectorstore.hybrid_search(
    collection_name="docs",
    query_vector=embedding.embed("installation guide"),
    query_text="installation guide",
    top_k=10,
    alpha=0.7  # 0=keyword only, 1=vector only
)
```

### Custom RAG Pipeline

```python
from quaerium.rag import (
    QueryRewriter,
    LLMReranker,
    ContextAssembler
)

# Build custom pipeline
pipeline = RagPipeline(
    vector_searcher=my_searcher,
    rewriter=QueryRewriter(llm),
    reranker=LLMReranker(llm, max_context=2000),
    assembler=ContextAssembler(max_tokens=4000),
    generator_llm=llm
)

# Add metadata hints
response = pipeline.run(
    "What is the pricing?",
    metadata_hint={"section": "pricing"},
    top_k=5
)
```

---

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=quaerium --cov-report=html

# Type checking
mypy src/quaerium

# Linting
ruff check src/quaerium
black src/quaerium --check
```

### Performance Benchmarks

```bash
# Run vector store benchmarks
make benchmark

# Generate HTML report
make benchmark-report
```

See [tests/benchmarks/README.md](tests/benchmarks/README.md) for detailed benchmark documentation.

---

## 📖 Documentation

Full documentation available at: [quaerium.readthedocs.io](https://quaerium.readthedocs.io) _(coming soon)_

---

## 🗺️ Roadmap

### v0.2.0 - Multi VectorStore
- [ ] Pinecone implementation
- [ ] Qdrant implementation
- [ ] Weaviate implementation

### v0.3.0 - Enhanced RAG
- [ ] Cross-encoder rerankers (Cohere, Jina)
- [ ] Multi-query retrieval
- [ ] Hypothetical document embeddings (HyDE)
- [ ] Parent-child chunking

### v0.4.0 - Production Features
- [ ] Async support throughout
- [ ] Distributed indexing
- [ ] Cost tracking
- [ ] Performance metrics
- [ ] OpenTelemetry integration

---

## 📚 Documentation

Comprehensive documentation is available at **[gmottola00.github.io/quaerium](https://gmottola00.github.io/quaerium/)**

### Quick Links

- **[Getting Started](https://gmottola00.github.io/quaerium/getting_started/installation.html)** - Installation and quickstart
- **[User Guide](https://gmottola00.github.io/quaerium/user_guide/core_concepts.html)** - Core concepts and protocols
- **[Examples](https://gmottola00.github.io/quaerium/examples/basic_rag.html)** - Complete working examples
- **[API Reference](https://gmottola00.github.io/quaerium/autoapi/index.html)** - Full API documentation

### Build Docs Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve
cd docs
make html
make serve  # Visit http://localhost:8000
```

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Setup development environment
git clone https://github.com/gmottola00/quaerium.git
cd quaerium
pip install -e ".[dev]"

# Run checks
pytest
ruff check .
mypy src/quaerium
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with inspiration from:
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Haystack](https://github.com/deepset-ai/haystack)

---

## 📞 Contact

- **Author**: Gianmarco Mottola
- **GitHub**: [@gmottola00](https://github.com/gmottola00)
- **Issues**: [GitHub Issues](https://github.com/gmottola00/quaerium/issues)

---

**Made with ❤️ for the RAG community**
