# 🚀 Rag-Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-28%20passed-success.svg)](https://github.com/gmottola00/rag-toolkit)
[![Coverage](https://img.shields.io/badge/coverage-19%25-yellow.svg)](https://github.com/gmottola00/rag-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://gmottola00.github.io/rag-toolkit/)

> **Advanced RAG library with multi-vectorstore support and production-ready components**

Build production-grade Retrieval-Augmented Generation (RAG) systems with a clean, Protocol-based architecture that makes it easy to switch between LLM providers, vector stores, and embedding models.

---

## ✨ Features

### 🧩 Protocol-Based Architecture
- **Zero inheritance required** - duck typing with type safety
- **Swap implementations easily** - change vector stores without rewriting code
- **Test-friendly** - mock any component with simple classes

### 🔌 Multi-Provider Support
- **LLMs**: Ollama, OpenAI (more coming soon)
- **Embeddings**: Ollama (nomic-embed-text), OpenAI (text-embedding-3-small/large)
- **Vector Stores**: Milvus (Pinecone, Qdrant, Weaviate coming soon)

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
pip install rag-toolkit

# From GitHub (current release)
pip install git+https://github.com/gmottola00/rag-toolkit.git@v0.1.0

# For development (editable install)
git clone https://github.com/gmottola00/rag-toolkit.git
cd rag-toolkit
pip install -e ".[dev]"
```

#### Optional Dependencies

```bash
# With Ollama support
pip install rag-toolkit[ollama]

# With OpenAI support
pip install rag-toolkit[openai]

# With document parsing (PDF, DOCX)
pip install rag-toolkit[pdf,docx]

# All features
pip install rag-toolkit[all]
```

### Basic Usage

```python
from rag_toolkit import RagPipeline, get_ollama_embedding, get_ollama_llm
from rag_toolkit.infra.vectorstore.milvus import MilvusVectorStore

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
from rag_toolkit.core import EmbeddingClient, LLMClient, VectorStoreClient

# Any class implementing these methods works automatically
class MyCustomEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 384  # Your implementation
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]
    
    @property
    def model_name(self) -> str:
        return "my-model"

# Works with any rag-toolkit component!
embedding: EmbeddingClient = MyCustomEmbedding()
```

### Clean Layered Architecture

```
rag-toolkit/
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
from rag_toolkit.core import VectorStoreClient, SearchResult

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
from rag_toolkit.rag import (
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
pytest --cov=rag_toolkit --cov-report=html

# Type checking
mypy src/rag_toolkit

# Linting
ruff check src/rag_toolkit
black src/rag_toolkit --check
```

---

## 📖 Documentation

Full documentation available at: [rag-toolkit.readthedocs.io](https://rag-toolkit.readthedocs.io) _(coming soon)_

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

Comprehensive documentation is available at **[gmottola00.github.io/rag-toolkit](https://gmottola00.github.io/rag-toolkit/)**

### Quick Links

- **[Getting Started](https://gmottola00.github.io/rag-toolkit/getting_started/installation.html)** - Installation and quickstart
- **[User Guide](https://gmottola00.github.io/rag-toolkit/user_guide/core_concepts.html)** - Core concepts and protocols
- **[Examples](https://gmottola00.github.io/rag-toolkit/examples/basic_rag.html)** - Complete working examples
- **[API Reference](https://gmottola00.github.io/rag-toolkit/autoapi/index.html)** - Full API documentation

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
git clone https://github.com/gmottola00/rag-toolkit.git
cd rag-toolkit
pip install -e ".[dev]"

# Run checks
pytest
ruff check .
mypy src/rag_toolkit
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
- **Issues**: [GitHub Issues](https://github.com/gmottola00/rag-toolkit/issues)

---

**Made with ❤️ for the RAG community**
