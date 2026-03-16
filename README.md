# 🏛️ Quaerium

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-27%20passed-success.svg)](https://github.com/gmottola00/quaerium)
[![Coverage](https://img.shields.io/badge/coverage-76%25-brightgreen.svg)](https://github.com/gmottola00/quaerium)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://gmottola00.github.io/quaerium/)

> **Advanced RAG framework with multi-store support (vector + graph) and protocol-based architecture**

*"Quaerere" (Latin): to seek, to search, to inquire - The art of intelligent knowledge retrieval.*

---

## Features

**Architecture**
- Protocol-based design — zero inheritance required, duck typing with type safety
- Swap implementations without rewriting pipeline code
- Clean layered architecture: `core` (protocols) → `infra` (implementations) → `rag` (orchestration)

**Vector & Graph Stores**
- LLMs: Ollama, OpenAI
- Embeddings: Ollama (`nomic-embed-text`), OpenAI (`text-embedding-3-small/large`)
- Vector Stores: Milvus, Qdrant, ChromaDB
- Graph Stores: Neo4j 5.x (async, APOC support)

**Pipeline**
- Hybrid search: vector + keyword with configurable alpha blending
- Query rewriting, LLM-based reranking, context assembly
- Document parsers: PDF (PyMuPDF), DOCX, plain text; optional OCR via EasyOCR
- Dynamic and token-based chunking with metadata preservation

**Evaluation**
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG, Hit Rate
- Generation metrics: LLM-as-judge for relevance, faithfulness, hallucination detection
- End-to-end observability: latency, token usage, cost estimation
- Dataset management: JSONL format

---

## Installation

```bash
pip install quaerium
```

| Extra | Command |
|-------|---------|
| Ollama | `pip install quaerium[ollama]` |
| OpenAI | `pip install quaerium[openai]` |
| PDF / DOCX | `pip install quaerium[pdf,docx]` |
| All | `pip install quaerium[all]` |

---

## Quick Start

```python
from quaerium import RagPipeline, get_ollama_embedding, get_ollama_llm
from quaerium.infra.vectorstore.milvus import MilvusVectorStore

embedding = get_ollama_embedding()(model="nomic-embed-text")
llm = get_ollama_llm()(model="llama3.2")
vectorstore = MilvusVectorStore(host="localhost", port=19530)

vectorstore.create_collection(name="my_docs", dimension=768, metric="IP")

texts = ["Document 1 content...", "Document 2 content..."]
vectors = [embedding.embed(text) for text in texts]
vectorstore.insert(
    collection_name="my_docs",
    vectors=vectors,
    texts=texts,
    metadata=[{"source": "doc1"}, {"source": "doc2"}]
)

pipeline = RagPipeline(
    vectorstore=vectorstore,
    embedding=embedding,
    llm=llm,
    collection_name="my_docs"
)

response = pipeline.run("What is mentioned in the documents?")
print(response.answer)
for citation in response.citations:
    print(f"- {citation.text[:100]}... (score: {citation.score:.2f})")
```

---

## Advanced Usage

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

### Evaluation & Metrics

```python
from quaerium.infra.evaluation.pipeline import MetricsObserver
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator
from quaerium.infra.evaluation.generation import StandardGenerationEvaluator, OpenAIJudge

observer = MetricsObserver()
pipeline = RagPipeline(..., observers=[observer])
response = pipeline.run("What is RAG?")

result = observer.get_results()
print(f"Latency: {result.end_to_end_metrics.total_latency_ms:.0f}ms")
print(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")

retrieval_eval = StandardRetrievalEvaluator(k_values=[5, 10])
retrieval_metrics = retrieval_eval.evaluate_retrieval(
    query="What is RAG?",
    retrieved_docs=retrieved_docs,
    relevant_doc_ids=["doc1", "doc3"]
)
print(f"Precision@5: {retrieval_metrics.precision_at_k[5]:.2%}")

judge = OpenAIJudge(llm=llm)
generation_eval = StandardGenerationEvaluator(judge)
generation_metrics = generation_eval.evaluate_answer(
    question="What is RAG?",
    generated_answer=response.answer,
    context=[c.text for c in response.citations]
)
print(f"Relevance: {generation_metrics.relevance_score:.2%}")
print(f"Faithfulness: {generation_metrics.faithfulness_score:.2%}")
```

### Custom Vector Store

```python
from quaerium.core import VectorStoreClient, SearchResult

class PineconeVectorStore:
    def create_collection(self, name: str, dimension: int, **kwargs): ...
    def insert(self, collection_name: str, vectors, texts, metadata, **kwargs): ...
    def search(self, collection_name: str, query_vector, top_k=10, **kwargs): ...

store: VectorStoreClient = PineconeVectorStore()
```

---

## Documentation

Full documentation: [gmottola00.github.io/quaerium](https://gmottola00.github.io/quaerium/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
