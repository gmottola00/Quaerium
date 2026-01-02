# Quick Start

Get up and running with rag-toolkit in less than 5 minutes!

## Prerequisites

Before starting, ensure you have:

- Python 3.11+ installed
- pip package manager
- Docker (for running Milvus)

## Step 1: Installation

Install rag-toolkit with Ollama support:

```bash
pip install rag-toolkit[ollama]
```

## Step 2: Start Services

### Start Milvus (Vector Store)

```bash
# Using docker-compose
docker-compose up -d milvus

# Or using the Makefile
make docker-milvus
```

### Start Ollama (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama2
ollama pull nomic-embed-text
```

## Step 3: Your First RAG Pipeline

Create a file `my_first_rag.py`:

```python
from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding.ollama import OllamaEmbedding
from rag_toolkit.infra.llm.ollama import OllamaLLMClient
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

# Initialize components
embedding = OllamaEmbedding(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)

llm = OllamaLLMClient(
    base_url="http://localhost:11434",
    model="llama2"
)

vector_store = MilvusVectorStore(
    host="localhost",
    port="19530",
    collection_name="quickstart_docs"
)

# Create RAG pipeline
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
)

# Index some documents
documents = [
    "RAG stands for Retrieval-Augmented Generation. It's a technique that combines information retrieval with text generation.",
    "Vector stores enable semantic search by storing embeddings of text and finding similar content.",
    "rag-toolkit is a Python library that makes it easy to build production-ready RAG applications.",
]

print("📚 Indexing documents...")
pipeline.index_documents(documents)

# Query the system
print("\n🔍 Querying the system...\n")
response = pipeline.query("What is RAG and how does it work?")

print(f"Answer: {response.answer}")
print(f"\nSources used: {len(response.sources)}")
for i, source in enumerate(response.sources, 1):
    print(f"  {i}. {source[:100]}...")
```

Run it:

```bash
python my_first_rag.py
```

## Step 4: Understanding the Code

Let's break down what's happening:

### 1. Component Initialization

```python
embedding = OllamaEmbedding(model="nomic-embed-text")
llm = OllamaLLMClient(model="llama2")
vector_store = MilvusVectorStore(collection_name="quickstart_docs")
```

We create three components:
- **Embedding**: Converts text to vectors
- **LLM**: Generates responses
- **Vector Store**: Stores and retrieves embeddings

### 2. Pipeline Creation

```python
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
)
```

The pipeline orchestrates all components.

### 3. Document Indexing

```python
pipeline.index_documents(documents)
```

Documents are:
1. Split into chunks
2. Converted to embeddings
3. Stored in the vector store

### 4. Querying

```python
response = pipeline.query("What is RAG?")
```

When you query:
1. Query is embedded
2. Similar documents are retrieved
3. LLM generates answer using context

## Next Steps

### Customize Your Pipeline

```python
# Use different models
embedding = OllamaEmbedding(model="mxbai-embed-large")
llm = OllamaLLMClient(model="mistral")

# Configure retrieval
response = pipeline.query(
    "What is RAG?",
    top_k=5,  # Retrieve top 5 documents
    rerank=True,  # Use reranking
)

# Access metadata
for source in response.sources:
    print(f"Score: {source.score}")
    print(f"Content: {source.text}")
```

### Add Document Parsing

```python
from rag_toolkit.infra.parsers.pdf import PDFParser

# Parse PDF files
parser = PDFParser()
documents = parser.parse("path/to/document.pdf")

# Index parsed content
pipeline.index_documents(documents)
```

### Use OpenAI Instead

```python
from rag_toolkit.infra.embedding.openai_embedding import OpenAIEmbedding
from rag_toolkit.infra.llm.openai_llm import OpenAILLMClient

embedding = OpenAIEmbedding(
    api_key="your-api-key",
    model="text-embedding-3-small"
)

llm = OpenAILLMClient(
    api_key="your-api-key",
    model="gpt-4"
)
```

### Implement Custom Components

```python
# Your own vector store
class MyVectorStore:
    def create_collection(self, name, dimension, **kwargs): ...
    def insert(self, collection, vectors, texts, metadata): ...
    def search(self, collection, query_vector, top_k): ...

# It just works! (Protocol-based design)
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=MyVectorStore(),  # ✅ No inheritance needed
)
```

## Common Patterns

### Pattern 1: Multi-Document RAG

```python
# Index multiple sources
pipeline.index_documents(
    documents=pdf_docs,
    metadata=[{"source": "manual.pdf"}]
)

pipeline.index_documents(
    documents=web_docs,
    metadata=[{"source": "website"}]
)

# Query with filtering
response = pipeline.query(
    "How to install?",
    filters={"source": "manual.pdf"}
)
```

### Pattern 2: Conversation Memory

```python
# Maintain conversation history
conversation = []

while True:
    query = input("You: ")
    
    # Include conversation context
    context = "\n".join(conversation[-5:])  # Last 5 turns
    full_query = f"{context}\nUser: {query}"
    
    response = pipeline.query(full_query)
    print(f"Assistant: {response.answer}")
    
    # Update history
    conversation.append(f"User: {query}")
    conversation.append(f"Assistant: {response.answer}")
```

### Pattern 3: Batch Processing

```python
queries = [
    "What is RAG?",
    "How do vector stores work?",
    "What are embeddings?",
]

responses = [pipeline.query(q) for q in queries]

for q, r in zip(queries, responses):
    print(f"Q: {q}")
    print(f"A: {r.answer}\n")
```

## Troubleshooting

### "Connection refused" Error

Ensure Milvus is running:
```bash
docker ps | grep milvus
```

### "Model not found" Error

Pull the Ollama model:
```bash
ollama pull llama2
```

### Slow Performance

Use smaller models or batch processing:
```python
# Smaller embedding model
embedding = OllamaEmbedding(model="all-minilm")

# Smaller LLM
llm = OllamaLLMClient(model="tinyllama")
```

## Learn More

- [Core Concepts](user_guide/core_concepts.md) - Understand the architecture
- [Protocols Guide](user_guide/protocols.md) - Learn about Protocol-based design
- [Examples](examples/index.md) - See real-world applications
- [API Reference](autoapi/index.html) - Complete API documentation

## Getting Help

- 📖 [Documentation](https://gmottola00.github.io/rag-toolkit/)
- 💬 [GitHub Discussions](https://github.com/gmottola00/rag-toolkit/discussions)
- 🐛 [Issue Tracker](https://github.com/gmottola00/rag-toolkit/issues)
- 📧 Email: gianmarcomottola00@gmail.com
