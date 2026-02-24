# :material-rocket-launch: Quick Start

Build your first RAG application with RAG Toolkit in less than 5 minutes!

---

## :material-checkbox-marked-circle-outline: Prerequisites

!!! info "Before You Begin"
    Ensure you have the following installed:

<div class="grid cards" markdown>

- :material-language-python: **Python 3.11+**

    ---

    Download from [python.org](https://www.python.org/downloads/)

- :material-package-variant: **pip**

    ---

    Python package manager (included with Python)

- :material-docker: **Docker**

    ---

    For running Milvus vector store
    
    [Get Docker](https://www.docker.com/get-started)

</div>

---

## :material-numeric-1-box: Step 1: Installation

!!! success "Install with Ollama Support"
    ```bash
    pip install quaerium[ollama]
    ```

!!! tip "Alternative: OpenAI"
    If you prefer using OpenAI models:
    ```bash
    pip install quaerium[openai]
    ```

---

## :material-numeric-2-box: Step 2: Start Services

### :material-database: Start Milvus (Vector Store)

!!! abstract "Vector Database Setup"
    Milvus provides high-performance vector similarity search.

=== "Docker Compose"
    ```bash
    docker-compose up -d milvus
    ```

=== "Makefile"
    ```bash
    make docker-milvus
    ```

!!! success "Verify Milvus is Running"
    ```bash
    docker ps | grep milvus
    ```
    
    You should see the Milvus container running on port `19530`.

### :material-robot-outline: Start Ollama (Optional)

!!! info "Local LLM Server"
    Ollama allows you to run large language models locally.

=== "Install Ollama"
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

=== "Pull Models"
    ```bash
    # Language model for text generation
    ollama pull llama2
    
    # Embedding model for vector representation
    ollama pull nomic-embed-text
    ```

!!! tip "Verify Ollama"
    ```bash
    ollama list
    ```
    
    You should see `llama2` and `nomic-embed-text` in the list.

---

## :material-numeric-3-box: Step 3: Your First RAG Pipeline

!!! example "Create Your First Application"
    Create a file `my_first_rag.py`:

```python title="my_first_rag.py" linenums="1" hl_lines="7-9 13-15 19-21 26 36"
from quaerium import RagPipeline
from quaerium.infra.embedding.ollama import OllamaEmbedding
from quaerium.infra.llm.ollama import OllamaLLMClient
from quaerium.infra.vectorstores.milvus import MilvusVectorStore

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
    "quaerium is a Python library that makes it easy to build production-ready RAG applications.",
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

!!! success "Run Your Application"
    ```bash
    python my_first_rag.py
    ```

!!! quote "Expected Output"
    ```
    📚 Indexing documents...
    🔍 Querying the system...
    
    Answer: RAG stands for Retrieval-Augmented Generation, which is a 
    technique that combines information retrieval with text generation...
    
    Sources used: 2
      1. RAG stands for Retrieval-Augmented Generation. It's a technique...
      2. quaerium is a Python library that makes it easy to build...
    ```
---

## :material-numeric-4-box: Step 4: Understanding the Code

!!! info "Code Breakdown"
    Let's understand what each part does:

### 1. Component Initialization

```python hl_lines="2-4"
# Initialize embedding model
embedding = OllamaEmbedding(model="nomic-embed-text")
llm = OllamaLLMClient(model="llama2")
vector_store = MilvusVectorStore(collection_name="quickstart_docs")
```

We create three core components:

| Component | Purpose | Provider |
|-----------|---------|----------|
| **Embedding** | Converts text to vectors | Ollama (nomic-embed-text) |
| **LLM** | Generates natural language responses | Ollama (llama2) |
| **Vector Store** | Stores and retrieves embeddings | Milvus |

### 2. Pipeline Creation

```python
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
)
```

The `RagPipeline` orchestrates all components through **dependency injection**.

!!! tip "Protocol-Based Design"
    Any class implementing the protocol can be used—no inheritance required!

### 3. Document Indexing

```python
pipeline.index_documents(documents)
```

```mermaid
graph LR
    A[Documents] --> B[Split into Chunks]
    B --> C[Convert to Embeddings]
    C --> D[Store in Vector DB]
    style A fill:#e3f2fd
    style D fill:#c8e6c9
```

### 4. Querying

```python
response = pipeline.query("What is RAG?")
```

```mermaid
graph LR
    A[User Query] --> B[Embed Query]
    B --> C[Search Similar Docs]
    C --> D[Retrieve Context]
    D --> E[Generate Response]
    style A fill:#e3f2fd
    style E fill:#c8e6c9
```

---

## :material-tune: Next Steps

### :material-cog: Customize Your Pipeline

!!! example "Advanced Configuration"
    ```python
    # Use different models
    embedding = OllamaEmbedding(model="mxbai-embed-large")
    llm = OllamaLLMClient(model="mistral")
    
    # Configure retrieval parameters
    response = pipeline.query(
        "What is RAG?",
        top_k=5,        # Retrieve top 5 documents
        rerank=True,    # Use reranking for better results
    )
    
    # Access metadata and scores
    for source in response.sources:
        print(f"Relevance Score: {source.score:.4f}")
        print(f"Content: {source.text}")
    ```

### :material-file-document-multiple: Add Document Parsing

!!! example "Parse PDF Files"
    ```python
    from quaerium.infra.parsers.pdf import PDFParser
    
    # Parse PDF files
    parser = PDFParser()
    documents = parser.parse("path/to/document.pdf")
    
    # Index parsed content
    pipeline.index_documents(documents)
    ```

### :material-openai: Use OpenAI Instead

!!! example "Cloud-Based Models"
    ```python
    from quaerium.infra.embedding.openai_embedding import OpenAIEmbedding
    from quaerium.infra.llm.openai_llm import OpenAILLMClient
    
    embedding = OpenAIEmbedding(
        api_key="your-api-key",
        model="text-embedding-3-small"
    )
    
    llm = OpenAILLMClient(
        api_key="your-api-key",
        model="gpt-4"
    )
    ```

### :material-puzzle: Implement Custom Components

!!! success "Protocol-Based Flexibility"
    ```python
    # Your own vector store - no inheritance needed!
    class MyVectorStore:
        def create_collection(self, name, dimension, **kwargs): ...
        def insert(self, collection, vectors, texts, metadata): ...
        def search(self, collection, query_vector, top_k): ...
    
    # It just works! ✨
    pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=MyVectorStore(),  # Protocol-based design
    )
    ```

---

## :material-school: Common Patterns

### Pattern 1: Multi-Document RAG

!!! example "Multiple Data Sources"
    ```python
    # Index PDF documents
    pipeline.index_documents(
        documents=pdf_docs,
        metadata=[{"source": "manual.pdf", "type": "documentation"}]
    )
    
    # Index website content
    pipeline.index_documents(
        documents=web_docs,
        metadata=[{"source": "website", "type": "web"}]
    )
    
    # Query with filtering
    response = pipeline.query(
        "How to install?",
        filters={"source": "manual.pdf"}  # Only search in PDF
    )
    ```

### Pattern 2: Conversation Memory

!!! example "Contextual Conversations"
    ```python
    conversation = []
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        # Include conversation history
        context = "\n".join(conversation[-5:])  # Last 5 turns
        full_query = f"{context}\nUser: {query}"
        
        response = pipeline.query(full_query)
        print(f"Assistant: {response.answer}")
        
        # Update history
        conversation.append(f"User: {query}")
        conversation.append(f"Assistant: {response.answer}")
    ```

### Pattern 3: Batch Processing

!!! example "Process Multiple Queries"
    ```python
    queries = [
        "What is RAG?",
        "How do vector stores work?",
        "What are embeddings?",
    ]
    
    # Process all queries
    responses = [pipeline.query(q) for q in queries]
    
    # Display results
    for q, r in zip(queries, responses):
        print(f"Q: {q}")
        print(f"A: {r.answer}\n")
    ```

---

## :material-wrench: Troubleshooting

!!! bug "Common Issues & Solutions"

### :material-lan-disconnect: "Connection refused" Error

!!! failure "Problem"
    ```
    ConnectionRefusedError: [Errno 61] Connection refused
    ```

!!! success "Solution"
    Ensure Milvus is running:
    ```bash
    docker ps | grep milvus
    ```
    
    If not running, start it:
    ```bash
    docker-compose up -d milvus
    ```

### :material-file-question: "Model not found" Error

!!! failure "Problem"
    ```
    Error: model 'llama2' not found
    ```

!!! success "Solution"
    Pull the Ollama model:
    ```bash
    ollama pull llama2
    ollama pull nomic-embed-text
    ```
    
    Verify installation:
    ```bash
    ollama list
    ```

### :material-speedometer-slow: Slow Performance

!!! failure "Problem"
    Queries are taking too long to process.

!!! success "Solutions"
    
    === "Use Smaller Models"
        ```python
        # Smaller embedding model
        embedding = OllamaEmbedding(model="all-minilm")
        
        # Smaller LLM
        llm = OllamaLLMClient(model="tinyllama")
        ```
    
    === "Reduce top_k"
        ```python
        # Retrieve fewer documents
        response = pipeline.query("query", top_k=3)
        ```
    
    === "Enable Batch Processing"
        ```python
        # Process documents in batches
        vectors = embedding.embed_batch(texts)
        ```

---

## :material-book-open: Learn More

<div class="grid cards" markdown>

- :material-library: **Core Concepts**

    ---

    Understand the architecture and design principles

    [:material-arrow-right: Learn](../guides/core_concepts.md)

- :material-protocol: **Protocols Guide**

    ---

    Deep dive into protocol-based design

    [:material-arrow-right: Read](../guides/protocols.md)

- :material-application-brackets: **Examples**

    ---

    Explore real-world RAG applications

    [:material-arrow-right: Browse](../examples/index.md)

- :material-api: **API Reference**

    ---

    Complete API documentation

    [:material-arrow-right: Explore](../api/index.md)

</div>

---

## :material-lifebuoy: Getting Help

!!! question "Need Assistance?"

<div class="grid" markdown>

- :material-book-open-page-variant: **[Documentation](https://gmottola00.github.io/quaerium/)**  
  Comprehensive guides and tutorials

- :material-forum: **[GitHub Discussions](https://github.com/gmottola00/quaerium/discussions)**  
  Ask questions and share ideas

- :material-bug: **[Issue Tracker](https://github.com/gmottola00/quaerium/issues)**  
  Report bugs and request features

- :material-email: **[Email Support](mailto:gianmarcomottola00@gmail.com)**  
  Direct contact for enterprise inquiries

</div>

!!! tip "Quick Tips"
    - Start with the [installation guide](installation.md) if you haven't already
    - Check out [architecture overview](architecture.md) to understand the design
    - Browse [examples](../examples/index.md) for inspiration
    - Join our [community discussions](https://github.com/gmottola00/quaerium/discussions)
