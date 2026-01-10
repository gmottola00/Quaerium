# :material-book-open: Basic RAG Example

!!! success "Your First RAG Application"
    Learn how to build a simple RAG application for document question-answering in minutes.

---

## :material-rocket-launch: Complete Example

!!! example "Full Working Code"
    Copy and run this complete example to see RAG in action.

```python title="basic_rag.py" linenums="1" hl_lines="19-22 28-33 38-43 48-62 65-70 75-82"
"""
Basic RAG Pipeline Example
===========================

This example demonstrates:
1. Setting up embedding and LLM clients
2. Creating a vector store
3. Indexing documents
4. Querying with context
"""

from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding.ollama import OllamaEmbedding
from rag_toolkit.infra.llm.ollama import OllamaLLMClient
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

def main():
    # Step 1: Initialize components
    print("🚀 Initializing components...")
    
    embedding = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    
    llm = OllamaLLMClient(
        base_url="http://localhost:11434",
        model="llama2",
    )
    
    vector_store = MilvusVectorStore(
        host="localhost",
        port="19530",
        collection_name="basic_rag_docs",
    )
    
    # Step 2: Create RAG pipeline
    pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=vector_store,
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # Step 3: Prepare documents
    documents = [
        """
        RAG (Retrieval-Augmented Generation) is a technique that enhances 
        large language models by retrieving relevant information from a 
        knowledge base before generating responses. This approach reduces 
        hallucinations and provides more accurate, contextual answers.
        """,
        """
        Vector stores are databases optimized for storing and searching 
        high-dimensional vectors. They enable semantic search by finding 
        vectors that are mathematically similar to a query vector, allowing 
        retrieval of semantically related content.
        """,
        """
        Embeddings are numerical representations of text that capture 
        semantic meaning. Similar texts have similar embedding vectors, 
        enabling machines to understand and compare text based on meaning 
        rather than just keywords.
        """,
        """
        Chunking is the process of breaking down large documents into 
        smaller, manageable pieces. This is important for RAG systems 
        because it allows more precise retrieval and helps manage context 
        window limitations of language models.
        """,
    ]
    
    # Step 4: Index documents
    print("\n📚 Indexing documents...")
    pipeline.index_documents(
        documents=documents,
        metadata=[
            {"topic": "RAG", "source": "tutorial"},
            {"topic": "Vector Stores", "source": "tutorial"},
            {"topic": "Embeddings", "source": "tutorial"},
            {"topic": "Chunking", "source": "tutorial"},
        ],
    )
    print("✅ Indexing complete!")
    
    # Step 5: Query the system
    queries = [
        "What is RAG and why is it useful?",
        "How do vector stores work?",
        "Explain the relationship between embeddings and semantic search",
    ]
    
    print("\n🔍 Running queries...\n")
    for query in queries:
        print(f"Q: {query}")
        response = pipeline.query(query, top_k=2)
        print(f"A: {response.answer}\n")
        print(f"Sources used: {len(response.sources)}")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. {source.text[:100]}...")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
```

---

## :material-stairs: Step-by-Step Breakdown

### :material-numeric-1-box: Component Setup

!!! info "Initialize Core Components"
    Set up the embedding model, LLM, and vector store.

=== "Embedding"

    ```python title="setup_embedding.py" linenums="1" hl_lines="2-5"
    # Embedding: Converts text to vectors
    embedding = OllamaEmbedding(
        model="nomic-embed-text",  # 768-dimensional embeddings
        base_url="http://localhost:11434"
    )
    ```

=== "LLM"

    ```python title="setup_llm.py" linenums="1" hl_lines="2-5"
    # LLM: Generates responses
    llm = OllamaLLMClient(
        model="llama2",  # or "mistral", "mixtral", etc.
        base_url="http://localhost:11434"
    )
    ```

=== "Vector Store"

    ```python title="setup_vectorstore.py" linenums="1" hl_lines="2-6"
    # Vector Store: Stores and searches embeddings
    vector_store = MilvusVectorStore(
        collection_name="my_documents",
        host="localhost",
        port="19530"
    )
    ```

---

### :material-numeric-2-box: Pipeline Creation

!!! tip "Configure Your Pipeline"
    Combine components into a unified RAG pipeline.

```python title="create_pipeline.py" linenums="1" hl_lines="2-7"
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
    chunk_size=512,        # Chunk documents into 512 chars
    chunk_overlap=50,      # 50 char overlap between chunks
)
```

---

### :material-numeric-3-box: Document Indexing

!!! example "Add Your Documents"
    Index documents with optional metadata for better organization.

```python title="index_documents.py" linenums="1" hl_lines="1-4 6-9 11-14"
documents = [
    "Your document text here...",
    "Another document...",
]

metadata = [
    {"source": "doc1.pdf", "page": 1},
    {"source": "doc2.pdf", "page": 1},
]

pipeline.index_documents(
    documents=documents,
    metadata=metadata,  # Optional but recommended
)
```

---

### :material-numeric-4-box: Querying

!!! success "Ask Questions"
    Query your indexed documents and get contextual answers.

```python title="query.py" linenums="1" hl_lines="2-4 7-9"
response = pipeline.query(
    "What is RAG?",
    top_k=5,  # Retrieve top 5 most relevant chunks
)

print(response.answer)      # Generated answer
print(response.sources)     # Retrieved chunks used
print(response.metadata)    # Additional info
```

---

## :material-file-pdf-box: PDF Documents

!!! tip "Process PDF Files"
    Extract text from PDFs and index them directly.

```python title="pdf_processing.py" linenums="1" hl_lines="1 4 5 8 9 12"
from rag_toolkit.infra.parsers.pdf import PDFParser

# Parse PDF
parser = PDFParser()
text = parser.parse("document.pdf")

# Index
pipeline.index_documents([text])

# Query
response = pipeline.query("What does the document say about...?")
```

---

## :material-folder-multiple: Multiple Collections

!!! info "Organize Documents"
    Separate documents into different collections for better organization.

```python title="multiple_collections.py" linenums="1" hl_lines="2-7 10-15 18-19 22-23"
# Technical docs collection
tech_store = MilvusVectorStore(collection_name="tech_docs")
tech_pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=tech_store,
)

# Marketing docs collection
marketing_store = MilvusVectorStore(collection_name="marketing_docs")
marketing_pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=marketing_store,
)

# Index separately
tech_pipeline.index_documents(technical_documents)
marketing_pipeline.index_documents(marketing_documents)

# Query specific collections
tech_response = tech_pipeline.query("How does the API work?")
marketing_response = marketing_pipeline.query("What's our value proposition?")
```

---

## :material-filter: Filtering by Metadata

!!! tip "Smart Filtering"
    Use metadata to filter search results and improve relevance.

```python title="metadata_filtering.py" linenums="1" hl_lines="2-8 11-14"
# Index with metadata
pipeline.index_documents(
    documents=documents,
    metadata=[
        {"category": "api", "version": "2.0"},
        {"category": "tutorial", "version": "2.0"},
        {"category": "api", "version": "1.0"},
    ],
)

# Query with filters
response = pipeline.query(
    "How to use the API?",
    filters={"category": "api", "version": "2.0"},
)
```

---

## :material-lightning-bolt: Async Support

!!! success "Concurrent Operations"
    Use async methods for better performance with multiple queries.

=== "Single Query"

    ```python title="async_single.py" linenums="1" hl_lines="3-5"
    import asyncio

    async def async_query_example():
        response = await pipeline.aquery("What is RAG?")
        return response
    ```

=== "Batch Queries"

    ```python title="async_batch.py" linenums="1" hl_lines="2-7 9-11"
    async def batch_query():
        queries = [
            "What is RAG?",
            "How do embeddings work?",
            "Explain vector stores",
        ]
        
        tasks = [pipeline.aquery(q) for q in queries]
        responses = await asyncio.gather(*tasks)
        
        return responses

    # Run
    responses = asyncio.run(batch_query())
    ```

---

## :material-alert-circle: Error Handling

!!! warning "Handle Errors Gracefully"
    Implement proper error handling for production applications.

```python title="error_handling.py" linenums="1" hl_lines="2-10"
try:
    response = pipeline.query("What is RAG?")
except ConnectionError as e:
    print(f"Failed to connect to services: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## :material-application: Complete Working Example

!!! example "Production-Ready Application"
    Save this as `my_rag_app.py` for a complete interactive RAG application.

```python title="my_rag_app.py" linenums="1" hl_lines="8-16 18-22 24-40 42-56"
#!/usr/bin/env python3
"""Complete RAG application."""

from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding.ollama import OllamaEmbedding
from rag_toolkit.infra.llm.ollama import OllamaLLMClient
from rag_toolkit.infra.vectorstores.milvus import MilvusVectorStore

def setup_pipeline():
    """Initialize and return configured pipeline."""
    embedding = OllamaEmbedding(model="nomic-embed-text")
    llm = OllamaLLMClient(model="llama2")
    store = MilvusVectorStore(collection_name="docs")
    
    return RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=store,
    )

def index_documents(pipeline, documents):
    """Index documents into the pipeline."""
    print("📚 Indexing documents...")
    pipeline.index_documents(documents)
    print("✅ Done!")

def interactive_query(pipeline):
    """Interactive query loop."""
    print("\n🤖 RAG Assistant Ready!")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            response = pipeline.query(query)
            print(f"\nAssistant: {response.answer}\n")
            
            if response.sources:
                print(f"📎 {len(response.sources)} sources used")
        
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    # Setup
    pipeline = setup_pipeline()
    
    # Index your documents
    documents = [
        "Your document content here...",
        # Add more documents
    ]
    index_documents(pipeline, documents)
    
    # Interactive queries
    interactive_query(pipeline)

if __name__ == "__main__":
    main()
```

!!! tip "Run the Application"
    ```bash
    python my_rag_app.py
    ```

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

-   :material-database-cog: **[Custom Vector Store](custom_vectorstore.md)**

    ---

    Use different databases for storage

-   :material-magnify-plus: **[Hybrid Search](hybrid_search.md)**

    ---

    Improve retrieval with hybrid search

-   :material-tune-vertical: **[Advanced Pipeline](advanced_pipeline.md)**

    ---

    Add production features

-   :material-server-network: **[Production Setup](production_setup.md)**

    ---

    Deploy to production

</div>
