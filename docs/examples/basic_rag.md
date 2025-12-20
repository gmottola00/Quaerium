# Basic RAG Example

This example shows how to build a simple RAG application for document question-answering.

## Complete Example

```python
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

## Step-by-Step Breakdown

### 1. Component Setup

```python
# Embedding: Converts text to vectors
embedding = OllamaEmbedding(
    model="nomic-embed-text",  # 768-dimensional embeddings
    base_url="http://localhost:11434"
)

# LLM: Generates responses
llm = OllamaLLMClient(
    model="llama2",  # or "mistral", "mixtral", etc.
    base_url="http://localhost:11434"
)

# Vector Store: Stores and searches embeddings
vector_store = MilvusVectorStore(
    collection_name="my_documents",
    host="localhost",
    port="19530"
)
```

### 2. Pipeline Creation

```python
pipeline = RagPipeline(
    embedding_client=embedding,
    llm_client=llm,
    vector_store=vector_store,
    chunk_size=512,        # Chunk documents into 512 chars
    chunk_overlap=50,      # 50 char overlap between chunks
)
```

### 3. Document Indexing

```python
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

### 4. Querying

```python
response = pipeline.query(
    "What is RAG?",
    top_k=5,  # Retrieve top 5 most relevant chunks
)

print(response.answer)      # Generated answer
print(response.sources)     # Retrieved chunks used
print(response.metadata)    # Additional info
```

## PDF Documents

Process PDF files:

```python
from rag_toolkit.infra.parsers.pdf import PDFParser

# Parse PDF
parser = PDFParser()
text = parser.parse("document.pdf")

# Index
pipeline.index_documents([text])

# Query
response = pipeline.query("What does the document say about...?")
```

## Multiple Collections

Organize documents in separate collections:

```python
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

## Filtering by Metadata

```python
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

## Async Support

For concurrent operations:

```python
import asyncio

async def async_query_example():
    # Use async LLM methods
    response = await pipeline.aquery("What is RAG?")
    return response

# Run multiple queries concurrently
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

## Error Handling

```python
try:
    response = pipeline.query("What is RAG?")
except ConnectionError as e:
    print(f"Failed to connect to services: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Complete Working Example

Save this as `my_rag_app.py`:

```python
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

Run it:
```bash
python my_rag_app.py
```

## Next Steps

- Try [Custom Vector Store](custom_vectorstore.md) for different databases
- Learn [Hybrid Search](hybrid_search.md) for better retrieval
- See [Advanced Pipeline](advanced_pipeline.md) for production features
- Read [Production Setup](production_setup.md) for deployment
