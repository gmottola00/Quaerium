"""
Quickstart Example: Basic RAG pipeline with Ollama + Milvus.

This example demonstrates the simplest way to get started with rag-toolkit.

Prerequisites:
    1. Milvus running: docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
    2. Ollama running: ollama serve
    3. Models pulled: ollama pull nomic-embed-text && ollama pull llama3.2

Usage:
    python examples/quickstart.py
"""

from __future__ import annotations

from src.rag_toolkit import get_ollama_embedding, get_ollama_llm
from src.rag_toolkit.infra.vectorstores.milvus import MilvusService
from src.rag_toolkit.rag import ContextAssembler, QueryRewriter, RagPipeline
from src.rag_toolkit.rag.rerankers import LLMReranker


def main() -> None:
    """Run the quickstart example."""
    print("🚀 rag-toolkit Quickstart\n")
    
    # ========================================================================
    # Step 1: Initialize components
    # ========================================================================
    print("📦 Initializing components...")
    
    # Embedding model (converts text to vectors)
    OllamaEmbedding = get_ollama_embedding()
    embedding = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="nomic-embed-text"
    )
    print(f"  ✓ Embedding: {embedding.model_name} (dim: {embedding.dimension})")
    
    # LLM for generation
    OllamaLLM = get_ollama_llm()
    llm = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3.2"
    )
    print(f"  ✓ LLM: {llm.model_name}")
    
    # Vector store (Milvus)
    vectorstore = MilvusService(
        host="localhost",
        port=19530,
        collection_name="quickstart_docs"
    )
    print(f"  ✓ Vector Store: Milvus (collection: quickstart_docs)\n")
    
    # ========================================================================
    # Step 2: Create collection and index sample documents
    # ========================================================================
    print("📚 Indexing sample documents...")
    
    # Sample documents about RAG
    documents = [
        {
            "text": "RAG (Retrieval-Augmented Generation) is a technique that combines "
                   "information retrieval with large language models. It retrieves relevant "
                   "documents from a knowledge base and uses them as context for generation.",
            "metadata": {"source": "intro", "topic": "rag-basics"}
        },
        {
            "text": "Vector databases store embeddings (numerical representations of text) "
                   "and enable fast similarity search. Popular vector databases include "
                   "Milvus, Pinecone, Qdrant, and Weaviate.",
            "metadata": {"source": "intro", "topic": "vector-db"}
        },
        {
            "text": "Chunking is the process of splitting documents into smaller pieces. "
                   "Good chunking strategies preserve semantic coherence and include "
                   "metadata like section paths and page numbers.",
            "metadata": {"source": "guide", "topic": "chunking"}
        },
        {
            "text": "Embeddings convert text into dense vectors where similar texts have "
                   "similar vectors. Common models include OpenAI's text-embedding-3 and "
                   "Ollama's nomic-embed-text.",
            "metadata": {"source": "guide", "topic": "embeddings"}
        },
        {
            "text": "Hybrid search combines vector similarity search with keyword search "
                   "to improve recall. The alpha parameter controls the balance between "
                   "semantic and keyword matching.",
            "metadata": {"source": "advanced", "topic": "search"}
        }
    ]
    
    # Create collection
    try:
        vectorstore.drop_collection("quickstart_docs")
    except Exception:
        pass
    
    # TODO: Replace with proper create_collection once Milvus adapter is complete
    print("  ⚠️  Note: Using simplified indexing for demo\n")
    
    # ========================================================================
    # Step 3: Build RAG pipeline
    # ========================================================================
    print("🔧 Building RAG pipeline...")
    
    # Create search strategy (simplified for demo)
    from quaerium.core.index.search_strategies import VectorSearch
    
    # Mock indexer for demo
    class MockIndexer:
        def __init__(self, vectorstore, embedding):
            self.vectorstore = vectorstore
            self.embedding = embedding
    
    indexer = MockIndexer(vectorstore, embedding)
    
    vector_search = VectorSearch(
        index_service=indexer,
        embed_fn=lambda q: embedding.embed(q)
    )
    
    # Build complete pipeline
    pipeline = RagPipeline(
        vector_searcher=vector_search,
        rewriter=QueryRewriter(llm),
        reranker=LLMReranker(llm, max_context=2000),
        assembler=ContextAssembler(max_tokens=2000),
        generator_llm=llm
    )
    print("  ✓ Pipeline ready\n")
    
    # ========================================================================
    # Step 4: Ask questions!
    # ========================================================================
    print("💬 RAG in Action\n")
    print("=" * 60)
    
    questions = [
        "What is RAG?",
        "What are vector databases?",
        "How does hybrid search work?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 60)
        
        try:
            response = pipeline.run(question, top_k=3)
            
            print(f"💡 Answer: {response.answer}\n")
            
            print(f"📎 Citations ({len(response.citations)}):")
            for j, citation in enumerate(response.citations, 1):
                print(f"  [{j}] {citation.text[:80]}...")
                print(f"      Score: {citation.score:.4f}")
                if citation.metadata:
                    print(f"      Source: {citation.metadata.get('source', 'unknown')}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("\n✅ Quickstart complete!")
    print("\n📚 Next steps:")
    print("  - Explore examples/advanced_rag.py for more features")
    print("  - Read the docs at https://rag-toolkit.readthedocs.io")
    print("  - Check out the GitHub repo: https://github.com/gmottola00/rag-toolkit")


if __name__ == "__main__":
    main()
