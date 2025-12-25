rag-toolkit Documentation
=========================

.. image:: https://img.shields.io/pypi/v/rag-toolkit.svg
   :target: https://pypi.org/project/rag-toolkit/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/rag-toolkit.svg
   :target: https://pypi.org/project/rag-toolkit/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/gmottola00/rag-toolkit.svg
   :target: https://github.com/gmottola00/rag-toolkit/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

**rag-toolkit** is a professional, production-ready RAG (Retrieval-Augmented Generation) library with multi-vectorstore support and Protocol-based architecture.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Quick Start
        :link: quickstart
        :link-type: doc

        Get started with rag-toolkit in minutes with our comprehensive quickstart guide.

    .. grid-item-card:: 📚 User Guide
        :link: user_guide/index
        :link-type: doc

        Learn core concepts, protocols, and best practices for building RAG applications.

    .. grid-item-card:: 🔌 API Reference
        :link: autoapi/index
        :link-type: doc

        Complete API documentation with all classes, functions, and protocols.

    .. grid-item-card:: 💡 Examples
        :link: examples/index
        :link-type: doc

        Real-world examples showing how to build RAG applications with rag-toolkit.

    .. grid-item-card:: ⚡ Benchmarks
        :link: benchmarks
        :link-type: doc

        Performance benchmarks comparing Milvus, Qdrant, and ChromaDB implementations.

Features
--------

✨ **Protocol-Based Architecture**
   Clean abstractions using Python Protocols (duck typing, no inheritance required)

🗄️ **Multi-VectorStore Support**
   Unified interface for Milvus, with Pinecone and Qdrant coming soon

🔌 **Multiple LLM Providers**
   Built-in support for Ollama and OpenAI with easy extensibility

🧩 **Modular Installation**
   Optional dependencies let you install only what you need

📦 **Production Ready**
   Type hints, comprehensive tests, and professional code quality

🧪 **Test-Friendly Design**
   Easy mocking and dependency injection for testing

Quick Example
-------------

.. code-block:: python

   from rag_toolkit import RagPipeline
   from rag_toolkit.core import EmbeddingClient, LLMClient, VectorStoreClient

   # Initialize components
   embedding = get_ollama_embedding(model="nomic-embed-text")
   llm = get_ollama_llm(model="llama2")
   vector_store = MilvusVectorStore(collection_name="my_docs")

   # Create RAG pipeline
   pipeline = RagPipeline(
       embedding_client=embedding,
       llm_client=llm,
       vector_store=vector_store,
   )

   # Index your documents
   pipeline.index_documents([
       "RAG combines retrieval and generation.",
       "Vector stores enable semantic search.",
   ])

   # Query with context
   response = pipeline.query("What is RAG?")
   print(response.answer)
   print(f"Sources: {response.sources}")

Why rag-toolkit?
----------------

**Clean Architecture**
   Protocol-based design means no inheritance requirements. Any class matching the protocol signature works seamlessly.

**Production Ready**
   Built with best practices: type hints, comprehensive docstrings, modular design, and extensive testing.

**Extensible**
   Add new vector stores, LLM providers, or embedding models without touching core code.

**Developer Friendly**
   Clear documentation, working examples, and intuitive APIs make development fast and enjoyable.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   architecture

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/core_concepts
   user_guide/protocols
   user_guide/vector_stores
   user_guide/embeddings
   user_guide/llms
   user_guide/rag_pipeline
   user_guide/chunking
   user_guide/reranking

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_rag
   examples/custom_vectorstore
   examples/hybrid_search
   examples/advanced_pipeline
   examples/production_setup

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   benchmarks
   benchmark_results
   changelog
   roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
