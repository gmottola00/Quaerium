# Examples

Real-world examples showing how to use rag-toolkit for various use cases.

## Available Examples

::::{grid} 2
:gutter: 3

:::{grid-item-card} Basic RAG
:link: basic_rag
:link-type: doc

Build your first RAG application with document indexing and querying.
:::

:::{grid-item-card} Custom Vector Store
:link: custom_vectorstore
:link-type: doc

Implement a custom vector store using ChromaDB or other databases.
:::

:::{grid-item-card} Hybrid Search
:link: hybrid_search
:link-type: doc

Combine vector search with keyword search for better results.
:::

:::{grid-item-card} Advanced Pipeline
:link: advanced_pipeline
:link-type: doc

Build production-ready pipelines with reranking, query rewriting, and more.
:::

:::{grid-item-card} Production Setup
:link: production_setup
:link-type: doc

Deploy rag-toolkit in production with monitoring, scaling, and best practices.
:::

::::

## Example Categories

### Getting Started
- [Basic RAG](basic_rag.md) - Simple document Q&A
- [PDF Processing](basic_rag.md#pdf-documents) - Parse and index PDF files
- [Multiple Collections](basic_rag.md#multiple-collections) - Organize documents

### Customization
- [Custom Vector Store](custom_vectorstore.md) - Implement your own store
- [Custom Embeddings](custom_vectorstore.md#custom-embeddings) - Use different models
- [Custom LLMs](custom_vectorstore.md#custom-llms) - Integrate new providers

### Advanced Features
- [Hybrid Search](hybrid_search.md) - Vector + keyword search
- [Query Rewriting](advanced_pipeline.md#query-rewriting) - Improve retrieval
- [Reranking](advanced_pipeline.md#reranking) - Better result ordering

### Production
- [Monitoring](production_setup.md#monitoring) - Track performance
- [Scaling](production_setup.md#scaling) - Handle high traffic
- [Caching](production_setup.md#caching) - Optimize performance

## Quick Links

```{toctree}
:maxdepth: 2

basic_rag
custom_vectorstore
hybrid_search
advanced_pipeline
production_setup
```

## Running Examples

All examples are available in the `examples/` directory:

```bash
# Clone the repository
git clone https://github.com/gmottola00/rag-toolkit.git
cd rag-toolkit

# Install with examples dependencies
pip install -e ".[all]"

# Run an example
python examples/basic_rag.py
```

## Need Help?

- 📖 Check the [User Guide](../user_guide/index.md)
- 🔍 See [API Reference](../autoapi/index.html)
- 💬 Ask in [Discussions](https://github.com/gmottola00/rag-toolkit/discussions)
- 🐛 Report issues on [GitHub](https://github.com/gmottola00/rag-toolkit/issues)
