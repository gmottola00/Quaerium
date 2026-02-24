# :material-code-braces: Examples

!!! quote "Real-World Applications"
    Practical examples showing how to build production-ready RAG applications with quaerium.

---

## :material-star: Featured Examples

<div class="grid cards" markdown>

-   :material-book-open: **[Basic RAG](basic_rag.md)**

    ---

    Build your first RAG application with document indexing and querying.

    [:octicons-arrow-right-24: Get Started](basic_rag.md)

-   :material-database-cog: **[Custom Vector Store](custom_vectorstore.md)**

    ---

    Implement a custom vector store using ChromaDB or other databases.

    [:octicons-arrow-right-24: Learn More](custom_vectorstore.md)

-   :material-magnify-expand: **[Hybrid Search](hybrid_search.md)**

    ---

    Combine vector search with keyword search for better results.

    [:octicons-arrow-right-24: Explore](hybrid_search.md)

-   :material-tune-vertical: **[Advanced Pipeline](advanced_pipeline.md)**

    ---

    Build production-ready pipelines with reranking, query rewriting, and more.

    [:octicons-arrow-right-24: Deep Dive](advanced_pipeline.md)

-   :material-server-network: **[Production Setup](production_setup.md)**

    ---

    Deploy quaerium in production with monitoring, scaling, and best practices.

    [:octicons-arrow-right-24: Deploy](production_setup.md)

</div>

---

## :material-folder-open: Example Categories

=== "🎯 Getting Started"

    <div class="grid cards" markdown>

    -   :material-rocket-launch: **[Basic RAG](basic_rag.md)**

        ---

        Simple document Q&A to get started

    -   :material-file-pdf-box: **[PDF Processing](basic_rag.md#pdf-documents)**

        ---

        Parse and index PDF files

    -   :material-folder-multiple: **[Multiple Collections](basic_rag.md#multiple-collections)**

        ---

        Organize documents efficiently

    </div>

=== "🎨 Customization"

    <div class="grid cards" markdown>

    -   :material-database: **[Custom Vector Store](custom_vectorstore.md)**

        ---

        Implement your own storage backend

    -   :material-vector-polyline: **[Custom Embeddings](custom_vectorstore.md#custom-embeddings)**

        ---

        Use different embedding models

    -   :material-robot: **[Custom LLMs](custom_vectorstore.md#custom-llms)**

        ---

        Integrate new LLM providers

    </div>

=== "⚡ Advanced Features"

    <div class="grid cards" markdown>

    -   :material-magnify-plus-outline: **[Hybrid Search](hybrid_search.md)**

        ---

        Vector + keyword search combination

    -   :material-refresh: **[Query Rewriting](advanced_pipeline.md#query-rewriting)**

        ---

        Improve retrieval accuracy

    -   :material-sort-ascending: **[Reranking](advanced_pipeline.md#reranking)**

        ---

        Better result ordering

    </div>

=== "🚀 Production"

    <div class="grid cards" markdown>

    -   :material-monitor-dashboard: **[Monitoring](production_setup.md#monitoring)**

        ---

        Track performance metrics

    -   :material-chart-line: **[Scaling](production_setup.md#scaling)**

        ---

        Handle high traffic loads

    -   :material-cached: **[Caching](production_setup.md#caching)**

        ---

        Optimize response times

    </div>

---

## :material-run-fast: Running Examples

!!! tip "Quick Start"
    All examples are ready to run from the `examples/` directory.

=== "Clone & Install"

    ```bash title="setup.sh" linenums="1"
    # Clone the repository
    git clone https://github.com/gmottola00/quaerium.git
    cd quaerium

    # Install with all dependencies
    pip install -e ".[all]"
    ```

=== "Run Example"

    ```bash title="run_example.sh" linenums="1"
    # Run basic RAG example
    python examples/basic_rag.py

    # Run with custom configuration
    python examples/basic_rag.py --config my_config.yaml
    ```

=== "Jupyter Notebook"

    ```bash title="jupyter_setup.sh" linenums="1"
    # Install Jupyter
    pip install jupyter

    # Start notebook
    jupyter notebook examples/
    ```

---

## :material-help-circle: Need Help?

<div class="grid cards" markdown>

-   :material-book-open-variant: **[User Guide](../guides/index.md)**

    ---

    Comprehensive documentation and tutorials

-   :material-api: **[API Reference](../api/index.md)**

    ---

    Detailed API documentation

-   :material-forum: **[Discussions](https://github.com/gmottola00/quaerium/discussions)**

    ---

    Ask questions and share ideas

-   :material-bug: **[Report Issues](https://github.com/gmottola00/quaerium/issues)**

    ---

    Found a bug? Let us know!

</div>
