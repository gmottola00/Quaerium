# :material-tools: Tools

!!! quote "Production-Ready Utilities"
    Professional tools for migration, benchmarking, and performance optimization of your RAG applications.

---

## :material-star: Featured Tools

<div class="grid cards" markdown>

-   :material-swap-horizontal: **[Vector Store Migration](migration.md)**

    ---

    Transfer vector data seamlessly between Milvus, Qdrant, and ChromaDB with validation and progress tracking.

    [:octicons-arrow-right-24: Start Migrating](migration.md)

-   :material-speedometer: **[Performance Benchmarks](benchmarks.md)**

    ---

    Comprehensive benchmark suite to measure and compare vector store performance.

    [:octicons-arrow-right-24: Run Benchmarks](benchmarks.md)

-   :material-chart-line: **[Benchmark Results](benchmark_results.md)**

    ---

    Real-world performance comparison with detailed metrics and insights.

    [:octicons-arrow-right-24: View Results](benchmark_results.md)

</div>

---

## :material-folder-open: Tool Categories

=== "🔄 Migration"

    <div class="grid cards" markdown>

    -   :material-database-sync: **[Data Migration](migration.md)**

        ---

        Transfer vectors between different stores

    -   :material-filter-check: **[Filtered Migration](migration.md#filtered-migration)**

        ---

        Migrate subsets based on metadata

    -   :material-calculator: **[Migration Estimation](migration.md#estimation)**

        ---

        Plan migrations before execution

    -   :material-check-all: **[Data Validation](migration.md#validation)**

        ---

        Ensure integrity during transfer

    </div>

=== "📊 Benchmarking"

    <div class="grid cards" markdown>

    -   :material-upload: **[Insert Benchmarks](benchmarks.md#insert-benchmarks-9-tests)**

        ---

        Measure vector insertion performance

    -   :material-magnify: **[Search Benchmarks](benchmarks.md#search-benchmarks-9-tests)**

        ---

        Test similarity search speed

    -   :material-trending-up: **[Scale Tests](benchmarks.md#scale-benchmarks-6-tests)**

        ---

        Performance at scale (10K+ vectors)

    -   :material-chart-box: **[HTML Reports](benchmarks.md#generate-html-report)**

        ---

        Beautiful visual reports with charts

    </div>

=== "🎯 Performance"

    <div class="grid cards" markdown>

    -   :material-speedometer-medium: **[Store Comparison](benchmark_results.md#key-findings)**

        ---

        Compare Milvus, Qdrant, and ChromaDB

    -   :material-lightbulb: **[Best Practices](migration.md#best-practices)**

        ---

        Optimize batch sizes and retries

    -   :material-monitor-dashboard: **[Real Metrics](benchmark_results.md#insights)**

        ---

        Production-ready insights

    -   :material-tune: **[Configuration](benchmarks.md#benchmark-configuration)**

        ---

        Fine-tune benchmark settings

    </div>

---

## :material-rocket-launch: Quick Start

### :material-swap-horizontal: Migration Tool

!!! example "Migrate Between Vector Stores"
    Transfer data with automatic validation and progress tracking.

=== "Install"

    ```bash title="install.sh" linenums="1"
    # Install RAG Toolkit
    pip install quaerium

    # With all vector stores
    pip install quaerium[all]
    ```

=== "Basic Migration"

    ```python title="simple_migration.py" linenums="1"
    from quaerium.migration import VectorStoreMigrator
    from quaerium.infra.vectorstores import (
        get_chromadb_service,
        get_qdrant_service,
    )

    # Initialize stores
    source = get_chromadb_service(host="localhost")
    target = get_qdrant_service(host="localhost")

    # Create migrator
    migrator = VectorStoreMigrator(
        source=source,
        target=target,
        validate=True,
    )

    # Run migration
    result = migrator.migrate(
        source_collection="my_documents",
        target_collection="my_documents",
        batch_size=1000,
    )

    print(f"Success: {result.success}")
    print(f"Migrated: {result.vectors_migrated}")
    ```

=== "With Progress"

    ```python title="progress_migration.py" linenums="1"
    def on_progress(progress):
        print(
            f"Progress: {progress.percentage:.1f}% "
            f"ETA: {progress.eta_seconds:.0f}s"
        )

    migrator = VectorStoreMigrator(
        source=source,
        target=target,
        on_progress=on_progress,
    )

    result = migrator.migrate(
        source_collection="docs",
        batch_size=500,
    )
    ```

### :material-speedometer: Benchmark Suite

!!! example "Measure Performance"
    Run comprehensive benchmarks across all vector stores.

=== "Run All Tests"

    ```bash title="benchmark.sh" linenums="1"
    # Start vector stores
    docker-compose up -d

    # Run complete benchmark suite
    make benchmark

    # Generate HTML report
    make benchmark-report
    ```

=== "Specific Category"

    ```bash title="category_benchmark.sh" linenums="1"
    # Run only insert benchmarks
    pytest tests/benchmarks/test_insert_benchmark.py -v

    # Run only search benchmarks
    pytest tests/benchmarks/test_search_benchmark.py -v

    # Run scale tests
    pytest tests/benchmarks/test_scale_benchmark.py -v
    ```

=== "View Results"

    ```bash title="view_results.sh" linenums="1"
    # Open interactive HTML report
    open benchmark_report.html

    # Or view in docs
    mkdocs serve
    # Navigate to Tools > Benchmark Results
    ```

---

## :material-lightbulb: Key Features

### :material-database-sync: Vector Store Migration

<div class="grid cards" markdown>

-   :material-check-circle: **Validated Transfers**

    ---

    Automatic data integrity verification

-   :material-progress-check: **Real-Time Progress**

    ---

    Track migration with ETA calculation

-   :material-restart: **Retry Logic**

    ---

    Exponential backoff for transient failures

-   :material-test-tube: **Dry-Run Mode**

    ---

    Test migrations without writing data

-   :material-filter: **Metadata Filtering**

    ---

    Migrate only relevant subsets

-   :material-clock-fast: **Estimation**

    ---

    Predict time and resource requirements

</div>

### :material-chart-bar: Performance Benchmarks

<div class="grid cards" markdown>

-   :material-test-tube: **30 Benchmark Tests**

    ---

    Comprehensive test coverage

-   :material-package-variant: **Automatic Batching**

    ---

    Handle large datasets effortlessly

-   :material-chart-line: **Visual Reports**

    ---

    Interactive HTML with Chart.js

-   :material-compare: **Store Comparison**

    ---

    Side-by-side performance metrics

-   :material-cog: **Configurable**

    ---

    Adjust iterations and parameters

-   :material-docker: **Docker Integration**

    ---

    Easy setup with docker-compose

</div>

---

## :material-application-brackets: Common Use Cases

### :material-cloud-upload: Development to Production

Migrate from local development to production:

```python
# Dev: ChromaDB
dev_store = get_chromadb_service(host="localhost")

# Prod: Qdrant Cloud
prod_store = get_qdrant_service(
    host="production.qdrant.com",
    api_key=os.getenv("QDRANT_API_KEY"),
)

migrator = VectorStoreMigrator(
    source=dev_store,
    target=prod_store,
    validate=True,
)

result = migrator.migrate(
    source_collection="dev_documents",
    target_collection="prod_documents",
)
```

### :material-chart-box: Performance Testing

Compare vector stores for your workload:

```bash
# Start all stores
docker-compose up -d

# Run benchmarks
make benchmark

# Generate report
make benchmark-report

# View results
open benchmark_report.html
```

### :material-backup-restore: Backup & Restore

Create vector data backups:

```python
from datetime import datetime

# Backup
backup_migrator = VectorStoreMigrator(
    source=qdrant_service,
    target=chromadb_backup,
)

backup_result = backup_migrator.migrate(
    source_collection="critical_data",
    target_collection=f"backup_{datetime.now().isoformat()}",
)

# Restore later
restore_migrator = VectorStoreMigrator(
    source=chromadb_backup,
    target=qdrant_service,
)

restore_result = restore_migrator.migrate(
    source_collection="backup_2026_01_20",
    target_collection="critical_data_restored",
)
```

### :material-test-tube: A/B Testing

Test different vector stores:

```python
stores = [
    ("Milvus", get_milvus_service()),
    ("Qdrant", get_qdrant_service()),
    ("ChromaDB", get_chromadb_service()),
]

for name, store in stores:
    migrator = VectorStoreMigrator(source=baseline, target=store)

    result = migrator.migrate(
        source_collection="test_data",
        batch_size=1000,
    )

    print(f"{name}: {result.duration_seconds:.2f}s")
```

---

## :material-file-document: Documentation

<div class="grid cards" markdown>

-   :material-swap-horizontal: **[Migration Guide](migration.md)**

    ---

    Complete migration tool documentation

-   :material-speedometer: **[Benchmark Guide](benchmarks.md)**

    ---

    Running and configuring benchmarks

-   :material-chart-line: **[Results Analysis](benchmark_results.md)**

    ---

    Understanding benchmark metrics

</div>

---

## :material-help-circle: Need Help?

<div class="grid cards" markdown>

-   :material-book-open-variant: **[User Guide](../guides/index.md)**

    ---

    Comprehensive documentation and tutorials

-   :material-file-code: **[Examples](../examples/index.md)**

    ---

    Practical code examples

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
