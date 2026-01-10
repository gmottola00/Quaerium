# :material-speedometer: Performance Benchmarks

Comprehensive benchmark suite to measure and compare RAG Toolkit performance across different vector store implementations.

---

## :material-chart-line: Overview

!!! success "Benchmark Suite Features"
    
    <div class="grid cards" markdown>
    
    - :material-test-tube: **30 Benchmark Tests**
    
        ---
        
        Across 4 categories for comprehensive evaluation
    
    - :material-api: **Unified API**
    
        ---
        
        Through `VectorStoreWrapper` for consistent testing
    
    - :material-package-variant: **Automatic Batching**
    
        ---
        
        For large datasets without manual chunking
    
    - :material-chart-bar: **HTML Reports**
    
        ---
        
        With Chart.js visualizations and metrics
    
    - :material-compare: **Comparative Analysis**
    
        ---
        
        Across Milvus, Qdrant, and ChromaDB
    
    </div>

!!! info "View Latest Results"
    
    === "Interactive Report"
        [:material-chart-line: Full Benchmark Report](../_static/benchmark_report.html){ .md-button .md-button--primary }
        
        Complete benchmark data with interactive charts
    
    === "Landing Page"
        [:material-home: Benchmarks Home](../_static/benchmarks_index.html){ .md-button }
        
        Introduction and detailed instructions
    
    !!! tip "Local Viewing"
        For local viewing, open HTML files directly from `docs/_build/html/_static/`

---

## :material-folder-multiple: Benchmark Categories

!!! abstract "Four Test Categories"

### 1. :material-upload: Insert Benchmarks (9 tests)

!!! example "Vector Insertion Performance"
    Measure how fast each store can insert vectors.

**Test Scenarios:**

- :material-numeric-1-circle: Single insert (1 vector)
- :material-package: Batch insert (100 vectors)
- :material-database-arrow-up: Large batch insert (1,000 vectors)

**Tested on**: Milvus, Qdrant, ChromaDB

### 2. :material-magnify: Search Benchmarks (9 tests)

!!! example "Similarity Search Performance"
    Measure vector similarity search speed and accuracy.

**Test Scenarios:**

- :material-numeric-1-circle: Top-1 search (find closest match)
- :material-format-list-numbered: Top-10 search
- :material-format-list-bulleted: Top-100 search

**Tested on**: Milvus, Qdrant, ChromaDB

### 3. :material-sync: Batch Operations (6 tests)

!!! example "Complex Operation Performance"
    Measure performance of compound operations.

**Test Scenarios:**

- :material-delete-sweep: Bulk insert + delete cycles
- :material-sync-circle: Insert + search cycles

**Tested on**: Milvus, Qdrant, ChromaDB

### 4. :material-trending-up: Scale Benchmarks (6 tests)

!!! example "Performance at Scale"
    Measure how stores handle large-scale operations.

**Test Scenarios:**

- :material-database-plus: 10K vector insertion
- :material-magnify-expand: Search in large databases (10K+ vectors)

**Tested on**: Milvus, Qdrant, ChromaDB

---

## :material-play: Running Benchmarks

### :material-rocket-launch: Quick Start

=== "All Benchmarks"
    ```bash
    # Run complete benchmark suite
    make benchmark
    ```

=== "Specific Category"
    ```bash
    # Run only insert benchmarks
    pytest tests/benchmarks/test_insert_benchmark.py -v
    
    # Run only search benchmarks
    pytest tests/benchmarks/test_search_benchmark.py -v
    ```

=== "Custom Iterations"
    ```bash
    # Run with 10 iterations per test
    pytest tests/benchmarks/ --benchmark-min-rounds=10
    
    # Run with custom warmup
    pytest tests/benchmarks/ --benchmark-warmup=on
    ```

### :material-chart-box: Generate HTML Report

!!! success "Visual Reports"
    Generate beautiful HTML reports with interactive charts.

```bash title="Generate Report"
make benchmark-report
```

**Report Contents:**

<div class="grid cards" markdown>

- :material-chart-bar: **Summary Statistics**

    ---

    Min, max, mean, median for all tests

- :material-chart-line: **Comparative Charts**

    ---

    Bar charts comparing all vector stores

- :material-table: **Detailed Tables**

    ---

    Timing breakdowns with color coding

- :material-palette: **Color-Coded Stores**

    ---

    Easy visual identification

</div>

!!! tip "Viewing the Report"
    The report opens automatically in your browser, or manually open `benchmark_report.html`.

**Integration with Documentation:**

=== "Step 1: Generate"
    ```bash
    make benchmark-report
    ```

=== "Step 2: Copy to Docs"
    ```bash
    cp benchmark_report.html docs/_static/
    ```

=== "Step 3: Rebuild Docs"
    ```bash
    cd docs && make html
    ```

=== "Step 4: Access"
    Report available at: `<your-docs-url>/_static/benchmark_report.html`

!!! example "Add Link to Documentation"
    ```markdown
    View the latest [benchmark results](../_static/benchmark_report.html).
    ```

### :material-compare: Compare Results

```bash
# Compare current vs previous results
make benchmark-compare
```

### :material-delete: Clean Results

```bash
# Remove all benchmark data
make benchmark-clean
```

---

## :material-cog: Benchmark Configuration

### :material-docker: Environment Setup

!!! warning "Prerequisites"
    Ensure all vector stores are running before benchmarking.

=== "Milvus"
    ```bash
    # Start Milvus (port 19530)
    docker run -d --name milvus \
      -p 19530:19530 \
      milvusdb/milvus:latest standalone
    ```
    
    Check status:
    ```bash
    curl http://localhost:19530/healthz
    ```

=== "Qdrant"
    ```bash
    # Start Qdrant (port 6333)
    docker run -d --name qdrant \
      -p 6333:6333 \
      qdrant/qdrant:latest
    ```
    
    Check status:
    ```bash
    curl http://localhost:6333/health
    ```

=== "ChromaDB"
    ```bash
    # ChromaDB runs in-process
    # No Docker setup required
    pip install chromadb
    ```
    
    Runs automatically with sufficient RAM

### :material-language-python: Python Environment

```bash title="Install Dependencies"
# Install benchmark dependencies
pip install pytest-benchmark>=4.0.0

# Verify installation
pytest --version
```

---

## :material-cogs: Architecture

### :material-package: VectorStoreWrapper

!!! info "Unified API"
    Abstract API differences across vector stores.

```python title="wrapper_usage.py" linenums="1" hl_lines="1 4-7 10-13"
from tests.benchmarks.utils.wrapper import VectorStoreWrapper

# Initialize wrapper
wrapper = VectorStoreWrapper(
    service=milvus_service,
    collection_name="benchmark_collection",
    store_type="milvus"
)

# Unified API (works for all stores)
wrapper.add_vectors(data)
wrapper.search(query_vector, top_k=10)
wrapper.delete_vectors(ids)
wrapper.count()
```

---

### :material-layers: Automatic Batching

!!! tip "Smart Batching"
    Large operations are automatically batched to avoid payload limits.

```python title="auto_batching.py" linenums="1" hl_lines="2 3"
# 10K vectors automatically batched in chunks of 500
data = generator.generate_data(10000)
wrapper.add_vectors(data)  # Internally: 20 batches of 500
```

---

### :material-database: Data Generation

!!! example "Reproducible Test Data"
    Generate test data with configurable dimensions.

```python title="data_generation.py" linenums="1" hl_lines="1 3 6-8"
from tests.benchmarks.utils.data_generator import VectorDataGenerator

generator = VectorDataGenerator(dimension=384, seed=42)

# Generate store-specific formats
milvus_data = generator.generate_milvus_data(100)
qdrant_data = generator.generate_qdrant_points(100)
chroma_data = generator.generate_chroma_data(100)
```

---

## :material-chart-box: Interpreting Results

### :material-speedometer: Performance Metrics

!!! info "Benchmark Statistics"
    Each benchmark reports comprehensive timing statistics.

| Metric | Description |
|--------|-------------|
| **Min/Max/Mean** | Timing statistics |
| **StdDev** | Consistency measure |
| **Median** | Typical performance |
| **IQR** | Variability indicator |
| **OPS** | Operations per second |
| **Rounds/Iterations** | Test repetitions |

---

### :material-chart-line: Example Output

```bash title="Benchmark Results"
------------------------ benchmark 'insert': 6 tests -------------------------
Name (time in ms)                     Min       Max      Mean    StdDev    Median
----------------------------------------------------------------------------------
test_qdrant_single_insert          1.0323    4.4449    1.2977    0.4554    1.1683
test_qdrant_batch_insert_100      23.9068   68.0399   28.0110    8.4520   25.6845
test_milvus_single_insert      11067.0126 11151.9131 11094.9597   33.1737 11083.0722
```

---

### :material-compare: Performance Comparison

!!! success "Expected Relative Performance"

<div class="grid cards" markdown>

-   :material-upload: **Insert**

    ---

    Qdrant > ChromaDB > Milvus

-   :material-magnify: **Search**

    ---

    Milvus ≈ Qdrant > ChromaDB

-   :material-chart-areaspline: **Scale**

    ---

    Qdrant > Milvus > ChromaDB

</div>

!!! warning "Hardware Dependent"
    Actual performance depends on hardware, configuration, and data characteristics.

---

## :material-alert-octagon: Known Limitations

### :material-file-alert: Payload Size Limits

| Store | Limit | Benchmark Strategy |
|-------|-------|--------------------|
| **Qdrant** | 33MB per request | Automatic batching at 500 vectors |
| **Milvus** | No practical limit | Direct batch operations |
| **ChromaDB** | In-memory | Scales with available RAM |

---

### :material-sync: Flush Requirements

=== "Milvus"
    ```python
    # Requires explicit flush for immediate availability
    service.flush(collection_name)
    ```

=== "Qdrant"
    ```python
    # Automatic background indexing
    # No manual flush needed
    ```

=== "ChromaDB"
    ```python
    # Immediate consistency
    # Data available instantly
    ```

---

## :material-wrench: Troubleshooting

### :material-timer-sand: Tests Hanging

!!! failure "Connectivity Issues"
    Check vector store connectivity if tests hang.

=== "Milvus"
    ```bash
    # Test Milvus health
    curl http://localhost:19530/healthz
    ```

=== "Qdrant"
    ```bash
    # Test Qdrant health
    curl http://localhost:6333/health
    ```

=== "ChromaDB"
    ```bash
    # Check process memory
    ps aux | grep python
    ```

---

### :material-import: Import Errors

!!! warning "Environment Check"
    Ensure correct Python environment.

```bash title="verify_environment.sh"
# Check Python version (3.11-3.13)
python --version

# Verify dependencies
pip list | grep -E "pytest-benchmark|milvus|qdrant|chromadb"
```

---

### :material-speedometer-slow: Performance Issues

!!! tip "Optimization Steps"
    For slow benchmarks, try these solutions:

1. **Reduce Dataset Sizes** - Modify test parameters
2. **Use Faster Storage** - Switch to SSD
3. **Increase Docker Resources** - Allocate more CPU/memory
4. **Close Applications** - Free up system resources

---

## :material-road-variant: Future Enhancements

!!! info "Planned Features"
    Upcoming benchmark additions:

<div class="grid cards" markdown>

-   :material-magnify-plus: **Hybrid Search**

    ---

    Benchmark hybrid search performance

-   :material-account-group: **Concurrent Operations**

    ---

    Multi-threaded operation tests

-   :material-memory: **Memory Profiling**

    ---

    Track memory usage patterns

-   :material-wifi: **Network Simulation**

    ---

    Test with latency/packet loss

-   :material-server-network: **Scaling Tests**

    ---

    Multi-node performance

-   :material-currency-usd: **Cost Analysis**

    ---

    Per-operation cost tracking

</div>

---

## :material-hands-pray: Contributing

!!! success "Add New Benchmarks"
    Follow these steps to contribute:

1. **Create Test File** - Add to `tests/benchmarks/`
2. **Use Decorator** - `@pytest.mark.benchmark(group="category")`
3. **Follow Patterns** - Match existing test structure
4. **Update Documentation** - Add to this guide
5. **Submit PR** - Follow [Contributing Guide](contributing.md)

---

## :material-book-open: References

<div class="grid cards" markdown>

-   :material-flask: **pytest-benchmark**

    ---

    [Documentation](https://pytest-benchmark.readthedocs.io/)

-   :material-database-arrow-up: **Milvus**

    ---

    [Benchmark Guide](https://milvus.io/docs/benchmark.md)

-   :material-lightning-bolt: **Qdrant**

    ---

    [Performance Benchmarks](https://qdrant.tech/benchmarks/)

-   :material-palette: **ChromaDB**

    ---

    [Performance Guide](https://docs.trychroma.com/guides)

</div>
