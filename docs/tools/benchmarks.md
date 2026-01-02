# Performance Benchmarks

The RAG Toolkit includes a comprehensive benchmark suite to measure and compare performance across different vector store implementations.

```{note}
**View Latest Results**: 

- [Interactive Benchmark Report](../_static/benchmark_report.html) - Full benchmark data with charts
- [Benchmark Landing Page](../_static/benchmarks_index.html) - Introduction and instructions

*Note: For local viewing, you may need to open the HTML files directly from `docs/_build/html/_static/`*
```

## Overview

The benchmark suite provides:

- **30 benchmark tests** across 4 categories
- **Unified API** through `VectorStoreWrapper`
- **Automatic batching** for large datasets
- **HTML reports** with Chart.js visualizations
- **Comparative analysis** across Milvus, Qdrant, and ChromaDB

## Benchmark Categories

### 1. Insert Benchmarks (9 tests)
Measure vector insertion performance:
- Single insert (1 vector)
- Batch insert (100 vectors)
- Large batch insert (1,000 vectors)

**Tested for**: Milvus, Qdrant, ChromaDB

### 2. Search Benchmarks (9 tests)
Measure similarity search performance:
- Top-1 search
- Top-10 search
- Top-100 search

**Tested for**: Milvus, Qdrant, ChromaDB

### 3. Batch Operations (6 tests)
Measure complex operation performance:
- Bulk insert + delete cycles
- Insert + search cycles

**Tested for**: Milvus, Qdrant, ChromaDB

### 4. Scale Benchmarks (6 tests)
Measure performance at scale:
- 10K vector insertion
- Search in large databases (10K vectors)

**Tested for**: Milvus, Qdrant, ChromaDB

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
make benchmark

# Run specific category
pytest tests/benchmarks/test_insert_benchmark.py -v

# Run with custom iterations
pytest tests/benchmarks/ --benchmark-min-rounds=10
```

### Generate HTML Report

```bash
# Generate report from benchmark results
make benchmark-report
```

The report will be saved to `benchmark_report.html` at the project root with:
- Summary statistics
- Comparative bar charts
- Detailed timing tables
- Color-coded store identification

**Viewing the Report**: The report opens automatically in your browser, or you can manually open `benchmark_report.html`.

**Integration with Documentation**: To include the report in the Sphinx documentation:

1. Generate the report: `make benchmark-report`
2. Copy to docs: `cp benchmark_report.html docs/_static/`
3. Rebuild docs: `cd docs && make html`
4. The report will be available at: `<your-docs-url>/_static/benchmark_report.html`

Alternatively, add a link in your documentation:

```markdown
View the latest [benchmark results](../_static/benchmark_report.html).
```

### Compare Results

```bash
# Compare current vs previous results
make benchmark-compare
```

### Clean Results

```bash
# Remove all benchmark data
make benchmark-clean
```

## Benchmark Configuration

### Environment Setup

Ensure all vector stores are running:

```bash
# Milvus (port 19530)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# Qdrant (port 6333)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# ChromaDB (runs in-process)
# No setup required
```

### Python Environment

```bash
# Install benchmark dependencies
pip install pytest-benchmark>=4.0.0

# Verify installation
pytest --version
```

## Architecture

### VectorStoreWrapper

The benchmark suite uses a unified wrapper to abstract API differences:

```python
from tests.benchmarks.utils.wrapper import VectorStoreWrapper

# Initialize wrapper
wrapper = VectorStoreWrapper(
    service=milvus_service,
    collection_name="benchmark_collection",
    store_type="milvus"
)

# Unified API
wrapper.add_vectors(data)  # Works for all stores
wrapper.search(query_vector, top_k=10)
wrapper.delete_vectors(ids)
wrapper.count()
```

### Automatic Batching

Large operations are automatically batched to avoid payload size limits:

```python
# 10K vectors automatically batched in chunks of 500
data = generator.generate_data(10000)
wrapper.add_vectors(data)  # Internally: 20 batches of 500
```

### Data Generation

Reproducible test data with configurable dimensions:

```python
from tests.benchmarks.utils.data_generator import VectorDataGenerator

generator = VectorDataGenerator(dimension=384, seed=42)

# Generate store-specific formats
milvus_data = generator.generate_milvus_data(100)
qdrant_data = generator.generate_qdrant_points(100)
chroma_data = generator.generate_chroma_data(100)
```

## Interpreting Results

### Performance Metrics

Each benchmark reports:
- **Min/Max/Mean**: Timing statistics
- **StdDev**: Consistency measure
- **Median**: Typical performance
- **IQR**: Variability indicator
- **OPS**: Operations per second
- **Rounds/Iterations**: Test repetitions

### Example Output

```
-------------------------- benchmark 'insert': 6 tests --------------------------
Name (time in ms)                     Min       Max      Mean    StdDev    Median
---------------------------------------------------------------------------------
test_qdrant_single_insert          1.0323    4.4449    1.2977    0.4554    1.1683
test_qdrant_batch_insert_100      23.9068   68.0399   28.0110    8.4520   25.6845
test_milvus_single_insert      11067.0126 11151.9131 11094.9597   33.1737 11083.0722
```

### Performance Comparison

Expected relative performance:
1. **Insert**: Qdrant > ChromaDB > Milvus
2. **Search**: Milvus ≈ Qdrant > ChromaDB
3. **Scale**: Qdrant > Milvus > ChromaDB

*Note: Actual performance depends on hardware, configuration, and data characteristics.*

## Known Limitations

### Payload Size Limits

- **Qdrant**: 33MB per request → Automatic batching at 500 vectors
- **Milvus**: No practical limit for benchmarks
- **ChromaDB**: In-memory, scales with RAM

### Flush Requirements

- **Milvus**: Requires explicit flush for immediate availability
- **Qdrant**: Automatic background indexing
- **ChromaDB**: Immediate consistency

## Troubleshooting

### Tests Hanging

If tests hang, check vector store connectivity:

```bash
# Test Milvus
curl http://localhost:19530/healthz

# Test Qdrant
curl http://localhost:6333/health

# ChromaDB (check process memory)
ps aux | grep python
```

### Import Errors

Ensure correct Python environment:

```bash
# Check Python version (3.11-3.13)
python --version

# Verify dependencies
pip list | grep -E "pytest-benchmark|milvus|qdrant|chromadb"
```

### Performance Issues

For slow benchmarks:
1. Reduce dataset sizes in test files
2. Use faster storage (SSD)
3. Increase Docker resource limits
4. Close unnecessary applications

## Future Enhancements

Planned benchmark additions:
- [ ] Hybrid search benchmarks
- [ ] Concurrent operation tests
- [ ] Memory usage profiling
- [ ] Network latency simulation
- [ ] Multi-node scaling tests
- [ ] Cost analysis per operation

## Contributing

To add new benchmarks:

1. Create test file in `tests/benchmarks/`
2. Use `@pytest.mark.benchmark(group="category")` decorator
3. Follow existing test patterns
4. Update this documentation
5. Submit pull request

See [Contributing Guide](contributing.md) for details.

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Milvus benchmarks](https://milvus.io/docs/benchmark.md)
- [Qdrant benchmarks](https://qdrant.tech/benchmarks/)
- [ChromaDB performance](https://docs.trychroma.com/guides)
