# Vector Store Performance Benchmarks

Comprehensive performance benchmarking suite for comparing vector stores (Milvus, Qdrant, ChromaDB) across different operations and scales.

> 📚 **Full Documentation**: See [docs/benchmarks.md](../../docs/benchmarks.md) for complete benchmark documentation.

## Overview

This benchmark suite tests:
- **Insert Operations**: Single and batch insert performance (1, 100, 1K vectors)
- **Search Operations**: Top-k search with varying k values (1, 10, 100)
- **Batch Operations**: Combined insert+delete and insert+search cycles
- **Scalability**: Large-scale operations (10K vectors)

## Requirements

```bash
# Install with benchmark dependencies
make dev
# Or manually
pip install pytest-benchmark
```

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
make benchmark

# Compare with previous run
make benchmark-compare

# Generate HTML report
make benchmark-report
```

### Advanced Usage

```bash
# Run specific benchmark group
pytest tests/benchmarks/ -k insert --benchmark-only

# Run benchmarks for specific vector store
pytest tests/benchmarks/ -k milvus --benchmark-only

# Save results with custom name
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave --benchmark-name=my_run

# Compare specific runs
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=0001 --benchmark-compare-fail=mean:10%
```

## Benchmark Groups

### Insert Benchmarks (`test_insert_benchmark.py`)
- Single vector insert
- Batch insert (100 vectors)
- Batch insert (1,000 vectors)

### Search Benchmarks (`test_search_benchmark.py`)
- Top-1 search
- Top-10 search
- Top-100 search
- All tests run on 1,000 vector database

### Batch Benchmarks (`test_batch_benchmark.py`)
- Bulk insert + delete (500 vectors)
- Insert + search cycle (100 vectors)

### Scale Benchmarks (`test_scale_benchmark.py`)
- 10K vector insert
- Search in large database (10K vectors)

## Report Generation

The benchmark suite includes an HTML report generator with:
- **Summary Cards**: Total benchmarks, fastest/slowest operations, average times
- **Interactive Charts**: Bar charts comparing performance across stores
- **Detailed Tables**: Complete statistics (mean, stddev, min, max, ops/sec)
- **Visual Styling**: Color-coded by vector store, responsive design

### Manual Report Generation

```python
from tests.benchmarks.utils.report_generator import BenchmarkReportGenerator

# Generate report from JSON file
gen = BenchmarkReportGenerator('.benchmarks/Linux-CPython-3.11-64bit/0001_*.json')
gen.generate_html('my_report.html')
```

## Data Generation

All benchmarks use the `VectorDataGenerator` utility for reproducible synthetic data:

```python
from tests.benchmarks.utils.data_generator import VectorDataGenerator

# Create generator (dimension=384, seed=42)
gen = VectorDataGenerator(dimension=384, seed=42)

# Generate format-specific data
milvus_data = gen.generate_milvus_data(100)  # INT64 IDs
qdrant_data = gen.generate_qdrant_points(100)  # UUID IDs
chroma_ids, chroma_emb, chroma_meta = gen.generate_chroma_data(100)
```

## Fixtures

Shared fixtures in `conftest.py`:
- `data_generator`: Session-scoped data generator
- `sample_data_100/1k/10k`: Pre-generated datasets
- `milvus_benchmark_service`: Fresh Milvus instance per test
- `qdrant_benchmark_service`: Fresh Qdrant instance per test
- `chroma_benchmark_service`: Fresh ChromaDB instance per test

## Configuration

Benchmarks use environment variables for service connections:
- `MILVUS_URI`: Default `http://localhost:19530`
- `QDRANT_URL`: Default `http://localhost:6333`
- ChromaDB: Local persistence at `./chroma_benchmark_data`

## Interpreting Results

### Console Output
```
--------------------------------------------------------------------------------------- benchmark: 9 tests ---------------------------------------------------------------------------------------
Name (time in ms)                          Min                 Max                Mean            StdDev              Median               IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_milvus_single_insert                 5.2315 (1.0)        8.4521 (1.0)       5.8932 (1.0)      0.6234 (1.0)       5.7234 (1.0)      0.3421 (1.0)         12;5  169.6944 (1.0)         100          1
test_qdrant_single_insert                 6.1234 (1.17)       9.2341 (1.09)      6.7823 (1.15)     0.7123 (1.14)      6.5234 (1.14)     0.4321 (1.26)         8;4  147.4521 (0.87)        100          1
test_chroma_single_insert                 7.2341 (1.38)      11.3421 (1.34)      8.2341 (1.40)     0.8234 (1.32)      7.9234 (1.38)     0.5421 (1.58)         6;3  121.4521 (0.72)        100          1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Key Metrics
- **Mean**: Average execution time (lower is better)
- **StdDev**: Consistency (lower is better)
- **OPS**: Operations per second (higher is better)
- **Min/Max**: Best/worst case performance

### HTML Report
- Color-coded bars per vector store (Milvus=Green, Qdrant=Yellow, Chroma=Blue)
- Summary cards for quick overview
- Detailed tables with all statistics
- Responsive design for mobile/desktop viewing

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Run Benchmarks
  run: |
    make benchmark
    make benchmark-report
    
- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-report
    path: benchmark_report.html
```

## Tips

1. **Warm-up**: First run may be slower due to service initialization
2. **Consistency**: Run benchmarks on same hardware for valid comparisons
3. **Isolation**: Close other applications to minimize interference
4. **Services**: Ensure vector stores are running (`make docker-up`)
5. **Cleanup**: Benchmarks clean up after themselves, but check for orphaned collections

## Troubleshooting

### Services Not Running
```bash
# Start all services
make docker-up

# Check health
make docker-health
```

### Import Errors
```bash
# Install dev dependencies
make dev
```

### Slow Benchmarks
```bash
# Run specific group
pytest tests/benchmarks/test_insert_benchmark.py --benchmark-only

# Skip scalability tests
pytest tests/benchmarks/ --benchmark-only -k "not scale"
```

## Architecture

```
tests/benchmarks/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_insert_benchmark.py # Insert performance
├── test_search_benchmark.py # Search performance
├── test_batch_benchmark.py  # Batch operations
├── test_scale_benchmark.py  # Scalability tests
└── utils/
    ├── __init__.py
    ├── data_generator.py    # Synthetic data generation
    └── report_generator.py  # HTML report generation
```

## Contributing

To add new benchmarks:

1. Create test function with `@pytest.mark.benchmark(group="mygroup")` decorator
2. Use fixtures for services and data generation
3. Call `benchmark(function, *args, **kwargs)` to measure performance
4. Update this README with new benchmark descriptions

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [RAG Toolkit documentation](https://gmottola00.github.io/rag-toolkit/)
