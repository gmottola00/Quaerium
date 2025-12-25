# Benchmark Suite Setup Complete ✅

## What Was Created

A comprehensive performance benchmarking suite for comparing vector stores (Milvus, Qdrant, ChromaDB).

### Directory Structure

```
tests/benchmarks/
├── __init__.py                  # Package initialization
├── conftest.py                  # Shared pytest fixtures
├── test_insert_benchmark.py     # Insert performance tests (9 tests)
├── test_search_benchmark.py     # Search performance tests (9 tests)
├── test_batch_benchmark.py      # Batch operations tests (6 tests)
├── test_scale_benchmark.py      # Scalability tests (6 tests)
├── README.md                    # Complete documentation
└── utils/
    ├── __init__.py
    ├── data_generator.py        # Synthetic data generator
    └── report_generator.py      # HTML report generator
```

**Total: 30 benchmark tests across 4 categories**

## Components

### 1. Data Generator (`utils/data_generator.py`)
- **Class**: `VectorDataGenerator`
- **Features**:
  - Reproducible data generation (seed=42)
  - Vector dimension: 384
  - Format-specific generators for Milvus (INT64 IDs), Qdrant (UUID), ChromaDB
  - Metadata generation (category, score, text fields)

### 2. Report Generator (`utils/report_generator.py`)
- **Class**: `BenchmarkReportGenerator`
- **Features**:
  - Reads pytest-benchmark JSON output
  - Generates standalone HTML with Chart.js
  - Summary cards (total benchmarks, fastest/slowest, average)
  - Interactive bar charts (color-coded by store)
  - Detailed tables (mean, stddev, min, max, ops/sec)
  - Gradient styling (#667eea → #764ba2)
  - Responsive design

### 3. Fixtures (`conftest.py`)
- **Session-scoped**:
  - `data_generator`: Single generator instance
  - `sample_data_100/1k/10k`: Pre-generated datasets
  
- **Function-scoped**:
  - `milvus_benchmark_service`: Fresh Milvus per test
  - `qdrant_benchmark_service`: Fresh Qdrant per test
  - `chroma_benchmark_service`: Fresh ChromaDB per test

### 4. Benchmark Tests

#### Insert Benchmarks (9 tests)
- Single insert (1 vector)
- Batch insert 100 vectors
- Batch insert 1,000 vectors
- One test per vector store

#### Search Benchmarks (9 tests)
- Top-1 search
- Top-10 search
- Top-100 search
- All on 1,000 vector database
- One test per vector store

#### Batch Benchmarks (6 tests)
- Bulk insert + delete (500 vectors)
- Insert + search cycle (100 vectors)
- One test per vector store

#### Scale Benchmarks (6 tests)
- 10K vector insert
- Search in 10K vector database
- One test per vector store

## Installation

### Dependencies Added
- `pytest-benchmark>=4.0.0` added to `[project.optional-dependencies.dev]` in `pyproject.toml`

### Install
```bash
# Install all dev dependencies including pytest-benchmark
make dev

# Or manually
pip install -e ".[dev,all]"
```

## Usage

### Quick Start
```bash
# Run all benchmarks
make benchmark

# Compare with previous run
make benchmark-compare

# Generate HTML report
make benchmark-report

# Clean benchmark results
make benchmark-clean
```

### Advanced Usage
```bash
# Run specific group
pytest tests/benchmarks/ -k insert --benchmark-only

# Run specific store
pytest tests/benchmarks/ -k milvus --benchmark-only

# Custom benchmark name
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave --benchmark-name=production_hw

# Compare specific runs
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=0001 --benchmark-compare-fail=mean:10%

# Skip slow tests
pytest tests/benchmarks/ --benchmark-only -k "not scale"
```

## Makefile Targets

Added to `Makefile`:
- `make benchmark` - Run all benchmarks with autosave
- `make benchmark-compare` - Compare with last run
- `make benchmark-report` - Generate HTML report and open in browser
- `make benchmark-clean` - Remove all benchmark data

## Output

### Console Output
- Comparison table with all benchmarks
- Statistics: Min, Max, Mean, StdDev, Median, IQR, Outliers, OPS
- Relative performance (e.g., "1.17x slower")

### JSON Output
- Saved to `.benchmarks/<machine-info>/NNNN_*.json`
- Contains full statistics and metadata
- Used for report generation

### HTML Report
- Filename: `benchmark_report.html`
- Features:
  - Summary cards at top
  - One chart per benchmark group
  - Detailed tables below each chart
  - Color-coded by vector store
  - Responsive layout

## Configuration

### Environment Variables
- `MILVUS_URI` - Default: `http://localhost:19530`
- `QDRANT_URL` - Default: `http://localhost:6333`
- ChromaDB uses local persistence: `./chroma_benchmark_data`

### Prerequisites
```bash
# Start vector stores
make docker-up

# Check health
make docker-health
```

## Example Workflow

```bash
# 1. Start services
make docker-up

# 2. Install dependencies
make dev

# 3. Run benchmarks
make benchmark

# 4. Generate report
make benchmark-report
# Opens benchmark_report.html in browser

# 5. Compare later run
# ... make changes ...
make benchmark-compare

# 6. Cleanup
make benchmark-clean
make docker-down
```

## CI/CD Integration

Add to `.github/workflows/benchmarks.yml`:
```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: make dev
      
      - name: Start services
        run: make docker-up
      
      - name: Run benchmarks
        run: make benchmark
      
      - name: Generate report
        run: make benchmark-report
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            .benchmarks/
            benchmark_report.html
```

## Key Features

✅ **Comprehensive**: 30 tests across 4 operation types
✅ **Production-Ready**: Fixtures handle cleanup automatically
✅ **Reproducible**: Seeded data generation for consistent results
✅ **Visual**: HTML reports with charts and tables
✅ **Flexible**: Easy to add new benchmarks
✅ **CI-Friendly**: JSON output and comparison tools
✅ **Well-Documented**: Complete README with examples

## Next Steps

1. **Run First Benchmark**: `make benchmark`
2. **Review Report**: `make benchmark-report`
3. **Baseline**: Save results as baseline for future comparisons
4. **Integrate CI/CD**: Add to GitHub Actions
5. **Extend**: Add custom benchmarks for specific use cases

## Testing the Suite

```bash
# Quick validation (skip actual benchmarks)
pytest tests/benchmarks/ --collect-only

# Run just insert benchmarks (fast)
pytest tests/benchmarks/test_insert_benchmark.py --benchmark-only -k "single"

# Full run (may take several minutes)
make benchmark
```

## Troubleshooting

### Import Errors
```bash
make dev  # Reinstall dependencies
```

### Services Not Running
```bash
make docker-up
make docker-health
```

### Slow Performance
- Close other applications
- Run specific groups: `pytest tests/benchmarks/test_insert_benchmark.py`
- Skip scale tests: `-k "not scale"`

## Documentation

- Full guide: `tests/benchmarks/README.md`
- pytest-benchmark docs: https://pytest-benchmark.readthedocs.io/
- RAG Toolkit docs: https://gmottola00.github.io/rag-toolkit/

---

**Created**: December 2024
**Python Version**: 3.11+
**Framework**: pytest-benchmark 4.0+
**Stores**: Milvus, Qdrant, ChromaDB
