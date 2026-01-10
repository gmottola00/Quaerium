# Benchmark Results

<style>
.benchmark-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 8px;
    margin: 30px 0;
    text-align: center;
    color: white;
}

.benchmark-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 30px 0;
}

.stat-card {
    background: var(--md-code-bg-color);
    padding: 20px;
    border-radius: 8px;
    border: 1px solid var(--md-default-fg-color--lightest);
}

.stat-card h3 {
    font-size: 0.9rem;
    text-transform: uppercase;
    color: var(--md-primary-fg-color);
    margin-bottom: 10px;
}

.stat-card .value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--md-default-fg-color);
}
</style>

<div class="benchmark-hero">
  <h2>🚀 Vector Store Performance Benchmarks</h2>
  <p>Real-world performance comparison across Milvus, Qdrant, and ChromaDB</p>
  <a href="../_static/benchmark_report.html" class="md-button md-button--primary">View Interactive Report</a>
</div>

## Summary

<div class="benchmark-stats">
  <div class="stat-card">
    <h3>Total Benchmarks</h3>
    <div class="value">29</div>
  </div>
  <div class="stat-card">
    <h3>Fastest Operation</h3>
    <div class="value">0.64 ms</div>
  </div>
  <div class="stat-card">
    <h3>Slowest Operation</h3>
    <div class="value">222.7 s</div>
  </div>
  <div class="stat-card">
    <h3>Average Time</h3>
    <div class="value">10.1 s</div>
  </div>
</div>

## Key Findings

### Insert Performance

=== "Single Insert"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **Qdrant** 🥇 | 1.19 ms | 843 ops/s |
    | **ChromaDB** 🥈 | 1.64 ms | 609 ops/s |
    | **Milvus** | 11,087 ms | 0.09 ops/s |

=== "Batch Insert (100 vectors)"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **ChromaDB** 🥇 | 6.46 ms | 155 ops/s |
    | **Qdrant** 🥈 | 25.53 ms | 39 ops/s |
    | **Milvus** | 11,082 ms | 0.09 ops/s |

=== "Large Batch (1K vectors)"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **ChromaDB** 🥇 | 46.53 ms | 21 ops/s |
    | **Qdrant** 🥈 | 260.07 ms | 3.85 ops/s |
    | **Milvus** | 22,110 ms | 0.05 ops/s |

### Search Performance

=== "Top-1 Search"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **ChromaDB** 🥇 | 0.64 ms | 1551 ops/s |
    | **Qdrant** 🥈 | 1.41 ms | 711 ops/s |
    | **Milvus** | 2.27 ms | 440 ops/s |

=== "Top-10 Search"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **ChromaDB** 🥇 | 0.90 ms | 1108 ops/s |
    | **Milvus** 🥈 | 2.30 ms | 435 ops/s |
    | **Qdrant** | 2.75 ms | 364 ops/s |

=== "Top-100 Search"
    | Store | Mean Time | Ops/Sec |
    |-------|-----------|---------|
    | **Milvus** 🥇 | 2.16 ms | 462 ops/s |
    | **ChromaDB** 🥈 | 2.99 ms | 334 ops/s |
    | **Qdrant** | 8.78 ms | 114 ops/s |

### Scale Performance (10K vectors)

| Store | Insert Time | Search Time | Total |
|-------|------------|-------------|-------|
| **ChromaDB** 🥇 | 601 ms | 1.02 ms | 602 ms |
| **Qdrant** 🥈 | 3,178 ms | 3.29 ms | 3,181 ms |
| **Milvus** | 222,734 ms | 2.40 ms | 222,736 ms |

## Insights

!!! success "ChromaDB: Best for Rapid Prototyping"
    - **Fastest** single and batch inserts
    - **Lowest latency** searches (0.64ms)
    - **Best scale performance** (10K in 601ms)
    - **Ideal for**: Development, testing, small to medium datasets

!!! tip "Qdrant: Balanced Performance"
    - **Good** insert performance (1.19ms single)
    - **Consistent** search latency
    - **Reliable** at scale (3.2s for 10K)
    - **Ideal for**: Production, medium to large datasets, distributed deployments

!!! warning "Milvus: Specialized Use Cases"
    - **Slow** inserts due to flush requirements (11s)
    - **Competitive** search performance (2.27ms)
    - **Not ideal** for real-time applications
    - **Ideal for**: Batch processing, large-scale analytics, read-heavy workloads

## Methodology

### Test Environment

- **Hardware**: MacBook Air M1
- **Python**: 3.12.12
- **Benchmark Tool**: pytest-benchmark 5.2.3
- **Iterations**: 5-10 rounds per test
- **Date**: December 25, 2025

### Benchmark Categories

- **Insert Benchmarks**: Single, batch (100), large batch (1K)
- **Search Benchmarks**: Top-1, Top-10, Top-100
- **Batch Operations**: Bulk insert/delete cycles
- **Scale Tests**: 10K vector operations

### Vector Configuration

- **Dimension**: 384 (nomic-embed-text)
- **Metric**: Cosine similarity
- **Batch Size**: 500 (for large operations)

## Interactive Report

For detailed results with charts and full statistics:

[View Interactive Benchmark Report](../_static/benchmark_report.html){ .md-button .md-button--primary }

The interactive report includes:

- 📊 Bar charts for visual comparison
- 📈 Detailed timing statistics
- 📉 Min/Max/StdDev metrics
- 🔍 Filterable results

## Running Benchmarks

To reproduce these results:

```bash
# Start vector stores
docker-compose up -d

# Run benchmarks
make benchmark

# Generate report
make benchmark-report
```

See [Benchmarks Guide](benchmarks.md) for detailed instructions.

## Recommendations

### For Development

Use **ChromaDB** for fast iteration:

```python
store = get_chromadb_service(host="localhost")
```

### For Production

Use **Qdrant** for reliability:

```python
store = get_qdrant_service(
    host="qdrant.example.com",
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY")
)
```

### For Analytics

Use **Milvus** for batch processing:

```python
store = get_milvus_service(
    host="milvus.example.com",
    port=19530
)
```

## Version Information

- **RAG Toolkit**: 0.1.0
- **Milvus**: 2.3+
- **Qdrant**: 1.7+
- **ChromaDB**: 0.4+

---

<small>Last updated: January 5, 2026</small>
