# Benchmark Suite - Architecture Note

## Issue Discovered

The vector store services (MilvusService, QdrantService, ChromaService) have significantly different APIs:

- **Milvus**: `service.insert(collection_name, data)` with INT64 IDs
- **Qdrant**: `service.data.upsert(collection_name, points)` with UUID IDs  
- **ChromaDB**: `service.add(collection_name, ids, embeddings, metadatas)`

The original benchmark design assumed a unified `add_vectors()` method that doesn't exist.

## Options

### Option 1: Use Unified Tests (Recommended)
The existing `tests/test_infra/test_vectorstores/test_unified.py` already provides comprehensive testing across all stores with proper API usage. To benchmark:

```bash
pytest tests/test_infra/test_vectorstores/test_unified.py --benchmark-only
```

### Option 2: Create Store-Specific Benchmarks
Create separate benchmark files for each store with store-specific APIs:
- `test_milvus_benchmarks.py` - Uses Milvus-specific insert/search
- `test_qdrant_benchmarks.py` - Uses Qdrant-specific upsert/search
- `test_chroma_benchmarks.py` - Uses ChromaDB-specific add/search

### Option 3: Create Wrapper Layer
Implement a thin abstraction layer in the benchmark utils to normalize the APIs.

## Current Status

- ✅ Directory structure created
- ✅ Data generator working
- ✅ HTML report generator working  
- ✅ Fixtures configured correctly
- ❌ Test implementations need store-specific API calls

## Next Steps

1. Decide on approach (Option 1, 2, or 3)
2. Update test files accordingly
3. Run benchmarks with proper vector store services running

