"""Benchmark batch operations across vector stores."""

from __future__ import annotations

import pytest

from tests.benchmarks.utils.wrapper import VectorStoreWrapper
from tests.benchmarks.utils.data_generator import VectorDataGenerator


# Milvus Batch Benchmarks
@pytest.mark.benchmark(group="batch")
def test_milvus_bulk_insert_delete(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark bulk insert + delete in Milvus."""
    data = data_generator.generate_milvus_data(500)
    
    def bulk_operation():
        # Insert
        milvus_benchmark_service.add_vectors(data)
        # Delete all
        ids = [d["id"] for d in data]
        milvus_benchmark_service.delete_vectors(ids)
    
    benchmark(bulk_operation)


@pytest.mark.benchmark(group="batch")
def test_milvus_insert_search_cycle(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark insert + search cycle in Milvus."""
    data = data_generator.generate_milvus_data(100)
    query = data_generator.generate_vector()
    
    def insert_search():
        milvus_benchmark_service.add_vectors(data)
        milvus_benchmark_service.search(query, top_k=10)
    
    benchmark(insert_search)


# Qdrant Batch Benchmarks
@pytest.mark.benchmark(group="batch")
def test_qdrant_bulk_insert_delete(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark bulk insert + delete in Qdrant."""
    data = data_generator.generate_qdrant_points(500)
    
    def bulk_operation():
        # Insert
        qdrant_benchmark_service.add_vectors(data)
        # Delete all
        ids = [d["id"] for d in data]
        qdrant_benchmark_service.delete_vectors(ids)
    
    benchmark(bulk_operation)


@pytest.mark.benchmark(group="batch")
def test_qdrant_insert_search_cycle(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark insert + search cycle in Qdrant."""
    data = data_generator.generate_qdrant_points(100)
    query = data_generator.generate_vector()
    
    def insert_search():
        qdrant_benchmark_service.add_vectors(data)
        qdrant_benchmark_service.search(query, top_k=10)
    
    benchmark(insert_search)


# ChromaDB Batch Benchmarks
@pytest.mark.benchmark(group="batch")
def test_chroma_bulk_insert_delete(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark bulk insert + delete in ChromaDB."""
    data = data_generator.generate_chroma_data(500)
    
    def bulk_operation():
        # Insert
        chroma_benchmark_service.add_vectors(data)
        # Delete all
        ids = [d["id"] for d in data]
        chroma_benchmark_service.delete_vectors(ids)
    
    benchmark(bulk_operation)


@pytest.mark.benchmark(group="batch")
def test_chroma_insert_search_cycle(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark insert + search cycle in ChromaDB."""
    data = data_generator.generate_chroma_data(100)
    query = data_generator.generate_vector()
    
    def insert_search():
        chroma_benchmark_service.add_vectors(data)
        chroma_benchmark_service.search(query, top_k=10)
    
    benchmark(insert_search)
