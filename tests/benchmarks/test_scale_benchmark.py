"""Benchmark scale operations across vector stores."""

from __future__ import annotations

import pytest

from tests.benchmarks.utils.wrapper import VectorStoreWrapper
from tests.benchmarks.utils.data_generator import VectorDataGenerator


# Milvus Scalability Benchmarks
@pytest.mark.benchmark(group="scale")
def test_milvus_scale_10k_insert(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark 10K insert in Milvus."""
    data = data_generator.generate_milvus_data(10000)
    
    benchmark(milvus_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="scale")
def test_milvus_scale_search_large_db(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search in large database (10K vectors) in Milvus."""
    # Insert 10K vectors
    data = data_generator.generate_milvus_data(10000)
    milvus_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(milvus_benchmark_service.search, query, top_k=10)


# Qdrant Scalability Benchmarks
@pytest.mark.benchmark(group="scale")
def test_qdrant_scale_10k_insert(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark 10K insert in Qdrant."""
    data = data_generator.generate_qdrant_points(10000)
    
    benchmark(qdrant_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="scale")
def test_qdrant_scale_search_large_db(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search in large database (10K vectors) in Qdrant."""
    # Insert 10K vectors
    data = data_generator.generate_qdrant_points(10000)
    qdrant_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(qdrant_benchmark_service.search, query, top_k=10)


# ChromaDB Scalability Benchmarks
@pytest.mark.benchmark(group="scale")
def test_chroma_scale_10k_insert(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark 10K insert in ChromaDB."""
    data = data_generator.generate_chroma_data(10000)
    
    benchmark(chroma_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="scale")
def test_chroma_scale_search_large_db(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search in large database (10K vectors) in ChromaDB."""
    # Insert 10K vectors
    data = data_generator.generate_chroma_data(10000)
    chroma_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(chroma_benchmark_service.search, query, top_k=10)
