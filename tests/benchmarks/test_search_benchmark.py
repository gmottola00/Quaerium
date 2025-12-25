"""Benchmark search operations across vector stores."""

from __future__ import annotations

import pytest

from tests.benchmarks.utils.wrapper import VectorStoreWrapper
from tests.benchmarks.utils.data_generator import VectorDataGenerator


# Milvus Search Benchmarks
@pytest.mark.benchmark(group="search")
def test_milvus_search_top1(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-1 in Milvus."""
    # Insert 1000 vectors
    data = data_generator.generate_milvus_data(1000)
    milvus_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(milvus_benchmark_service.search, query, top_k=1)


@pytest.mark.benchmark(group="search")
def test_milvus_search_top10(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-10 in Milvus."""
    # Insert 1000 vectors
    data = data_generator.generate_milvus_data(1000)
    milvus_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(milvus_benchmark_service.search, query, top_k=10)


@pytest.mark.benchmark(group="search")
def test_milvus_search_top100(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-100 in Milvus."""
    # Insert 1000 vectors
    data = data_generator.generate_milvus_data(1000)
    milvus_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(milvus_benchmark_service.search, query, top_k=100)


# Qdrant Search Benchmarks
@pytest.mark.benchmark(group="search")
def test_qdrant_search_top1(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-1 in Qdrant."""
    # Insert 1000 vectors
    data = data_generator.generate_qdrant_points(1000)
    qdrant_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(qdrant_benchmark_service.search, query, top_k=1)


@pytest.mark.benchmark(group="search")
def test_qdrant_search_top10(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-10 in Qdrant."""
    # Insert 1000 vectors
    data = data_generator.generate_qdrant_points(1000)
    qdrant_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(qdrant_benchmark_service.search, query, top_k=10)


@pytest.mark.benchmark(group="search")
def test_qdrant_search_top100(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-100 in Qdrant."""
    # Insert 1000 vectors
    data = data_generator.generate_qdrant_points(1000)
    qdrant_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(qdrant_benchmark_service.search, query, top_k=100)


# ChromaDB Search Benchmarks
@pytest.mark.benchmark(group="search")
def test_chroma_search_top1(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-1 in ChromaDB."""
    # Insert 1000 vectors
    data = data_generator.generate_chroma_data(1000)
    chroma_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(chroma_benchmark_service.search, query, top_k=1)


@pytest.mark.benchmark(group="search")
def test_chroma_search_top10(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-10 in ChromaDB."""
    # Insert 1000 vectors
    data = data_generator.generate_chroma_data(1000)
    chroma_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(chroma_benchmark_service.search, query, top_k=10)


@pytest.mark.benchmark(group="search")
def test_chroma_search_top100(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark search top-100 in ChromaDB."""
    # Insert 1000 vectors
    data = data_generator.generate_chroma_data(1000)
    chroma_benchmark_service.add_vectors(data)
    
    # Search
    query = data_generator.generate_vector()
    benchmark(chroma_benchmark_service.search, query, top_k=100)
