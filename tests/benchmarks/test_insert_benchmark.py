"""Benchmark insert operations across vector stores."""

from __future__ import annotations

import pytest

from tests.benchmarks.utils.wrapper import VectorStoreWrapper
from tests.benchmarks.utils.data_generator import VectorDataGenerator


# Milvus Insert Benchmarks
@pytest.mark.benchmark(group="insert")
def test_milvus_single_insert(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark single insert in Milvus."""
    data = data_generator.generate_milvus_data(1)
    
    benchmark(milvus_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_milvus_batch_insert_100(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 100 vectors in Milvus."""
    data = data_generator.generate_milvus_data(100)
    
    benchmark(milvus_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_milvus_batch_insert_1k(
    benchmark,
    milvus_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 1,000 vectors in Milvus."""
    data = data_generator.generate_milvus_data(1000)
    
    benchmark(milvus_benchmark_service.add_vectors, data)


# Qdrant Insert Benchmarks
@pytest.mark.benchmark(group="insert")
def test_qdrant_single_insert(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark single insert in Qdrant."""
    data = data_generator.generate_qdrant_points(1)
    
    benchmark(qdrant_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_qdrant_batch_insert_100(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 100 vectors in Qdrant."""
    data = data_generator.generate_qdrant_points(100)
    
    benchmark(qdrant_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_qdrant_batch_insert_1k(
    benchmark,
    qdrant_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 1,000 vectors in Qdrant."""
    data = data_generator.generate_qdrant_points(1000)
    
    benchmark(qdrant_benchmark_service.add_vectors, data)


# ChromaDB Insert Benchmarks
@pytest.mark.benchmark(group="insert")
def test_chroma_single_insert(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark single insert in ChromaDB."""
    data = data_generator.generate_chroma_data(1)
    
    benchmark(chroma_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_chroma_batch_insert_100(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 100 vectors in ChromaDB."""
    data = data_generator.generate_chroma_data(100)
    
    benchmark(chroma_benchmark_service.add_vectors, data)


@pytest.mark.benchmark(group="insert")
def test_chroma_batch_insert_1k(
    benchmark,
    chroma_benchmark_service: VectorStoreWrapper,
    data_generator: VectorDataGenerator,
):
    """Benchmark batch insert of 1,000 vectors in ChromaDB."""
    data = data_generator.generate_chroma_data(1000)
    
    benchmark(chroma_benchmark_service.add_vectors, data)
