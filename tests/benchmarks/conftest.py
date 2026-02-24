"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import os
import pytest
from typing import Generator

from quaerium.infra.vectorstores.milvus import MilvusService, MilvusConfig
from quaerium.infra.vectorstores.qdrant import QdrantService, QdrantConfig
from quaerium.infra.vectorstores.chroma import ChromaService, ChromaConfig
from tests.benchmarks.utils.data_generator import VectorDataGenerator
from tests.benchmarks.utils.wrapper import VectorStoreWrapper


# Data generator fixture (session-scoped for reusability)
@pytest.fixture(scope="session")
def data_generator() -> VectorDataGenerator:
    """Provide a data generator for all tests."""
    return VectorDataGenerator(dimension=384, seed=42)


# Sample data fixtures for different scales
@pytest.fixture(scope="session")
def sample_data_100(data_generator: VectorDataGenerator) -> list:
    """Generate 100 sample data points."""
    return data_generator.generate_data(100)


@pytest.fixture(scope="session")
def sample_data_1k(data_generator: VectorDataGenerator) -> list:
    """Generate 1,000 sample data points."""
    return data_generator.generate_data(1000)


@pytest.fixture(scope="session")
def sample_data_10k(data_generator: VectorDataGenerator) -> list:
    """Generate 10,000 sample data points."""
    return data_generator.generate_data(10000)


# Milvus benchmark fixture
@pytest.fixture(scope="function")
def milvus_benchmark_service() -> Generator[VectorStoreWrapper, None, None]:
    """Provide a fresh Milvus service wrapped for benchmarking."""
    collection_name = f"benchmark_{os.getpid()}"
    
    config = MilvusConfig(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
    )
    service = MilvusService(config)
    
    # Create collection with schema
    schema = {
        "id": {"dtype": "INT64", "is_primary": True, "auto_id": False},
        "vector": {"dtype": "FLOAT_VECTOR", "dim": 384},
        "text": {"dtype": "VARCHAR", "max_length": 1000},
        "category": {"dtype": "VARCHAR", "max_length": 100},
        "score": {"dtype": "INT64"},
    }
    service.ensure_collection(collection_name, schema)
    
    wrapper = VectorStoreWrapper(service, collection_name, "milvus")
    
    try:
        yield wrapper
    finally:
        try:
            service.drop_collection(collection_name)
        except Exception:
            pass


# Qdrant benchmark fixture
@pytest.fixture(scope="function")
def qdrant_benchmark_service() -> Generator[VectorStoreWrapper, None, None]:
    """Provide a fresh Qdrant service wrapped for benchmarking."""
    collection_name = f"benchmark_{os.getpid()}"
    
    config = QdrantConfig(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    service = QdrantService(config)
    
    # Create collection
    service.ensure_collection(collection_name, vector_size=384)
    
    wrapper = VectorStoreWrapper(service, collection_name, "qdrant")
    
    try:
        yield wrapper
    finally:
        try:
            service.drop_collection(collection_name)
        except Exception:
            pass


# ChromaDB benchmark fixture
@pytest.fixture(scope="function")
def chroma_benchmark_service() -> Generator[VectorStoreWrapper, None, None]:
    """Provide a fresh ChromaDB service wrapped for benchmarking."""
    collection_name = f"benchmark_{os.getpid()}"
    
    config = ChromaConfig(
        path="./chroma_benchmark_data",
    )
    service = ChromaService(config)
    
    # Create collection
    service.create_collection(collection_name)
    
    wrapper = VectorStoreWrapper(service, collection_name, "chroma")
    
    try:
        yield wrapper
    finally:
        try:
            service.drop_collection(collection_name)
        except Exception:
            pass
