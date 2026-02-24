"""Shared fixtures for unified vector store testing."""

from __future__ import annotations

import os
from typing import Generator

import pytest

# Import all vector store services
from quaerium.infra.vectorstores.milvus.config import MilvusConfig
from quaerium.infra.vectorstores.milvus.service import MilvusService
from quaerium.infra.vectorstores.qdrant.config import QdrantConfig
from quaerium.infra.vectorstores.qdrant.service import QdrantService

# ChromaDB might not be installed
try:
    from quaerium.infra.vectorstores.chroma.config import ChromaConfig
    from quaerium.infra.vectorstores.chroma.service import ChromaService
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@pytest.fixture(scope="session")
def milvus_service() -> Generator[MilvusService, None, None]:
    """Provide Milvus service for testing."""
    config = MilvusConfig(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        user=os.getenv("MILVUS_USER"),
        password=os.getenv("MILVUS_PASSWORD"),
    )
    
    service = MilvusService(config)
    
    try:
        yield service
    finally:
        # Cleanup: disconnect if possible
        if hasattr(service.connection, 'disconnect'):
            service.connection.disconnect()


@pytest.fixture(scope="session")
def qdrant_service() -> Generator[QdrantService, None, None]:
    """Provide Qdrant service for testing."""
    config = QdrantConfig(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    service = QdrantService(config)
    
    try:
        yield service
    finally:
        service.close()


@pytest.fixture(scope="session")
def chroma_service() -> Generator[ChromaService | None, None, None]:
    """Provide ChromaDB service for testing."""
    if not CHROMA_AVAILABLE:
        pytest.skip("ChromaDB not installed")
    
    # Use remote server if available, otherwise in-memory
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    
    config = ChromaConfig(host=host, port=port)
    
    service = ChromaService(config)
    
    try:
        yield service
    finally:
        service.close()


# Pytest collection modifier to skip unavailable vector stores
def pytest_collection_modifyitems(config, items):
    """Skip tests for unavailable vector stores."""
    skip_chroma = pytest.mark.skip(reason="ChromaDB not installed")
    
    for item in items:
        if "chroma" in item.nodeid and not CHROMA_AVAILABLE:
            item.add_marker(skip_chroma)


__all__ = [
    "milvus_service",
    "qdrant_service",
    "chroma_service",
]
