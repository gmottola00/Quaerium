"""Factory for creating vector store services and index services.

This module provides production-ready factory functions for instantiating
vector stores with sensible defaults while allowing configuration overrides.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from quaerium.core.embedding import EmbeddingClient
from quaerium.core.index.service import IndexService
from quaerium.infra.vectorstores.milvus.config import MilvusConfig
from quaerium.infra.vectorstores.milvus.service import MilvusService
from quaerium.infra.vectorstores.qdrant.config import QdrantConfig
from quaerium.infra.vectorstores.qdrant.service import QdrantService
from quaerium.infra.vectorstores.chroma.config import ChromaConfig
from quaerium.infra.vectorstores.chroma.service import ChromaService


def create_milvus_service(
    *,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    db_name: Optional[str] = None,
    secure: bool = False,
    timeout: Optional[float] = None,
    alias: str = "default",
) -> MilvusService:
    """Create a configured MilvusService instance.
    
    Args:
        uri: Milvus connection URI (defaults to MILVUS_URI env var or localhost:19530).
        user: Username for authentication (defaults to MILVUS_USER env var).
        password: Password for authentication (defaults to MILVUS_PASSWORD env var).
        db_name: Database name (defaults to MILVUS_DB env var or "default").
        secure: Use TLS encryption.
        timeout: Connection timeout in seconds.
        alias: Connection alias for management.
        
    Returns:
        Configured MilvusService instance.
        
    Example:
        ```python
        from quaerium.infra.vectorstores.factory import create_milvus_service
        
        # Use environment variables
        service = create_milvus_service()
        
        # Override specific settings
        service = create_milvus_service(
            uri="http://localhost:19530",
            db_name="production"
        )
        ```
    """
    config = MilvusConfig(
        uri=uri or os.getenv("MILVUS_URI", "http://localhost:19530"),
        user=user or os.getenv("MILVUS_USER"),
        password=password or os.getenv("MILVUS_PASSWORD"),
        db_name=db_name or os.getenv("MILVUS_DB", "default"),
        secure=secure,
        timeout=timeout,
        alias=alias,
    )
    
    return MilvusService(config)


def create_index_service(
    embedding_dim: int,
    embed_fn: Callable[[List[str]], List[List[float]]],
    *,
    collection_name: Optional[str] = None,
    metric_type: Optional[str] = None,
    index_type: Optional[str] = None,
    vector_store: Optional[MilvusService] = None,
) -> IndexService:
    """Create a configured IndexService with Milvus backend.
    
    Args:
        embedding_dim: Dimensionality of embeddings.
        embed_fn: Function to generate embeddings from text.
        collection_name: Collection name (defaults to MILVUS_COLLECTION env var or "tender_chunks").
        metric_type: Distance metric (defaults to MILVUS_METRIC env var or "IP").
        index_type: Index type (defaults to MILVUS_INDEX_TYPE env var or "HNSW").
        vector_store: Optional pre-configured vector store (creates new if None).
        
    Returns:
        Configured IndexService instance.
        
    Example:
        ```python
        from quaerium.infra.vectorstores.factory import create_index_service
        from quaerium.core.embedding import EmbeddingClient
        
        # Create embedding client
        embed_client = EmbeddingClient(...)
        
        # Create index service
        index_service = create_index_service(
            embedding_dim=1536,
            embed_fn=lambda texts: embed_client.embed_batch(texts),
            collection_name="my_collection"
        )
        ```
    """
    if vector_store is None:
        vector_store = create_milvus_service()
    
    return IndexService(
        vector_store=vector_store,
        embedding_dim=embedding_dim,
        embed_fn=embed_fn,
        collection_name=collection_name or os.getenv("MILVUS_COLLECTION", "tender_chunks"),
        metric_type=metric_type or os.getenv("MILVUS_METRIC", "IP"),
        index_type=index_type or os.getenv("MILVUS_INDEX_TYPE", "HNSW"),
    )


def create_qdrant_service(
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    https: bool = False,
    grpc_port: Optional[int] = None,
    prefer_grpc: bool = False,
) -> QdrantService:
    """Create a configured QdrantService instance.
    
    Args:
        url: Qdrant server URL (defaults to QDRANT_URL env var or http://localhost:6333).
        api_key: API key for authentication (defaults to QDRANT_API_KEY env var).
        timeout: Request timeout in seconds.
        https: Use HTTPS for connection.
        grpc_port: gRPC port for high-performance operations.
        prefer_grpc: Prefer gRPC over HTTP when available.
        
    Returns:
        Configured QdrantService instance.
        
    Example:
        ```python
        from quaerium.infra.vectorstores.factory import create_qdrant_service
        
        # Use environment variables
        service = create_qdrant_service()
        
        # Override specific settings
        service = create_qdrant_service(
            url="http://localhost:6333",
            prefer_grpc=True
        )
        
        # Cloud instance
        service = create_qdrant_service(
            url="https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
            api_key="your-api-key",
            https=True
        )
        ```
    """
    config = QdrantConfig(
        url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=api_key or os.getenv("QDRANT_API_KEY"),
        timeout=timeout,
        https=https,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
    )
    
    return QdrantService(config)


def create_chroma_service(
    *,
    path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    ssl: bool = False,
    tenant: str = "default_tenant",
    database: str = "default_database",
) -> ChromaService:
    """Create a configured ChromaService instance.
    
    Args:
        path: Path for persistent storage (defaults to CHROMA_PATH env var).
            If None and host is None, uses in-memory mode.
        host: ChromaDB server host (defaults to CHROMA_HOST env var).
        port: ChromaDB server port (defaults to CHROMA_PORT env var or 8000).
        ssl: Use SSL for remote connections.
        tenant: Tenant name for multi-tenancy.
        database: Database name within tenant.
        
    Returns:
        Configured ChromaService instance.
        
    Example:
        ```python
        from quaerium.infra.vectorstores.factory import create_chroma_service
        
        # In-memory mode (default)
        service = create_chroma_service()
        
        # Persistent mode
        service = create_chroma_service(path="./chroma_data")
        
        # Remote server
        service = create_chroma_service(
            host="localhost",
            port=8000
        )
        
        # Cloud instance with SSL
        service = create_chroma_service(
            host="chroma.example.com",
            port=443,
            ssl=True
        )
        ```
    """
    config = ChromaConfig(
        path=path or os.getenv("CHROMA_PATH"),
        host=host or os.getenv("CHROMA_HOST"),
        port=port or int(os.getenv("CHROMA_PORT", "8000")) if os.getenv("CHROMA_PORT") else None,
        ssl=ssl,
        tenant=tenant,
        database=database,
    )
    
    return ChromaService(config)


__all__ = [
    "create_milvus_service",
    "create_qdrant_service",
    "create_chroma_service",
    "create_index_service",
]
