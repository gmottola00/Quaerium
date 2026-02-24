"""Configuration models for Qdrant vector store access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant connection configuration.
    
    Args:
        url: Qdrant server URL (e.g., "http://localhost:6333")
        api_key: API key for authentication (optional)
        timeout: Request timeout in seconds (optional)
        https: Use HTTPS for connection
        grpc_port: gRPC port for high-performance operations (optional)
        prefer_grpc: Prefer gRPC over HTTP when available
    """

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: Optional[float] = None
    https: bool = False
    grpc_port: Optional[int] = None
    prefer_grpc: bool = False


@dataclass(frozen=True)
class QdrantIndexConfig:
    """Qdrant index configuration.
    
    Args:
        distance: Distance metric ("Cosine", "Euclid", or "Dot")
        hnsw_config: HNSW index parameters
        quantization: Quantization config for memory efficiency
        on_disk: Store vectors on disk for large datasets
    """

    distance: str = "Cosine"  # "Cosine", "Euclid", "Dot"
    hnsw_config: Optional[dict] = None  # {"m": 16, "ef_construct": 100}
    quantization: Optional[dict] = None  # {"scalar": {"type": "int8"}}
    on_disk: bool = False


__all__ = [
    "QdrantConfig",
    "QdrantIndexConfig",
]
