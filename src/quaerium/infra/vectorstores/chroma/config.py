"""Configuration models for ChromaDB vector store access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB connection configuration.
    
    Args:
        path: Path for persistent storage (None for in-memory)
        host: Remote ChromaDB server host (optional)
        port: Remote ChromaDB server port (optional)
        ssl: Use SSL for remote connection
        headers: Additional headers for remote connection
        tenant: Multi-tenancy tenant name (default: "default_tenant")
        database: Multi-tenancy database name (default: "default_database")
    """

    path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    ssl: bool = False
    headers: Optional[dict] = None
    tenant: str = "default_tenant"
    database: str = "default_database"


@dataclass(frozen=True)
class ChromaIndexConfig:
    """ChromaDB index configuration.
    
    Args:
        distance: Distance metric ("cosine", "l2", "ip")
        hnsw_space: HNSW space type (same as distance, for compatibility)
        hnsw_construction_ef: HNSW construction ef parameter
        hnsw_search_ef: HNSW search ef parameter
        hnsw_M: HNSW M parameter (max connections per node)
    """

    distance: str = "cosine"  # "cosine", "l2", "ip"
    hnsw_space: Optional[str] = None  # Auto-set from distance if None
    hnsw_construction_ef: int = 100
    hnsw_search_ef: int = 10
    hnsw_M: int = 16


__all__ = [
    "ChromaConfig",
    "ChromaIndexConfig",
]
