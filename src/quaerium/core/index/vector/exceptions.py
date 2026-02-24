"""Custom exceptions for vector store operations."""

from __future__ import annotations


class VectorStoreError(RuntimeError):
    """Base exception for vector store failures."""


class ConfigurationError(VectorStoreError):
    """Configuration-related errors."""


class ConnectionError(VectorStoreError):
    """Raised when connection to Milvus fails."""


class CollectionError(VectorStoreError):
    """Raised when collection operations fail."""


class DataOperationError(VectorStoreError):
    """Raised when data-level operations fail."""


__all__ = [
    "VectorStoreError",
    "ConfigurationError",
    "ConnectionError",
    "CollectionError",
    "DataOperationError",
]
