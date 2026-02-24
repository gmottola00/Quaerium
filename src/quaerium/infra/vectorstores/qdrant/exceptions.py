"""Custom exceptions for Qdrant vector store operations."""

from __future__ import annotations


class QdrantError(RuntimeError):
    """Base exception for Qdrant vector store failures."""


class ConfigurationError(QdrantError):
    """Configuration-related errors."""


class ConnectionError(QdrantError):
    """Raised when connection to Qdrant fails."""


class CollectionError(QdrantError):
    """Raised when collection operations fail."""


class DataOperationError(QdrantError):
    """Raised when data-level operations fail."""


__all__ = [
    "QdrantError",
    "ConfigurationError",
    "ConnectionError",
    "CollectionError",
    "DataOperationError",
]
