"""Custom exceptions for ChromaDB vector store operations."""

from __future__ import annotations


class ChromaError(RuntimeError):
    """Base exception for ChromaDB vector store failures."""


class ConfigurationError(ChromaError):
    """Configuration-related errors."""


class ConnectionError(ChromaError):
    """Raised when connection to ChromaDB fails."""


class CollectionError(ChromaError):
    """Raised when collection operations fail."""


class DataOperationError(ChromaError):
    """Raised when data-level operations fail."""


__all__ = [
    "ChromaError",
    "ConfigurationError",
    "ConnectionError",
    "CollectionError",
    "DataOperationError",
]
