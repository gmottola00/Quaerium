"""Vector stores infrastructure package.

This package contains concrete implementations of vector database protocols
and factory functions for creating configured instances.
"""
from __future__ import annotations

__all__ = [
    "create_milvus_service",
    "create_qdrant_service",
    "create_chroma_service",
    "create_index_service",
]

_LAZY = {
    "create_milvus_service": "quaerium.infra.vectorstores.factory",
    "create_qdrant_service": "quaerium.infra.vectorstores.factory",
    "create_chroma_service": "quaerium.infra.vectorstores.factory",
    "create_index_service": "quaerium.infra.vectorstores.factory",
}


def __getattr__(name: str):
    """Lazy import so optional dependencies are loaded only when used."""
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))
