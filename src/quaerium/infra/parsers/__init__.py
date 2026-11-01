"""Infrastructure - Document parsing implementations.

This module contains concrete implementations of parsers for various document formats.
Use the factory to create fully configured services.
"""
from __future__ import annotations

__all__ = [
    "create_ingestion_service",
    "create_lightweight_ingestion_service",
]

_LAZY = {
    "create_ingestion_service": ".factory",
    "create_lightweight_ingestion_service": ".factory",
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
