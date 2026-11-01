"""Storage infrastructure layer (Supabase backend: pip install quaerium[supabase])."""

from __future__ import annotations

__all__ = [
    "StorageClient",
    "SupabaseStorageClient",
    "get_storage_client",
]

_LAZY = {
    "StorageClient": "quaerium.infra.storage.base",
    "SupabaseStorageClient": "quaerium.infra.storage.supabase",
    "get_storage_client": "quaerium.infra.storage.supabase",
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
