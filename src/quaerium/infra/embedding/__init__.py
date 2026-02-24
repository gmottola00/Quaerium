"""Embedding infrastructure implementations."""

from __future__ import annotations

__all__ = ["OllamaEmbeddingClient", "OpenAIEmbeddingClient"]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "OllamaEmbeddingClient":
        from .ollama import OllamaEmbeddingClient
        return OllamaEmbeddingClient
    elif name == "OpenAIEmbeddingClient":
        from .openai import OpenAIEmbeddingClient
        return OpenAIEmbeddingClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
