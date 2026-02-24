"""LLM infrastructure implementations."""

from __future__ import annotations

__all__ = ["OllamaLLMClient", "OpenAILLMClient"]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "OllamaLLMClient":
        from .ollama import OllamaLLMClient
        return OllamaLLMClient
    elif name == "OpenAILLMClient":
        from .openai import OpenAILLMClient
        return OpenAILLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
