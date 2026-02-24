"""Base Protocol for LLM clients."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol


class LLMClient(Protocol):
    """Protocol for Large Language Model clients.
    
    Uses Protocol instead of ABC for more flexible duck typing.
    Any class implementing these methods can be used as an LLMClient.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion from a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text completion
        """
        ...

    def generate_batch(self, prompts: Iterable[str], **kwargs: Any) -> Iterable[str]:
        """Optional batch generation.
        
        Default implementation iterates generate() for each prompt.
        Implementations can override for batch optimization.
        
        Args:
            prompts: Iterable of input prompts
            **kwargs: Additional parameters
            
        Yields:
            Generated text completions
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the underlying model name."""
        ...


class ChatMessage(Dict[str, str]):
    """Typed alias for chat messages (role/content)."""

    role: str
    content: str


__all__ = ["LLMClient", "ChatMessage"]
