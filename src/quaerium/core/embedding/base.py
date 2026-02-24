"""Base Protocol for embedding clients."""

from __future__ import annotations

from typing import List, Protocol, Sequence


class EmbeddingClient(Protocol):
    """Protocol for embedding providers.
    
    Uses Protocol instead of ABC for more flexible duck typing.
    Any class implementing these methods can be used as an EmbeddingClient.
    """

    def embed(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Vector representation as list of floats
        """
        ...

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts.
        
        Default implementation iterates embed() for each text.
        Implementations can override for batch optimization.
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            List of vector representations
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the underlying embedding model name."""
        ...

    @property
    def dimension(self) -> int | None:
        """Optionally return embedding dimension if known."""
        ...


__all__ = ["EmbeddingClient"]
