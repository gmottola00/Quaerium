"""OpenAI embedding client."""

from __future__ import annotations

import os
from typing import List, Sequence

from quaerium.core.embedding.base import EmbeddingClient

try:
    import openai
except ImportError as exc:  # pragma: no cover - optional dependency
    openai = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

DEFAULT_OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeddingClient(EmbeddingClient):
    """Client for OpenAI embeddings."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_EMBED_MODEL,
        api_key: str | None = DEFAULT_OPENAI_API_KEY,
        base_url: str | None = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        if _import_error:
            raise ImportError("openai is required for OpenAIEmbeddingClient") from _import_error
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self._model, input=[text])
        vector = resp.data[0].embedding
        return vector

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI batch API.

        Uses OpenAI's native batch embedding endpoint for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)

        Example:
            >>> client = OpenAIEmbeddingClient(model="text-embedding-3-small")
            >>> embeddings = client.embed_batch(["text1", "text2", "text3"])
            >>> len(embeddings)
            3
        """
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=list(texts),
            )
            return [item.embedding for item in response.data]
        except Exception:
            # Fallback to sequential if batch fails
            return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int | None:
        # OpenAI embeddings include dimensions in response; could be cached.
        return None


__all__ = ["OpenAIEmbeddingClient"]
