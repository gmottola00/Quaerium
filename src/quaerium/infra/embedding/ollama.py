"""Ollama embedding client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

from quaerium.core.embedding.base import EmbeddingClient

try:
    import requests
except ImportError as exc:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class OllamaEmbeddingClient(EmbeddingClient):
    """Client for Ollama's /api/embed endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 120,
    ) -> None:
        if _import_error:
            raise ImportError("requests is required for OllamaEmbeddingClient") from _import_error
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> List[float]:
        # Ollama >= 0.1.34 uses /api/embed with "input" key
        payload: Dict[str, Any] = {"model": self._model, "input": text}
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle different shapes: {"embedding": [...]}, {"data": [{"embedding": [...]}]}, {"embeddings": [...]}
        vector = data.get("embedding") or data.get("vector")
        if vector is None and isinstance(data.get("data"), list) and data["data"]:
            vector = data["data"][0].get("embedding")
        if vector is None and isinstance(data.get("embeddings"), list) and data["embeddings"]:
            vector = data["embeddings"][0]

        normalized = _normalize_embedding(vector)
        if normalized is None:
            raise ValueError(f"Invalid embedding response from Ollama: keys={list(data.keys())}")
        return normalized

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple texts.

        Note: Ollama doesn't have native batch API, so this calls
        embed() for each text. Override in subclass if batch API exists.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)

        Example:
            >>> client = OllamaEmbeddingClient()
            >>> embeddings = client.embed_batch(["text1", "text2", "text3"])
            >>> len(embeddings)
            3
        """
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int | None:
        """Return embedding dimension if known."""
        return None


def _normalize_embedding(raw: Any) -> List[float] | None:
    """Normalize embedding payload into a flat list of floats."""
    if raw is None:
        return None
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        else:
            # allow empty list or flat list
            pass
    if isinstance(raw, (list, tuple)):
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None
    return None


__all__ = ["OllamaEmbeddingClient"]
