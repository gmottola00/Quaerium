"""Ollama-based LLM client."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

from quaerium.core.llm.base import LLMClient

try:
    import requests
except ImportError as exc:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")


class OllamaLLMClient(LLMClient):
    """Client for Ollama's /api/generate endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 120,
    ) -> None:
        if _import_error:
            raise ImportError("requests is required for OllamaLLMClient") from _import_error
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload: Dict[str, Any] = {"model": self._model, "prompt": prompt, "stream": False}
        payload.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def generate_batch(self, prompts: Iterable[str], **kwargs: Any) -> Iterable[str]:
        for prompt in prompts:
            yield self.generate(prompt, **kwargs)


__all__ = ["OllamaLLMClient"]
