"""OpenAI-based LLM client."""

from __future__ import annotations

import os
from typing import Any, Iterable

from quaerium.core.llm.base import LLMClient

try:
    import openai
except ImportError as exc:  # pragma: no cover - optional dependency
    openai = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAILLMClient(LLMClient):
    """Client for OpenAI chat completions."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = DEFAULT_OPENAI_API_KEY,
        base_url: str | None = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        if _import_error:
            raise ImportError("openai is required for OpenAILLMClient") from _import_error
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def generate_batch(self, prompts: Iterable[str], **kwargs: Any) -> Iterable[str]:
        for prompt in prompts:
            yield self.generate(prompt, **kwargs)


__all__ = ["OpenAILLMClient"]
