"""Context assembler to fit within token budget."""

from __future__ import annotations

from typing import List

from quaerium.rag.models import RetrievedChunk


class ContextAssembler:
    """Assemble context respecting token budget (approx via word count)."""

    def __init__(self, max_tokens: int = 2000) -> None:
        self.max_tokens = max_tokens

    def assemble(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Return a subset of chunks fitting the token budget."""
        selected: List[RetrievedChunk] = []
        remaining = self.max_tokens
        for chunk in chunks:
            tokens = len(chunk.text.split())
            if tokens > remaining and selected:
                break
            selected.append(chunk)
            remaining -= tokens
        return selected


__all__ = ["ContextAssembler"]
