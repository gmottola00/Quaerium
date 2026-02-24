"""Reranker implementations for hybrid search."""

from __future__ import annotations

from typing import Dict, List

from quaerium.core.index.base import Reranker
from quaerium.core.llm import LLMClient


class LLMReranker(Reranker):
    """LLM-based reranker (lightweight, heuristic)."""

    def __init__(self, llm: LLMClient, max_context: int = 2000) -> None:
        self.llm = llm
        self.max_context = max_context

    def rerank(self, query: str, candidates: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
        """Rerank by prompting the LLM with concatenated candidates (lightweight fallback)."""
        # Build a compact prompt
        snippets = []
        for idx, c in enumerate(candidates[: top_k * 2]):  # consider a buffer
            text = str(c.get("text", ""))[: self.max_context]
            snippets.append(f"[{idx}] {text}")
        prompt = (
            "Ordina i seguenti passaggi per rilevanza rispetto alla domanda.\n"
            f"Domanda: {query}\n"
            "Passaggi:\n"
            "\n".join(snippets)
            + "\nRestituisci gli indici in ordine di rilevanza separati da virgola:"
        )
        ranking_str = self.llm.generate(prompt)
        order = _parse_ordering(ranking_str)
        ordered = []
        for idx in order:
            if 0 <= idx < len(candidates):
                ordered.append(candidates[idx])
        # Fill if missing
        for c in candidates:
            if c not in ordered:
                ordered.append(c)
        return ordered[:top_k]


def _parse_ordering(raw: str) -> List[int]:
    out: List[int] = []
    parts = raw.replace("[", "").replace("]", "").split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


__all__ = ["LLMReranker"]
