"""Query rewriter using an LLM."""

from __future__ import annotations

from typing import Dict

from quaerium.core.llm import LLMClient


class QueryRewriter:
    """Simple query rewriting using LLM prompts."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, query: str, metadata_hint: Dict[str, str] | None = None) -> str:
        """Rewrite/clarify the query optionally using metadata hints."""
        meta_part = ""
        if metadata_hint:
            meta_items = ", ".join(f"{k}: {v}" for k, v in metadata_hint.items())
            meta_part = f"Context: {meta_items}\n"
        prompt = (
            "Riscrivi la query per la ricerca documentale, includendo eventuali riferimenti a lotto/tender se presenti.\n"
            f"{meta_part}"
            f"Query: {query}\n"
            "Query riscritta:"
        )
        return self.llm.generate(prompt)


__all__ = ["QueryRewriter"]
