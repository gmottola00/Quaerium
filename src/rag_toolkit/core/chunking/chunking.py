"""Token-based chunking strategy with metadata enrichment."""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Sequence

from rag_toolkit.core.chunking.models import Chunk, TokenChunk

DEFAULT_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
DEFAULT_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "400"))
DEFAULT_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "120"))


def default_tokenizer(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


class TokenChunker:
    """Chunk text by tokens with overlap and metadata extraction."""

    def __init__(
        self,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP,
        tokenizer: Callable[[str], List[str]] = default_tokenizer,
    ) -> None:
        if min_tokens <= 0 or max_tokens <= 0 or overlap_tokens < 0:
            raise ValueError("Token sizes must be positive, overlap non-negative")
        if min_tokens > max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer

    def chunk(self, structured_chunks: Sequence[Chunk]) -> List[TokenChunk]:
        """Chunk higher-level sections into token-based chunks.

        Args:
            structured_chunks: Chunks produced by ``DynamicChunker``.

        Returns:
            List of token chunks with metadata and section path.
        """
        token_chunks: List[TokenChunk] = []
        for chunk in structured_chunks:
            tokens = self.tokenizer(chunk.text)
            spans = _build_spans(tokens, self.max_tokens, self.overlap_tokens, self.min_tokens)
            section_path = _derive_section_path(chunk)
            metadata = _extract_metadata(chunk.text, chunk.title)
            for start, end in spans:
                piece = " ".join(tokens[start:end]).strip()
                if not piece:
                    continue
                token_chunks.append(
                    TokenChunk(
                        id=f"{chunk.id}:{start}-{end}",
                        text=piece,
                        section_path=section_path,
                        metadata=metadata,
                        page_numbers=chunk.page_numbers,
                        source_chunk_id=chunk.id,
                    )
                )
        return token_chunks


def _build_spans(tokens: List[str], max_tokens: int, overlap: int, min_tokens: int) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    start = 0
    n = len(tokens)
    if n == 0:
        return spans
    while start < n:
        end = min(start + max_tokens, n)
        if end - start < min_tokens and start != 0:
            break
        spans.append((start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return spans


def _derive_section_path(chunk: Chunk) -> str:
    """Compose a section path from chunk title and nested headings inside."""
    parts: List[str] = []
    if chunk.title:
        parts.append(chunk.title.strip())
    for block in chunk.blocks:
        if block.get("type") == "heading":
            text = str(block.get("text", "")).strip()
            if text and text not in parts:
                parts.append(text)
    return " > ".join(parts)


# TODO: rendere generali le estrazioni di metadati
TENDER_CODE_PATTERN = re.compile(r"\b\d{6}-\d{4}\b")
LOT_ID_PATTERN = re.compile(r"\bLOT[-_\s]?\w+\b", re.IGNORECASE)
DOC_TYPE_KEYWORDS = {
    "bando": "tender_notice",
    "avviso": "notice",
    "rettifica": "corrigendum",
    "capitolato": "specs",
    "disciplinare": "disciplinare",
}


def _extract_metadata(text: str, title: str | None = None) -> Dict[str, str]:
    """Heuristic metadata extraction from text/title."""
    metadata: Dict[str, str] = {}
    merged_text = f"{title or ''} {text}".lower()

    tender_match = TENDER_CODE_PATTERN.search(merged_text)
    if tender_match:
        metadata["tender_code"] = tender_match.group(0)

    lot_match = LOT_ID_PATTERN.search(merged_text)
    if lot_match:
        metadata["lot_id"] = lot_match.group(0).lower()

    for keyword, doc_type in DOC_TYPE_KEYWORDS.items():
        if keyword in merged_text:
            metadata["document_type"] = doc_type
            break

    if "art." in merged_text or "articolo" in merged_text:
        metadata["clause_type"] = "article"

    return metadata


__all__ = ["TokenChunker"]
