"""Data models for RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetrievedChunk:
    """Chunk retrieved from the vector/keyword store."""

    id: str
    text: str
    section_path: Optional[str]
    metadata: Dict[str, str]
    page_numbers: List[int]
    source_chunk_id: Optional[str]
    score: float | None = None


@dataclass
class RagResponse:
    """Structured RAG response with citations."""

    answer: str
    citations: List[RetrievedChunk]


__all__ = ["RetrievedChunk", "RagResponse"]
