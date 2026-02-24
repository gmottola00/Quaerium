"""Concrete implementations of chunking protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    """Standard implementation of ChunkLike protocol."""

    id: str
    title: str
    heading_level: int
    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)

    def to_dict(self, *, include_blocks: bool = True) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        data = {
            "id": self.id,
            "title": self.title,
            "heading_level": self.heading_level,
            "text": self.text,
            "page_numbers": self.page_numbers,
        }
        if include_blocks:
            data["blocks"] = self.blocks
        return data


@dataclass
class TokenChunk:
    """Standard implementation of TokenChunkLike protocol."""

    id: str
    text: str
    section_path: str
    metadata: Dict[str, str] = field(default_factory=dict)
    page_numbers: List[int] = field(default_factory=list)
    source_chunk_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert token chunk to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "section_path": self.section_path,
            "metadata": self.metadata,
            "page_numbers": self.page_numbers,
            "source_chunk_id": self.source_chunk_id,
        }


__all__ = ["Chunk", "TokenChunk"]
