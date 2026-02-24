"""Dynamic chunk builder based on heading hierarchy."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

from quaerium.core.chunking.models import Chunk


class DynamicChunker:
    """Create chunks using level-1 headings as anchors."""

    def __init__(
        self,
        *,
        include_tables: bool = True,
        max_heading_level: int = 6,
        allow_preamble: bool = False,
    ) -> None:
        """Initialize the chunker.

        Args:
            include_tables: Whether to include table blocks.
            max_heading_level: Maximum heading level to include within a chunk.
            allow_preamble: If True, content before the first level-1 heading is grouped
                into a preamble chunk; otherwise it is skipped.
        """
        self.include_tables = include_tables
        self.max_heading_level = max_heading_level
        self.allow_preamble = allow_preamble

    def build_chunks(self, pages: Sequence[Dict[str, Any]]) -> List[Chunk]:
        """Build chunks from parsed pages.

        A chunk starts at a heading with level 1 and includes:
        - all subsequent headings with level > 1 (up to max_heading_level)
        - paragraphs/list/table blocks until the next level-1 heading (excluded).

        Args:
            pages: Parsed pages with blocks (as produced by the ingestion parser).

        Returns:
            List of chunks preserving document order.
        """
        chunks: List[Chunk] = []
        current_blocks: List[Dict[str, Any]] = []
        current_title: Optional[str] = None
        current_level: Optional[int] = None

        def finalize() -> None:
            nonlocal current_blocks, current_title, current_level
            if not current_blocks or current_title is None or current_level is None:
                current_blocks = []
                current_title = None
                current_level = None
                return
            page_numbers = _collect_page_numbers(current_blocks)
            text_blocks = _visible_blocks(current_blocks)
            text = "\n".join(
                b["text"] for b in text_blocks if isinstance(b.get("text"), str)
            ).strip()
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    title=current_title,
                    heading_level=current_level,
                    text=text,
                    blocks=current_blocks,
                    page_numbers=page_numbers,
                )
            )
            current_blocks = []
            current_title = None
            current_level = None

        for page_idx, page in enumerate(pages, start=1):
            blocks = page.get("blocks", [])
            for block in blocks:
                block.setdefault("page_number", page_idx)
                block_type = block.get("type")
                level = block.get("level")

                is_heading = block_type == "heading" and isinstance(level, int)
                is_table = block_type == "table_block"

                if is_table and not self.include_tables:
                    continue

                if is_heading and level == 1:
                    finalize()
                    current_title = str(block.get("text", "")).strip()
                    current_level = level
                    current_blocks = [block]
                    continue

                if current_title is None:
                    if self.allow_preamble:
                        current_title = "Preamble"
                        current_level = 0
                        current_blocks = [block]
                    continue

                if is_heading and level is not None and 1 < level <= self.max_heading_level:
                    current_blocks.append(block)
                    continue

                current_blocks.append(block)

        finalize()
        return chunks


def _collect_page_numbers(blocks: List[Dict[str, Any]]) -> List[int]:
    nums = []
    for block in blocks:
        page_number = block.get("page_number")
        if isinstance(page_number, int):
            nums.append(page_number)
    return sorted(list(dict.fromkeys(nums)))


def _visible_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exclude the first level-1 heading text to avoid duplication in chunk body."""
    filtered: List[Dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        if idx == 0 and block.get("type") == "heading" and block.get("level") == 1:
            continue
        filtered.append(block)
    return filtered


__all__ = ["DynamicChunker"]
