"""Rule-based heading detection for PDF blocks."""

from __future__ import annotations

import re
from statistics import median
from typing import Dict, List, Tuple

BlockDict = Dict[str, object]

ART_PATTERN = re.compile(r"^Art\.?\s+\d+.*", re.IGNORECASE)
CAPO_PATTERN = re.compile(r"^CAPO\s+[IVXLC]+.*", re.IGNORECASE)
CAPITOLO_PATTERN = re.compile(r"^Capitolo\s+\d+.*", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"^Titolo\s+\d+.*", re.IGNORECASE)
NUM_L1_PATTERN = re.compile(r"^\d+\.\s+.+")
NUM_L2_PATTERN = re.compile(r"^\d+\.\d+\.\s+.+")
ALL_CAPS_PATTERN = re.compile(r"^[A-Z0-9 ,.'’/\\-]+$")


def _font_threshold(blocks: List[BlockDict], boost: float) -> float:
    """Compute a heuristic threshold for heading font sizes."""
    sizes = [b.get("font_size") for b in blocks if isinstance(b.get("font_size"), (int, float))]
    if not sizes:
        return 0.0
    return median(sizes) + boost


def _is_bold(font_name: str | None) -> bool:
    """Heuristic to detect bold fonts from font name."""
    if not font_name:
        return False
    return any(token in font_name.lower() for token in ("bold", "black", "heavy", "semibold"))


def _relative_levels(blocks: List[BlockDict]) -> List[Tuple[float, int]]:
    """Map font sizes to relative levels (largest → level 1)."""
    sizes = sorted({b.get("font_size") for b in blocks if isinstance(b.get("font_size"), (int, float))}, reverse=True)
    levels: List[Tuple[float, int]] = []
    for idx, size in enumerate(sizes):
        levels.append((float(size), idx + 1))
    return levels


def _classify_block(
    text: str,
    font_size: float,
    font_name: str | None,
    size_threshold: float,
    size_levels: List[Tuple[float, int]],
) -> tuple[str, int | None]:
    """Classify a single block returning (type, level)."""
    block_type = "paragraph"
    level = None
    bold = _is_bold(font_name)

    if font_size is not None and font_size >= size_threshold > 0 and bold:
        block_type = "heading"
        level = 1

    if ART_PATTERN.match(text):
        block_type = "heading"
        level = 2
    elif CAPO_PATTERN.match(text) or CAPITOLO_PATTERN.match(text) or TITLE_PATTERN.match(text):
        block_type = "heading"
        level = 1
    elif NUM_L2_PATTERN.match(text):
        block_type = "heading"
        level = 2
    elif NUM_L1_PATTERN.match(text):
        block_type = "heading"
        level = 1
    elif ALL_CAPS_PATTERN.match(text) and len(text.split()) <= 8:
        block_type = "heading"
        level = 1
    elif bold and font_size > 0 and size_levels:
        # Assign level based on relative font size rank
        closest = min(size_levels, key=lambda lv: abs(font_size - lv[0]))
        block_type = "heading"
        level = closest[1]

    return block_type, level


def tag_headings(page_blocks: List[BlockDict], font_boost: float = 1.5) -> List[BlockDict]:
    """Assign heading tags to a list of page blocks.

    Args:
        page_blocks: Blocks extracted from a PDF page.
        font_boost: Additional size added to the median to mark headings.

    Returns:
        The same list of blocks with ``type`` and optional ``level`` keys added.
    """
    size_threshold = _font_threshold(page_blocks, boost=font_boost)
    size_levels = _relative_levels(page_blocks)

    for block in page_blocks:
        text = str(block.get("text", "")).strip()
        font_size = block.get("font_size")
        font_value = font_size if isinstance(font_size, (int, float)) else 0
        block_type, level = _classify_block(
            text,
            font_value,
            block.get("font_name"),
            size_threshold,
            size_levels,
        )

        block["type"] = block_type
        if level is not None:
            block["level"] = level

    return page_blocks


__all__ = ["tag_headings"]
