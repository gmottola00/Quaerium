"""PDF parsing utilities based on PyMuPDF."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import fitz  # type: ignore

from . import heading_detection, table_detection
from .normalizer import normalize_text

PageDict = Dict[str, Any]
BlockDict = Dict[str, Any]


def _validate_pdf_path(path: Union[str, Path]) -> Path:
    """Validate that the PDF path exists and points to a file."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    return pdf_path


def extract_pdf_layout(path: Union[str, Path]) -> List[PageDict]:
    """Extract text layout information from a PDF file using PyMuPDF.

    Args:
        path: Path to the PDF file.

    Returns:
        List of pages with extracted blocks (text, bbox, font info).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file.
        RuntimeError: If PyMuPDF fails to read the document.
    """
    pdf_path = _validate_pdf_path(path)

    try:
        doc = fitz.open(pdf_path.as_posix())
    except Exception as exc:  # pragma: no cover - passthrough
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    pages: List[PageDict] = []
    try:
        for page_index, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            raw_blocks = _extract_blocks_from_dict(text_dict)
            merged_blocks = _merge_label_value_blocks(raw_blocks)
            ordered_blocks = _sort_blocks_reading_order(merged_blocks)
            pages.append({"page_number": page_index, "blocks": ordered_blocks})
    finally:
        doc.close()

    return pages


def parse_pdf(
    path: Union[str, Path],
    detect_headings: bool = True,
    detect_tables: bool = True,
) -> List[PageDict]:
    """Parse a PDF file and optionally tag headings and tables.

    Args:
        path: Path to the PDF file.
        detect_headings: Whether to classify heading-like blocks. Defaults to True.
        detect_tables: Whether to detect tables via pdfplumber and heuristics. Defaults to True.

    Returns:
        A list of page dictionaries, each containing enriched blocks.
    """
    pages = extract_pdf_layout(path)

    if detect_headings:
        for page in pages:
            page["blocks"] = heading_detection.tag_headings(page.get("blocks", []))

    if detect_tables:
        pages = table_detection.integrate_tables(path, pages)

    pages = _remove_repeated_headers_footers(pages)
    pages = _drop_globally_repeated_blocks(pages)

    return pages


__all__ = ["extract_pdf_layout", "parse_pdf"]


# ---- Internal helpers -----------------------------------------------------


def _aggregate_bbox(bboxes: List[Tuple[float, float, float, float]]) -> List[float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def _merge_spans_into_lines(lines: List[Dict[str, Any]]) -> List[BlockDict]:
    merged_lines: List[BlockDict] = []
    for line in lines:
        spans = line.get("spans", [])
        texts: List[str] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        font_sizes: List[float] = []
        fonts: List[str] = []
        for span in spans:
            text = normalize_text(span.get("text", "")).strip()
            if not text:
                continue
            texts.append(text)
            bbox = span.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bboxes.append(tuple(bbox))  # type: ignore[arg-type]
            size = span.get("size")
            if isinstance(size, (int, float)):
                font_sizes.append(float(size))
            font = span.get("font")
            if isinstance(font, str):
                fonts.append(font)
        if not texts:
            continue
        font_name = None
        if fonts:
            font_name = Counter(fonts).most_common(1)[0][0]
        font_size = None
        if font_sizes:
            font_size = sum(font_sizes) / len(font_sizes)
        merged_lines.append(
            {
                "text": " ".join(texts),
                "bbox": _aggregate_bbox(bboxes) if bboxes else None,
                "font_size": font_size,
                "font_name": font_name,
                "type": "paragraph",
            }
        )
    return merged_lines


def _merge_lines_into_paragraphs(lines: List[BlockDict], line_gap_multiplier: float = 0.8) -> List[BlockDict]:
    paragraphs: List[BlockDict] = []
    current: List[BlockDict] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        texts = [b["text"] for b in current]
        bboxes = [tuple(b["bbox"]) for b in current if b.get("bbox")]  # type: ignore[arg-type]
        font_sizes = [b["font_size"] for b in current if isinstance(b.get("font_size"), (int, float))]
        font_names = [b["font_name"] for b in current if b.get("font_name")]
        paragraphs.append(
            {
                "text": " ".join(texts),
                "bbox": _aggregate_bbox(bboxes) if bboxes else None,
                "font_size": sum(font_sizes) / len(font_sizes) if font_sizes else None,
                "font_name": Counter(font_names).most_common(1)[0][0] if font_names else None,
                "type": current[0].get("type", "paragraph"),
            }
        )
        current = []

    for line in lines:
        if not current:
            current.append(line)
            continue
        prev = current[-1]
        prev_bbox = prev.get("bbox")
        line_bbox = line.get("bbox")
        if isinstance(prev_bbox, list) and isinstance(line_bbox, list):
            vertical_gap = line_bbox[1] - prev_bbox[3]
            font_size = prev.get("font_size") or line.get("font_size") or 10
            if 0 <= vertical_gap <= (font_size * line_gap_multiplier) and prev.get("font_name") == line.get("font_name"):
                current.append(line)
                continue
        flush_current()
        current.append(line)

    flush_current()
    return paragraphs


def _detect_list_item(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("-", "•", "–")) or stripped[:2].isdigit() and stripped[2:3] in {".", ")"}


def _extract_blocks_from_dict(text_dict: Dict[str, Any]) -> List[BlockDict]:
    blocks: List[BlockDict] = []
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        line_blocks = _merge_spans_into_lines(block.get("lines", []))
        # mark list items early
        for lb in line_blocks:
            if _detect_list_item(lb["text"]):
                lb["type"] = "list_item"
        paragraphs = _merge_lines_into_paragraphs(line_blocks)
        blocks.extend(paragraphs)
    return blocks


def _merge_label_value_blocks(blocks: List[BlockDict]) -> List[BlockDict]:
    """Merge label/value pairs split across lines (e.g., 'E-mail:' + next line)."""
    merged: List[BlockDict] = []
    skip_next = False
    for i, block in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue
        text = block.get("text", "")
        if isinstance(text, str) and text.rstrip().endswith(":") and i + 1 < len(blocks):
            next_block = blocks[i + 1]
            next_text = next_block.get("text", "")
            if isinstance(next_text, str):
                combined = f"{text} {next_text}".strip()
                merged_bbox: List[float] | None = None
                bboxes: List[Tuple[float, float, float, float]] = []
                for b in (block, next_block):
                    bbox = b.get("bbox")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bboxes.append(tuple(bbox))  # type: ignore[arg-type]
                if bboxes:
                    merged_bbox = _aggregate_bbox(bboxes)
                merged.append(
                    {
                        "text": combined,
                        "bbox": merged_bbox,
                        "font_size": block.get("font_size") or next_block.get("font_size"),
                        "font_name": block.get("font_name") or next_block.get("font_name"),
                        "type": block.get("type", "paragraph"),
                    }
                )
                skip_next = True
                continue
        merged.append(block)
    return merged


def _sort_blocks_reading_order(blocks: List[BlockDict]) -> List[BlockDict]:
    """Sort blocks primarily by vertical position, then horizontal, with stability.

    Two-column detection can reorder content unnaturally; instead, we sort by y
    (top), then x (left) to preserve proximity as seen in the document. This keeps
    subheadings close to their following paragraphs.
    """

    def sort_key(b: BlockDict) -> Tuple[float, float, int]:
        bbox = b.get("bbox", [0, 0, 0, 0])
        x0, y0 = 0.0, 0.0
        if isinstance(bbox, list) and len(bbox) == 4:
            x0, y0 = bbox[0], bbox[1]
        idx = b.get("_orig_idx", 0)
        return (y0, x0, idx)

    # Preserve original index for stability
    for i, b in enumerate(blocks):
        b["_orig_idx"] = i

    ordered = sorted(blocks, key=sort_key)
    merged = _merge_adjacent_blocks(ordered)

    for b in merged:
        b.pop("_orig_idx", None)
    return merged


def _merge_adjacent_blocks(blocks: List[BlockDict], gap_multiplier: float = 0.5) -> List[BlockDict]:
    """Merge adjacent blocks with same font/size that are vertically close."""
    merged: List[BlockDict] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        prev = merged[-1]
        prev_bbox = prev.get("bbox")
        curr_bbox = block.get("bbox")
        if (
            isinstance(prev_bbox, list)
            and isinstance(curr_bbox, list)
            and prev.get("font_name") == block.get("font_name")
            and prev.get("font_size") == block.get("font_size")
        ):
            vertical_gap = curr_bbox[1] - prev_bbox[3]
            font_size = block.get("font_size") or prev.get("font_size") or 10
            if 0 <= vertical_gap <= (font_size * gap_multiplier):
                prev["text"] = f"{prev.get('text', '')} {block.get('text', '')}".strip()
                prev["bbox"] = _aggregate_bbox([tuple(prev_bbox), tuple(curr_bbox)])  # type: ignore[arg-type]
                continue
        merged.append(block)
    return merged


def _remove_repeated_headers_footers(
    pages: List[PageDict],
    *,
    margin_ratio: float = 0.1,
    min_occurrences: int = 2,
) -> List[PageDict]:
    """Remove header/footer blocks repeated across pages using relative margins."""
    header_counter: Dict[str, int] = {}
    footer_counter: Dict[str, int] = {}

    # Count occurrences
    page_heights: List[float] = []
    for page in pages:
        max_y = 0.0
        for block in page.get("blocks", []):
            bbox = block.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                max_y = max(max_y, bbox[3])
        page_heights.append(max_y)

    for page, page_height in zip(pages, page_heights):
        if page_height <= 0:
            continue
        top_margin = margin_ratio * page_height
        bottom_margin = (1 - margin_ratio) * page_height
        for block in page.get("blocks", []):
            bbox = block.get("bbox")
            text = block.get("text")
            if not (isinstance(bbox, list) and isinstance(text, str)):
                continue
            key = f"{text}|{block.get('font_name')}|{round(block.get('font_size') or 0, 1)}"
            if bbox[1] <= top_margin:
                header_counter[key] = header_counter.get(key, 0) + 1
            if bbox[3] >= bottom_margin:
                footer_counter[key] = footer_counter.get(key, 0) + 1

    def is_repeated(block: BlockDict, counter: Dict[str, int], predicate) -> bool:
        bbox = block.get("bbox")
        text = block.get("text")
        if not (isinstance(bbox, list) and isinstance(text, str)):
            return False
        key = f"{text}|{block.get('font_name')}|{round(block.get('font_size') or 0, 1)}"
        return predicate(bbox) and counter.get(key, 0) >= min_occurrences

    cleaned_pages: List[PageDict] = []
    for page, page_height in zip(pages, page_heights):
        if page_height <= 0:
            cleaned_pages.append(page)
            continue
        top_margin = margin_ratio * page_height
        bottom_margin = (1 - margin_ratio) * page_height
        filtered_blocks: List[BlockDict] = []
        for block in page.get("blocks", []):
            if is_repeated(block, header_counter, lambda b: b[1] <= top_margin):
                continue
            if is_repeated(block, footer_counter, lambda b: b[3] >= bottom_margin):
                continue
            filtered_blocks.append(block)
        cleaned_pages.append({"page_number": page.get("page_number"), "blocks": filtered_blocks})
    return cleaned_pages


def _drop_globally_repeated_blocks(
    pages: List[PageDict],
    *,
    min_occurrences: int = 3,
) -> List[PageDict]:
    """Drop blocks whose text+bbox+font repeat many times across the document."""
    counter: Dict[str, int] = {}
    for page in pages:
        for block in page.get("blocks", []):
            key = _block_signature(block)
            if key:
                counter[key] = counter.get(key, 0) + 1

    cleaned_pages: List[PageDict] = []
    for page in pages:
        filtered: List[BlockDict] = []
        for block in page.get("blocks", []):
            key = _block_signature(block)
            if key and counter.get(key, 0) >= min_occurrences:
                continue
            filtered.append(block)
        cleaned_pages.append({"page_number": page.get("page_number"), "blocks": filtered})
    return cleaned_pages


def _block_signature(block: BlockDict) -> str | None:
    """Create a stable signature for a block using text and font attributes."""
    text = block.get("text")
    font = block.get("font_name")
    size = block.get("font_size")
    if not isinstance(text, str):
        return None
    font_key = (font or "").lower()
    size_key = f"{size:.1f}" if isinstance(size, (int, float)) else ""
    normalized_text = " ".join(text.split()).strip()
    return f"{normalized_text}|{font_key}|{size_key}"
