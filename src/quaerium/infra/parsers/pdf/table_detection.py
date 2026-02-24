"""Table detection utilities for PDF documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

BlockDict = Dict[str, Any]
PageDict = Dict[str, Any]


def table_to_block(table_cells: Sequence[Sequence[str | None]]) -> BlockDict:
    """Convert table cells to a normalized block dictionary."""
    text_rows = [" | ".join(cell or "" for cell in row) for row in table_cells]
    text = "\n".join(text_rows).strip()
    return {"type": "table_block", "text": text, "raw_cells": [list(row) for row in table_cells]}


def extract_pdf_tables(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract tables from a PDF via pdfplumber.

    Args:
        path: Path to the PDF file.

    Returns:
        List of dictionaries with ``page_number`` and ``block`` keys.
    """
    if pdfplumber is None:
        return []

    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    table_blocks: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path.as_posix()) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                table_blocks.append({"page_number": page_index, "block": table_to_block(table)})
    return table_blocks


def _is_table_like_text(text: str) -> bool:
    """Heuristic to decide if a block of text represents a table."""
    if not text:
        return False

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    if "|" in text or "\t" in text:
        return True

    column_counts = []
    for line in lines:
        segments = [seg for seg in re.split(r"\s{2,}", line) if seg]
        column_counts.append(len(segments))

    return any(count >= 2 for count in column_counts) and (
        sum(column_counts) / len(column_counts) > 1.5
    )


def flag_table_like_blocks(blocks: List[BlockDict]) -> List[BlockDict]:
    """Mark blocks that appear to be table-like based on heuristics."""
    for block in blocks:
        text = str(block.get("text", "")).strip()
        if block.get("type") == "table_block":
            continue
        if _is_table_like_text(text):
            block["type"] = "table_block"
    return blocks


def integrate_tables(path: Union[str, Path], pages: List[PageDict]) -> List[PageDict]:
    """Append detected tables and flag table-like blocks on provided pages.

    Args:
        path: Path to the source PDF (used for pdfplumber extraction).
        pages: Parsed pages from PyMuPDF.

    Returns:
        Pages enriched with table blocks.
    """
    plumber_tables = extract_pdf_tables(path)
    tables_by_page: Dict[int, List[BlockDict]] = {}
    for table in plumber_tables:
        tables_by_page.setdefault(table["page_number"], []).append(table["block"])

    enriched_pages: List[PageDict] = []
    for page in pages:
        page_number = page.get("page_number")
        blocks = flag_table_like_blocks(list(page.get("blocks", [])))
        for table_block in tables_by_page.get(page_number, []):
            blocks.append(table_block)
        enriched_pages.append({"page_number": page_number, "blocks": blocks})

    return enriched_pages


__all__ = ["extract_pdf_tables", "table_to_block", "flag_table_like_blocks", "integrate_tables"]
