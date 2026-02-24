"""OCR helpers wrapping ocrmypdf."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

import fitz  # type: ignore

PDFPath = Union[str, Path]


def is_mostly_image_pdf(path: PDFPath, text_threshold: int = 200) -> bool:
    """Determine if a PDF likely requires OCR based on extracted text length."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    doc = fitz.open(pdf_path.as_posix())
    try:
        total_text_len = sum(len(page.get_text() or "") for page in doc)
    finally:
        doc.close()

    return total_text_len < text_threshold


def run_ocrmypdf(
    input_path: PDFPath,
    output_path: PDFPath,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    """Execute ``ocrmypdf`` with safe defaults."""
    cmd = [
        "ocrmypdf",
        "--deskew",
        "--optimize",
        "3",
        "--output-type",
        "pdf",
        str(input_path),
        str(output_path),
    ]
    if extra_args:
        cmd[1:1] = list(extra_args)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ocrmypdf executable not found. Please install ocrmypdf.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        raise RuntimeError(f"ocrmypdf failed with exit code {exc.returncode}") from exc


def ocr_if_needed(
    input_path: PDFPath,
    output_path: PDFPath,
    text_threshold: int = 200,
    force: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> bool:
    """Run OCR if the PDF is mostly images.

    Args:
        input_path: Source PDF path.
        output_path: Destination PDF path for OCR output.
        text_threshold: Minimum characters of existing text before skipping OCR.
        force: If ``True``, run OCR regardless of text density.
        extra_args: Additional arguments passed to ``ocrmypdf``.

    Returns:
        ``True`` if OCR was executed, otherwise ``False``.
    """
    needs_ocr = force or is_mostly_image_pdf(input_path, text_threshold=text_threshold)
    if not needs_ocr:
        return False

    run_ocrmypdf(input_path, output_path, extra_args=extra_args)
    return True


__all__ = ["is_mostly_image_pdf", "run_ocrmypdf", "ocr_if_needed"]
