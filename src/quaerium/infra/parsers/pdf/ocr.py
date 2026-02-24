"""OCR engine implementation using Tesseract via ocrmypdf.

This module provides a concrete implementation of the OCREngine protocol
using Tesseract OCR through the ocrmypdf wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from quaerium.core.ingestion.base import OCREngine

# Import original OCR functions
from . import tesseract_ocr as ocr_module


class TesseractOCREngine:
    """Concrete OCR implementation using Tesseract (via ocrmypdf).
    
    Implements the OCREngine protocol from quaerium.core.ingestion.base.
    """
    
    def __init__(
        self,
        *,
        extra_args: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the Tesseract OCR engine.
        
        Args:
            extra_args: Additional arguments to pass to ocrmypdf
        """
        self.extra_args = extra_args
    
    def needs_ocr(self, input_path: Path, text_threshold: int = 200) -> bool:
        """Check if a PDF needs OCR based on text density.
        
        Args:
            input_path: Path to the PDF file
            text_threshold: Minimum text length to consider as text-based
            
        Returns:
            True if OCR is needed, False otherwise
        """
        return ocr_module.is_mostly_image_pdf(input_path, text_threshold=text_threshold)
    
    def apply_ocr(self, input_path: Path, output_path: Path) -> None:
        """Apply OCR to a PDF and save the result.
        
        Args:
            input_path: Path to the input PDF
            output_path: Path where to save the OCR-ed PDF
        """
        ocr_module.run_ocrmypdf(input_path, output_path, extra_args=self.extra_args)


__all__ = ["TesseractOCREngine"]
