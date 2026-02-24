"""PDF parser implementation wrapping PyMuPDF-based parsing functions.

This module provides a concrete implementation of the DocumentParser protocol
using PyMuPDF (fitz) for PDF extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from quaerium.core.ingestion.base import DocumentParser, PageDict

# Import the original parsing functions
from . import pdfplumber_parser as pdf_module


class PyMuPDFParser:
    """Concrete PDF parser using PyMuPDF.
    
    Implements the DocumentParser protocol from quaerium.core.ingestion.base.
    """
    
    def __init__(
        self,
        *,
        detect_headings: bool = True,
        detect_tables: bool = True,
    ) -> None:
        """Initialize the PDF parser.
        
        Args:
            detect_headings: Whether to detect and tag heading blocks
            detect_tables: Whether to detect and extract tables
        """
        self.detect_headings = detect_headings
        self.detect_tables = detect_tables
    
    def parse(self, path: Path) -> List[PageDict]:
        """Parse a PDF document into structured pages.
        
        Args:
            path: Path to the PDF file
            
        Returns:
            List of page dictionaries with extracted blocks and metadata
        """
        return pdf_module.parse_pdf(
            path,
            detect_headings=self.detect_headings,
            detect_tables=self.detect_tables,
        )


__all__ = ["PyMuPDFParser"]
