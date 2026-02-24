"""Abstract interfaces for document ingestion.

This module defines the protocol/interface layer for ingestion operations.
All concrete implementations should live in src/infra/parsers/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable


PageDict = Dict[str, Any]
"""A page is a dictionary with text, blocks, and metadata."""


@runtime_checkable
class DocumentParser(Protocol):
    """Abstract interface for parsing documents into structured pages.
    
    Implementations can be PDF parsers, DOCX parsers, or any other
    document format parser.
    """
    
    def parse(self, path: Path) -> List[PageDict]:
        """Parse a document file into a list of structured pages.
        
        Args:
            path: Path to the document file
            
        Returns:
            List of page dictionaries, each containing:
                - page_num: int
                - text: str (aggregated text)
                - blocks: List[Dict] (structured blocks)
                - metadata: Dict (optional)
        """
        ...


@runtime_checkable
class OCREngine(Protocol):
    """Abstract interface for OCR operations.
    
    Implementations can use Tesseract, cloud OCR services, or other engines.
    """
    
    def needs_ocr(self, input_path: Path, text_threshold: int = 200) -> bool:
        """Check if a document needs OCR based on existing text content.
        
        Args:
            input_path: Path to the document (usually PDF)
            text_threshold: Minimum text length to consider document as text-based
            
        Returns:
            True if OCR is needed, False otherwise
        """
        ...
    
    def apply_ocr(self, input_path: Path, output_path: Path) -> None:
        """Apply OCR to a document and save the result.
        
        Args:
            input_path: Path to the input document
            output_path: Path where to save the OCR-ed document
        """
        ...


@runtime_checkable
class LanguageDetector(Protocol):
    """Abstract interface for language detection.
    
    Implementations can use fastText, langdetect, or cloud services.
    """
    
    def detect(self, text: str) -> str:
        """Detect the language of a text string.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO language code (e.g., 'en', 'it', 'fr')
        """
        ...


@runtime_checkable
class HeadingDetector(Protocol):
    """Abstract interface for heading detection in documents.
    
    Can use heuristics, ML models, or rule-based systems.
    """
    
    def detect_headings(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify blocks as headings or regular text.
        
        Args:
            blocks: List of text blocks with formatting info
            
        Returns:
            Same blocks with added 'is_heading' boolean field
        """
        ...


@runtime_checkable
class TableDetector(Protocol):
    """Abstract interface for table detection in documents."""
    
    def detect_tables(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and extract tables from a page.
        
        Args:
            page_data: Page dictionary with layout information
            
        Returns:
            List of table dictionaries with extracted data
        """
        ...


__all__ = [
    "PageDict",
    "DocumentParser",
    "OCREngine",
    "LanguageDetector",
    "HeadingDetector",
    "TableDetector",
]
