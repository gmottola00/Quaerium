"""Generic document ingestion service with dependency injection.

This service orchestrates document parsing, OCR, and language detection
without knowing about concrete implementations. All dependencies are injected.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from quaerium.core.ingestion.base import (
    DocumentParser,
    LanguageDetector,
    OCREngine,
    PageDict,
)
from quaerium.core.utils.file_utils import temporary_directory


class IngestionService:
    """High-level orchestrator for document ingestion.

    This service coordinates parsing, OCR, and language detection
    using injected dependencies (dependency injection pattern).
    
    Args:
        pdf_parser: Parser implementation for PDF files
        docx_parser: Parser implementation for DOCX files
        ocr_engine: Optional OCR engine for image-based PDFs
        lang_detector: Optional language detector for extracted text
        ocr_text_threshold: Minimum text length before skipping OCR
    """

    def __init__(
        self,
        pdf_parser: DocumentParser,
        docx_parser: DocumentParser,
        ocr_engine: Optional[OCREngine] = None,
        lang_detector: Optional[LanguageDetector] = None,
        ocr_text_threshold: int = 200,
    ) -> None:
        self.pdf_parser = pdf_parser
        self.docx_parser = docx_parser
        self.ocr_engine = ocr_engine
        self.lang_detector = lang_detector
        self.ocr_text_threshold = ocr_text_threshold

    def parse_document(
        self, 
        path: Path | str, 
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a document file (PDF or DOCX) into structured data.
        
        Args:
            path: Path to the document file
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            Dictionary containing:
                - doc_id: Unique document identifier
                - filename: Original filename
                - language: Detected language code
                - pages: List of parsed pages with blocks
                
        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            pages = self._parse_pdf(file_path)
        elif ext in {".docx", ".doc"}:
            pages = self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Detect language if detector is available
        language = self._detect_language(pages) if self.lang_detector else "unknown"
        
        return {
            "doc_id": doc_id or str(uuid.uuid4()),
            "filename": file_path.name,
            "extension": ext,
            "language": language,
            "pages": pages,
            "total_pages": len(pages),
        }

    def _parse_pdf(self, pdf_path: Path) -> List[PageDict]:
        """Parse PDF with optional OCR preprocessing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of parsed pages
        """
        # Check if OCR is needed and available
        if self.ocr_engine and self.ocr_engine.needs_ocr(pdf_path, self.ocr_text_threshold):
            # Use temporary directory for OCR output
            with temporary_directory() as tmp_dir:
                ocr_output = tmp_dir / f"ocr_{pdf_path.name}"
                self.ocr_engine.apply_ocr(pdf_path, ocr_output)
                return self.pdf_parser.parse(ocr_output)
        
        # Parse directly without OCR
        return self.pdf_parser.parse(pdf_path)

    def _parse_docx(self, docx_path: Path) -> List[PageDict]:
        """Parse DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            List of parsed pages
        """
        return self.docx_parser.parse(docx_path)

    def _detect_language(self, pages: List[PageDict]) -> str:
        """Detect language from parsed pages.
        
        Args:
            pages: List of parsed pages with text blocks
            
        Returns:
            ISO language code or 'unknown'
        """
        if not self.lang_detector:
            return "unknown"
        
        # Aggregate text from all pages
        texts: List[str] = []
        for page in pages:
            blocks = page.get("blocks", [])
            for block in blocks:
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
        
        aggregated_text = " ".join(texts)
        
        if not aggregated_text.strip():
            return "unknown"
        
        return self.lang_detector.detect(aggregated_text)


__all__ = ["IngestionService"]
