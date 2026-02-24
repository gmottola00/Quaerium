"""Factory for creating fully configured IngestionService with concrete implementations.

This factory is the main entry point for production code that needs document ingestion.
It wires together all concrete implementations from infra/ with the generic service from core/.
"""

from __future__ import annotations

from typing import Optional

from quaerium.core.ingestion.service import IngestionService
from quaerium.infra.parsers.docx.parser import PythonDocxParser
from quaerium.infra.parsers.pdf.ocr import TesseractOCREngine
from quaerium.infra.parsers.pdf.parser import PyMuPDFParser
from quaerium.infra.parsers.text.detector import FastTextLanguageDetector


def create_ingestion_service(
    *,
    enable_ocr: bool = True,
    detect_headings: bool = True,
    detect_tables: bool = True,
    ocr_text_threshold: int = 200,
    lang_model_path: Optional[str] = None,
    lang_max_chars: int = 5000,
) -> IngestionService:
    """Create a fully configured IngestionService with production implementations.
    
    This is the recommended way to instantiate IngestionService for production use.
    All concrete implementations are wired together here.
    
    Args:
        enable_ocr: Whether to enable OCR for image-based PDFs (default: True)
        detect_headings: Whether to detect heading structures in PDFs (default: True)
        detect_tables: Whether to detect tables in PDFs (default: True)
        ocr_text_threshold: Minimum text chars before skipping OCR (default: 200)
        lang_model_path: Optional path to fastText language model
        lang_max_chars: Max characters for language detection (default: 5000)
        
    Returns:
        Configured IngestionService ready to parse documents
        
    Example:
        >>> from quaerium.infra.parsers.factory import create_ingestion_service
        >>> service = create_ingestion_service(enable_ocr=True)
        >>> result = service.parse_document("document.pdf")
        >>> print(result["language"])
        'en'
    """
    # Create PDF parser
    pdf_parser = PyMuPDFParser(
        detect_headings=detect_headings,
        detect_tables=detect_tables,
    )
    
    # Create DOCX parser
    docx_parser = PythonDocxParser()
    
    # Create OCR engine (optional)
    ocr_engine = TesseractOCREngine() if enable_ocr else None
    
    # Create language detector
    lang_detector = FastTextLanguageDetector(
        model_path=lang_model_path,
        max_chars=lang_max_chars,
    )
    
    return IngestionService(
        pdf_parser=pdf_parser,
        docx_parser=docx_parser,
        ocr_engine=ocr_engine,
        lang_detector=lang_detector,
        ocr_text_threshold=ocr_text_threshold,
    )


def create_lightweight_ingestion_service() -> IngestionService:
    """Create a lightweight IngestionService without OCR or language detection.
    
    Useful for testing or when processing text-based documents only.
    
    Returns:
        Minimal IngestionService configuration
    """
    return IngestionService(
        pdf_parser=PyMuPDFParser(detect_headings=False, detect_tables=False),
        docx_parser=PythonDocxParser(),
        ocr_engine=None,
        lang_detector=None,
    )


__all__ = [
    "create_ingestion_service",
    "create_lightweight_ingestion_service",
]
