"""Document ingestion - Core abstractions and orchestration.

This module provides the abstract interfaces and generic orchestration logic
for document ingestion. Concrete implementations live in src.infra.parsers.
"""

from .base import (
    DocumentParser,
    HeadingDetector,
    LanguageDetector,
    OCREngine,
    PageDict,
    TableDetector,
)
from .service import IngestionService

__all__ = [
    "DocumentParser",
    "HeadingDetector",
    "LanguageDetector",
    "OCREngine",
    "PageDict",
    "TableDetector",
    "IngestionService",
]
