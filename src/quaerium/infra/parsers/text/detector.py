"""Language detection implementation using fastText or lingua-py.

This module provides a concrete implementation of the LanguageDetector protocol
using fastText or lingua-py as fallback.
"""

from __future__ import annotations

from typing import Optional

from quaerium.core.ingestion.base import LanguageDetector

# Import original language detection functions
from . import fasttext_detector as lang_module


class FastTextLanguageDetector:
    """Concrete language detector using fastText (with lingua fallback).
    
    Implements the LanguageDetector protocol from quaerium.core.ingestion.base.
    """
    
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        max_chars: int = 5000,
    ) -> None:
        """Initialize the language detector.
        
        Args:
            model_path: Path to fastText model file (defaults to models/lid.176.bin)
            max_chars: Maximum characters to use for detection
        """
        self.model_path = model_path
        self.max_chars = max_chars
    
    def detect(self, text: str) -> str:
        """Detect the language of a text string.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO-639-1 language code (e.g., 'en', 'it', 'fr') or 'unknown'
        """
        return lang_module.detect_language(
            text,
            max_chars=self.max_chars,
            model_path=self.model_path,
        )


__all__ = ["FastTextLanguageDetector"]
