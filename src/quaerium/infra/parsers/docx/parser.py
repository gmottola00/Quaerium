"""DOCX parser implementation using python-docx.

This module provides a concrete implementation of the DocumentParser protocol
for Microsoft Word documents (.docx).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from quaerium.core.ingestion.base import DocumentParser, PageDict

# Import original parsing functions
from . import python_docx_parser as docx_module


class PythonDocxParser:
    """Concrete DOCX parser using python-docx library.
    
    Implements the DocumentParser protocol from quaerium.core.ingestion.base.
    """
    
    def parse(self, path: Path) -> List[PageDict]:
        """Parse a DOCX document into structured pages.
        
        Args:
            path: Path to the DOCX file
            
        Returns:
            List with a single page containing all blocks and tables
        """
        result = docx_module.parse_docx(path)
        
        # Convert to PageDict format
        blocks = result.get("blocks", [])
        tables = result.get("tables", [])
        
        # Merge tables as special blocks
        all_blocks = blocks + tables
        
        return [{
            "page_number": 1,
            "blocks": all_blocks,
        }]


__all__ = ["PythonDocxParser"]
