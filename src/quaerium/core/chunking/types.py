"""Core chunking types used across the application.

This module defines Protocol-based abstractions for chunking functionality,
allowing domain layers to extend with specific implementations while maintaining
type safety and flexibility.

The Protocol pattern (PEP 544) enables structural subtyping without inheritance,
allowing implementations to add domain-specific fields while conforming to the
core interface contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class ChunkLike(Protocol):
    """Protocol defining the interface for document chunks.

    This Protocol allows domain layers to implement their own chunk types
    with additional fields while maintaining compatibility with core chunking logic.

    Example domain implementation:
        ```python
        @dataclass
        class TenderChunk:
            id: str
            title: str
            heading_level: int
            text: str
            blocks: List[Dict[str, Any]]
            page_numbers: List[int]
            # Domain-specific fields
            tender_id: str
            lot_id: Optional[str]
            section_type: str

            def to_dict(self, *, include_blocks: bool = True) -> Dict[str, Any]:
                data = {
                    "id": self.id,
                    "title": self.title,
                    "heading_level": self.heading_level,
                    "text": self.text,
                    "page_numbers": self.page_numbers,
                    "tender_id": self.tender_id,
                    "lot_id": self.lot_id,
                    "section_type": self.section_type,
                }
                if include_blocks:
                    data["blocks"] = self.blocks
                return data
        ```

    Attributes:
        id: Unique identifier for the chunk
        title: Section title or heading
        heading_level: Hierarchical level of the heading (e.g., h1=1, h2=2)
        text: The actual text content of the chunk
        blocks: List of structured text blocks within this chunk
        page_numbers: List of page numbers where this chunk appears
    """

    id: str
    title: str
    heading_level: int
    text: str
    blocks: List[Dict[str, Any]]
    page_numbers: List[int]

    def to_dict(self, *, include_blocks: bool = True) -> Dict[str, Any]:
        """Convert chunk to dictionary representation.

        Args:
            include_blocks: Whether to include the blocks field in the output

        Returns:
            Dictionary containing at minimum: id, title, heading_level, text,
            page_numbers. Implementations may include additional fields.
        """
        ...


@runtime_checkable
class TokenChunkLike(Protocol):
    """Protocol defining the interface for token-optimized chunks.

    Token chunks are optimized for embedding and retrieval operations,
    typically derived from larger document chunks with enhanced metadata.

    Example domain implementation:
        ```python
        @dataclass
        class TenderTokenChunk:
            id: str
            text: str
            section_path: str
            metadata: Dict[str, str]
            page_numbers: List[int]
            source_chunk_id: str
            # Domain-specific fields
            tender_id: str
            lot_id: Optional[str]
            section_type: str

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "id": self.id,
                    "text": self.text,
                    "section_path": self.section_path,
                    "metadata": self.metadata,
                    "page_numbers": self.page_numbers,
                    "source_chunk_id": self.source_chunk_id,
                    "tender_id": self.tender_id,
                    "lot_id": self.lot_id,
                    "section_type": self.section_type,
                }
        ```

    Attributes:
        id: Unique identifier for the token chunk
        text: The token-optimized text content
        section_path: Hierarchical path to the section (e.g., "Section 1 > Subsection A")
        page_numbers: List of page numbers where this chunk appears
        metadata: Additional metadata as key-value pairs
        source_chunk_id: Reference to the original Chunk this was derived from
    """

    id: str
    text: str
    section_path: str
    metadata: Dict[str, str]
    page_numbers: List[int]
    source_chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert token chunk to dictionary representation.

        Returns:
            Dictionary containing at minimum: id, text, section_path, metadata,
            page_numbers, source_chunk_id. Implementations may include additional fields.
        """
        ...


# Legacy compatibility - users can still import these names
Chunk = ChunkLike
TokenChunk = TokenChunkLike

__all__ = ["ChunkLike", "TokenChunkLike", "Chunk", "TokenChunk"]
