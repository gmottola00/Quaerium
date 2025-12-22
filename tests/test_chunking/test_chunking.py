"""Test chunking strategies."""

from __future__ import annotations

import pytest

from rag_toolkit.core.chunking.chunking import TokenChunker, default_tokenizer
from rag_toolkit.core.chunking.dynamic_chunker import DynamicChunker
from rag_toolkit.core.chunking.models import Chunk
from rag_toolkit.core.chunking.types import ChunkLike


def test_dynamic_chunker_initialization():
    """Test DynamicChunker initialization with various parameters."""
    # Default parameters
    chunker = DynamicChunker()
    assert chunker.include_tables is True
    assert chunker.max_heading_level == 6
    assert chunker.allow_preamble is False
    
    # Custom parameters
    chunker = DynamicChunker(
        include_tables=False,
        max_heading_level=3,
        allow_preamble=True,
    )
    assert chunker.include_tables is False
    assert chunker.max_heading_level == 3
    assert chunker.allow_preamble is True


def test_dynamic_chunker_build_chunks():
    """Test DynamicChunker with sample parsed pages."""
    chunker = DynamicChunker()
    
    # Sample parsed pages structure (simplified)
    pages = [
        {
            "page_number": 1,
            "blocks": [
                {"type": "heading", "level": 1, "text": "Introduction", "page_number": 1},
                {"type": "paragraph", "text": "This is the introduction.", "page_number": 1},
            ],
        },
        {
            "page_number": 2,
            "blocks": [
                {"type": "heading", "level": 2, "text": "Subsection", "page_number": 2},
                {"type": "paragraph", "text": "More details here.", "page_number": 2},
                {"type": "heading", "level": 1, "text": "Conclusion", "page_number": 2},
                {"type": "paragraph", "text": "Final thoughts.", "page_number": 2},
            ],
        },
    ]
    
    chunks = chunker.build_chunks(pages)
    
    # Should create chunks based on level-1 headings
    assert len(chunks) >= 2  # At least Introduction and Conclusion
    # Chunks should be Chunk instances
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.id for c in chunks)  # All have IDs
    assert all(c.text for c in chunks)  # All have text


def test_token_chunker_initialization():
    """Test TokenChunker initialization and validation."""
    # Default parameters
    chunker = TokenChunker()
    assert chunker.max_tokens > 0
    assert chunker.min_tokens > 0
    assert chunker.overlap_tokens >= 0
    assert chunker.min_tokens <= chunker.max_tokens
    
    # Custom parameters
    chunker = TokenChunker(
        max_tokens=1000,
        min_tokens=500,
        overlap_tokens=100,
    )
    assert chunker.max_tokens == 1000
    assert chunker.min_tokens == 500
    assert chunker.overlap_tokens == 100
    
    # Validation errors
    with pytest.raises(ValueError, match="Token sizes must be positive"):
        TokenChunker(max_tokens=0)
    
    with pytest.raises(ValueError, match="Token sizes must be positive"):
        TokenChunker(min_tokens=-1)
    
    with pytest.raises(ValueError, match="min_tokens cannot exceed max_tokens"):
        TokenChunker(min_tokens=1000, max_tokens=500)
    
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        TokenChunker(max_tokens=500, overlap_tokens=500)


def test_default_tokenizer():
    """Test the default whitespace tokenizer."""
    text = "This is a simple test sentence."
    tokens = default_tokenizer(text)
    
    assert len(tokens) == 6
    assert tokens[0] == "This"
    assert tokens[-1] == "sentence."
    
    # Empty text
    assert default_tokenizer("") == []
    
    # Multiple spaces
    tokens = default_tokenizer("  multiple   spaces  ")
    assert all(t for t in tokens)  # No empty tokens


def test_token_chunker_basic():
    """Test TokenChunker with simple structured chunks."""
    chunker = TokenChunker(max_tokens=50, min_tokens=20, overlap_tokens=10)
    
    # Create a simple Chunk using test implementation
    sample_chunk = Chunk(
        id="test-chunk-1",
        title="Test Section",
        heading_level=1,
        text="This is a test chunk with some text. " * 20,  # Repeat to ensure multiple token chunks
        blocks=[],
        page_numbers=[1],
    )
    
    token_chunks = chunker.chunk([sample_chunk])
    
    # Should create at least one token chunk
    assert len(token_chunks) > 0
    
    # Verify token chunk properties
    for tc in token_chunks:
        assert tc.id.startswith(sample_chunk.id)
        assert tc.text
        assert tc.source_chunk_id == sample_chunk.id
        assert isinstance(tc.page_numbers, list)
        
        # Token count should be reasonable
        token_count = len(default_tokenizer(tc.text))
        assert token_count <= chunker.max_tokens


def test_token_chunker_with_metadata():
    """Test TokenChunker metadata extraction."""
    chunker = TokenChunker(max_tokens=100, min_tokens=50, overlap_tokens=20)
    
    # Chunk with metadata patterns
    sample_chunk = Chunk(
        id="test-chunk-2",
        title="Tender ABC123",
        heading_level=1,
        text="Tender code ABC123 for lot LOT-001. This is a sample document.",
        blocks=[],
        page_numbers=[1, 2],
    )
    
    token_chunks = chunker.chunk([sample_chunk])
    
    assert len(token_chunks) > 0
    
    # Check that metadata is extracted
    for tc in token_chunks:
        assert isinstance(tc.metadata, dict)
        # Metadata keys might vary based on extraction logic


def test_token_chunker_empty_input():
    """Test TokenChunker with empty input."""
    chunker = TokenChunker()
    
    # Empty list
    token_chunks = chunker.chunk([])
    assert len(token_chunks) == 0
    
    # Chunk with empty text
    empty_chunk = Chunk(
        id="empty",
        title="Empty",
        heading_level=1,
        text="",
        blocks=[],
        page_numbers=[],
    )
    
    token_chunks = chunker.chunk([empty_chunk])
    # Should handle gracefully (might be 0 or create minimal chunk)
    assert isinstance(token_chunks, list)


def test_token_chunker_overlap():
    """Test that TokenChunker creates overlapping chunks."""
    chunker = TokenChunker(max_tokens=20, min_tokens=10, overlap_tokens=5)
    
    # Create chunk with enough text for multiple splits
    text = " ".join([f"word{i}" for i in range(100)])
    sample_chunk = Chunk(
        id="overlap-test",
        title="Overlap Test",
        heading_level=1,
        text=text,
        blocks=[],
        page_numbers=[1],
    )
    
    token_chunks = chunker.chunk([sample_chunk])
    
    # Should create multiple chunks
    assert len(token_chunks) > 1
    
    # Check overlap (last words of chunk N should appear in chunk N+1)
    if len(token_chunks) >= 2:
        first_chunk_tokens = default_tokenizer(token_chunks[0].text)
        second_chunk_tokens = default_tokenizer(token_chunks[1].text)
        
        # Some tokens should overlap
        overlap_window = first_chunk_tokens[-5:]
        assert any(token in second_chunk_tokens for token in overlap_window)


def test_chunk_protocol():
    """Test that Chunk satisfies the ChunkLike protocol."""
    chunk = Chunk(
        id="test-id",
        title="Test Title",
        heading_level=1,
        text="Test text",
        blocks=[],
        page_numbers=[1],
    )
    
    # Check protocol compliance
    assert isinstance(chunk, ChunkLike)
    
    # Check required attributes
    assert chunk.id == "test-id"
    assert chunk.title == "Test Title"
    assert chunk.heading_level == 1
    assert chunk.text == "Test text"
    assert isinstance(chunk.blocks, list)
    assert isinstance(chunk.page_numbers, list)
    
    # Check to_dict method
    chunk_dict = chunk.to_dict()
    assert isinstance(chunk_dict, dict)
    assert "id" in chunk_dict
    assert "title" in chunk_dict
    assert "text" in chunk_dict


def test_two_stage_pipeline():
    """Test the recommended two-stage chunking pipeline."""
    # Stage 1: DynamicChunker
    dynamic_chunker = DynamicChunker(allow_preamble=True)
    
    pages = [
        {
            "page_number": 1,
            "blocks": [
                {"type": "heading", "level": 1, "text": "Chapter 1", "page_number": 1},
                {"type": "paragraph", "text": " ".join([f"word{i}" for i in range(200)]), "page_number": 1},
            ],
        },
    ]
    
    structural_chunks = dynamic_chunker.build_chunks(pages)
    assert len(structural_chunks) > 0
    
    # Stage 2: TokenChunker
    token_chunker = TokenChunker(max_tokens=50, min_tokens=20, overlap_tokens=10)
    token_chunks = token_chunker.chunk(structural_chunks)
    
    # Should create more granular chunks
    assert len(token_chunks) >= len(structural_chunks)
    
    # All token chunks should reference source chunks
    for tc in token_chunks:
        assert tc.source_chunk_id in [c.id for c in structural_chunks]
