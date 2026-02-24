"""Tests for MetadataEnricher."""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any

from quaerium.core.chunking import MetadataEnricher


@dataclass
class MockTokenChunk:
    """Mock token chunk for testing."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Additional fields to satisfy TokenChunkLike protocol
    id: str = ""
    section_path: str = ""
    page_numbers: list = field(default_factory=list)
    source_chunk_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


class TestMetadataEnricher:
    """Test metadata enrichment."""

    @pytest.fixture
    def enricher(self):
        """Create default enricher."""
        return MetadataEnricher()

    def test_enrich_text_basic(self, enricher):
        """Test basic text enrichment."""
        text = "Original text."
        metadata = {"author": "John", "year": "2024"}

        enriched = enricher.enrich_text(text, metadata)

        assert "Original text." in enriched
        assert "[author: John]" in enriched
        assert "[year: 2024]" in enriched

    def test_enrich_text_excludes_default_keys(self, enricher):
        """Test that default excluded keys are not added."""
        text = "Text"
        metadata = {"file_name": "doc.pdf", "chunk_id": "123", "author": "John"}

        enriched = enricher.enrich_text(text, metadata)

        assert "[file_name:" not in enriched
        assert "[chunk_id:" not in enriched
        assert "[author: John]" in enriched

    def test_enrich_text_excludes_id_and_source_chunk_id(self, enricher):
        """Test that id and source_chunk_id are excluded by default."""
        text = "Text"
        metadata = {"id": "123", "source_chunk_id": "456", "author": "John"}

        enriched = enricher.enrich_text(text, metadata)

        assert "[id:" not in enriched
        assert "[source_chunk_id:" not in enriched
        assert "[author: John]" in enriched

    def test_enrich_text_custom_excluded_keys(self):
        """Test custom excluded keys."""
        enricher = MetadataEnricher(excluded_keys=["internal_id"])
        text = "Text"
        metadata = {"internal_id": "123", "author": "John"}

        enriched = enricher.enrich_text(text, metadata)

        assert "[internal_id:" not in enriched
        assert "[author: John]" in enriched

    def test_enrich_text_custom_format(self):
        """Test custom format template."""
        enricher = MetadataEnricher(format_template="({key}={value})")
        text = "Text"
        metadata = {"author": "John"}

        enriched = enricher.enrich_text(text, metadata)

        assert "(author=John)" in enriched

    def test_enrich_text_custom_separator(self):
        """Test custom separator."""
        enricher = MetadataEnricher(separator=" | ")
        text = "Text"
        metadata = {"author": "John", "year": "2024"}

        enriched = enricher.enrich_text(text, metadata)

        assert " | " in enriched
        assert text in enriched
        assert "[author: John]" in enriched

    def test_enrich_text_ignores_empty_values(self, enricher):
        """Test that empty string values are not added."""
        text = "Text"
        metadata = {"author": "John", "empty_field": "", "whitespace": "   "}

        enriched = enricher.enrich_text(text, metadata)

        assert "[author: John]" in enriched
        assert "[empty_field:" not in enriched
        assert "[whitespace:" not in enriched

    def test_enrich_text_ignores_non_string_values(self, enricher):
        """Test that non-string values are ignored."""
        text = "Text"
        metadata = {"author": "John", "count": 123, "flag": True, "items": ["a", "b"]}

        enriched = enricher.enrich_text(text, metadata)

        assert "[author: John]" in enriched
        assert "[count:" not in enriched
        assert "[flag:" not in enriched
        assert "[items:" not in enriched

    def test_enrich_text_strips_whitespace(self, enricher):
        """Test that metadata values are stripped of whitespace."""
        text = "Text"
        metadata = {"author": "  John Doe  "}

        enriched = enricher.enrich_text(text, metadata)

        assert "[author: John Doe]" in enriched
        assert "  John Doe  " not in enriched

    def test_enrich_chunks_batch(self, enricher):
        """Test batch enrichment of chunks."""
        chunks = [
            MockTokenChunk(text="Text 1", metadata={"author": "John"}),
            MockTokenChunk(text="Text 2", metadata={"author": "Jane"}),
            MockTokenChunk(text="Text 3", metadata={"author": "Bob"}),
        ]

        enriched_texts = enricher.enrich_chunks(chunks)

        assert len(enriched_texts) == 3
        assert "Text 1" in enriched_texts[0]
        assert "[author: John]" in enriched_texts[0]
        assert "[author: Jane]" in enriched_texts[1]
        assert "[author: Bob]" in enriched_texts[2]

    def test_enrich_chunks_preserves_order(self, enricher):
        """Test that chunk order is preserved."""
        chunks = [
            MockTokenChunk(text=f"Text {i}", metadata={"id_value": str(i)})
            for i in range(10)
        ]

        enriched_texts = enricher.enrich_chunks(chunks)

        for i, enriched in enumerate(enriched_texts):
            assert f"Text {i}" in enriched
            assert f"[id_value: {i}]" in enriched

    def test_enrich_chunks_with_empty_metadata(self, enricher):
        """Test enrichment with chunks that have no metadata."""
        chunks = [
            MockTokenChunk(text="Text 1", metadata={}),
            MockTokenChunk(text="Text 2", metadata={}),
        ]

        enriched_texts = enricher.enrich_chunks(chunks)

        assert len(enriched_texts) == 2
        assert enriched_texts[0] == "Text 1"
        assert enriched_texts[1] == "Text 2"

    def test_enrich_chunks_with_mixed_metadata(self, enricher):
        """Test enrichment with mixed metadata."""
        chunks = [
            MockTokenChunk(text="Text 1", metadata={"author": "John"}),
            MockTokenChunk(text="Text 2", metadata={}),
            MockTokenChunk(text="Text 3", metadata={"year": "2024"}),
        ]

        enriched_texts = enricher.enrich_chunks(chunks)

        assert "[author: John]" in enriched_texts[0]
        assert enriched_texts[1] == "Text 2"
        assert "[year: 2024]" in enriched_texts[2]

    def test_invalid_format_template_missing_key(self):
        """Test that format template without {key} raises ValueError."""
        with pytest.raises(ValueError, match="must contain both {key} and {value}"):
            MetadataEnricher(format_template="[{value}]")

    def test_invalid_format_template_missing_value(self):
        """Test that format template without {value} raises ValueError."""
        with pytest.raises(ValueError, match="must contain both {key} and {value}"):
            MetadataEnricher(format_template="[{key}]")

    def test_enrich_text_with_no_metadata(self, enricher):
        """Test enrichment with empty metadata dict."""
        text = "Original text"
        metadata = {}

        enriched = enricher.enrich_text(text, metadata)

        assert enriched == "Original text"

    def test_enrich_text_with_all_excluded_keys(self, enricher):
        """Test enrichment when all metadata keys are excluded."""
        text = "Original text"
        metadata = {"file_name": "doc.pdf", "chunk_id": "123", "id": "abc"}

        enriched = enricher.enrich_text(text, metadata)

        assert enriched == "Original text"

    def test_empty_excluded_keys_list(self):
        """Test enricher with empty excluded keys list."""
        enricher = MetadataEnricher(excluded_keys=[])
        text = "Text"
        metadata = {"file_name": "doc.pdf", "author": "John"}

        enriched = enricher.enrich_text(text, metadata)

        # With empty excluded_keys, all string metadata should be included
        assert "[file_name: doc.pdf]" in enriched
        assert "[author: John]" in enriched

    def test_enrich_text_order_preservation(self):
        """Test that metadata order is somewhat predictable (dict iteration order in Python 3.7+)."""
        enricher = MetadataEnricher()
        text = "Text"
        metadata = {"author": "John", "year": "2024", "category": "legal"}

        enriched = enricher.enrich_text(text, metadata)

        # All metadata should be present
        assert "[author: John]" in enriched
        assert "[year: 2024]" in enriched
        assert "[category: legal]" in enriched
