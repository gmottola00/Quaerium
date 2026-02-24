# Claude Code Agent Instructions: Metadata Extraction & Enrichment Integration

## 🎯 Mission

Integrate **LLM-based metadata extraction** and **chunk text enrichment** into rag-toolkit as generic, reusable components. These features were identified from StudIA project and provide high value for RAG pipelines across domains.

---

## 📦 Deliverables

### **Agent 1: Core Implementation**
- Implement `LLMMetadataExtractor` in `rag_toolkit.core.metadata`
- Implement `MetadataEnricher` in `rag_toolkit.core.chunking`
- Add `embed_batch()` method to embedding clients
- Ensure Protocol compliance and type safety

### **Agent 2: Documentation**
- Write comprehensive docstrings (Google style)
- Create user guide in `docs/guides/metadata-extraction.md`
- Add API reference in `docs/api/metadata.md`
- Write usage examples for different domains

### **Agent 3: Testing**
- Write unit tests for `LLMMetadataExtractor` (mocking LLM)
- Write unit tests for `MetadataEnricher`
- Write integration tests with real embedding clients
- Add edge case tests (malformed JSON, empty metadata, etc.)

### **Agent 4: Integration & Examples**
- Update `__init__.py` exports
- Create working example in `examples/metadata_extraction_example.py`
- Add migration guide for StudIA users
- Update CHANGELOG.md

---

## 🏗️ Architecture Overview

```
rag-toolkit/src/rag_toolkit/
├── core/
│   ├── metadata/                    # ✨ NEW MODULE
│   │   ├── __init__.py
│   │   ├── llm_extractor.py        # LLMMetadataExtractor
│   │   └── types.py                # MetadataSchema Protocol
│   └── chunking/
│       ├── enrichment.py           # ✨ NEW: MetadataEnricher
│       └── ...
├── infra/
│   └── embedding/
│       ├── ollama.py               # ✨ UPDATE: add embed_batch()
│       └── openai.py               # ✨ UPDATE: add embed_batch()
└── examples/
    └── metadata_extraction.py      # ✨ NEW: Usage example
```

---

## 📋 Agent 1: Core Implementation

### Task 1.1: Create `LLMMetadataExtractor`

**File:** `rag_toolkit/src/rag_toolkit/core/metadata/llm_extractor.py`

```python
"""LLM-based metadata extraction from text.

This module provides a generic metadata extractor that uses LLMs to extract
structured information from unstructured text using customizable prompts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from rag_toolkit.core.llm import LLMClient

logger = logging.getLogger(__name__)


class LLMMetadataExtractor:
    """Extract structured metadata from text using LLM prompts.
    
    Generic implementation that accepts custom prompt templates for different domains.
    Useful for extracting entities, dates, classifications, or any structured data
    from unstructured documents.
    
    Examples:
        Legal Domain:
            >>> from rag_toolkit.infra.llm import OllamaLLMClient
            >>> 
            >>> LEGAL_SYSTEM_PROMPT = '''
            ... You are a legal document analyzer. Extract metadata in JSON format:
            ... {"case_number": "", "court": "", "date": "", "parties": []}
            ... '''
            >>> 
            >>> LEGAL_EXTRACTION_PROMPT = '''
            ... Given this legal document text:
            ... {context}
            ... 
            ... Extract: case number, court name, filing date, and party names.
            ... Return only valid JSON.
            ... '''
            >>> 
            >>> llm = OllamaLLMClient(model="llama3.2")
            >>> extractor = LLMMetadataExtractor(
            ...     llm_client=llm,
            ...     system_prompt=LEGAL_SYSTEM_PROMPT,
            ...     extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
            ... )
            >>> metadata = extractor.extract(document_text)
            >>> print(metadata["case_number"])
        
        Tender Domain (from StudIA):
            >>> TENDER_SYSTEM_PROMPT = '''
            ... You are an assistant for analyzing tender documents.
            ... Extract metadata in JSON: {"ente_appaltante": "", "cig": "", "importo": ""}
            ... '''
            >>> 
            >>> extractor = LLMMetadataExtractor(llm, TENDER_SYSTEM_PROMPT, ...)
            >>> metadata = extractor.extract(tender_text)
    
    Attributes:
        llm_client: LLM client implementing LLMClient protocol
        system_prompt: System prompt defining extraction task and output format
        extraction_prompt_template: User prompt template with {context} placeholder
        max_text_length: Maximum text length to send to LLM (truncates longer texts)
        response_parser: Optional custom parser for LLM response (default: JSON parser)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        extraction_prompt_template: str,
        *,
        max_text_length: int = 8000,
        response_parser: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize metadata extractor.
        
        Args:
            llm_client: LLM client for generation
            system_prompt: System prompt defining extraction schema
            extraction_prompt_template: Prompt template with {context} placeholder
            max_text_length: Max characters to send to LLM (default: 8000)
            response_parser: Custom parser function (default: JSON parser with cleanup)
        
        Raises:
            ValueError: If extraction_prompt_template doesn't contain {context}
        """
        if "{context}" not in extraction_prompt_template:
            raise ValueError(
                "extraction_prompt_template must contain {context} placeholder"
            )
        
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.extraction_prompt_template = extraction_prompt_template
        self.max_text_length = max_text_length
        self.response_parser = response_parser or self._default_json_parser

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text using LLM.
        
        Args:
            text: Input text to extract metadata from
        
        Returns:
            Dictionary of extracted metadata. Returns empty dict if parsing fails.
        
        Example:
            >>> metadata = extractor.extract("Contract between Acme Corp and...")
            >>> print(metadata["case_number"])
            "2024-CV-12345"
        """
        # Truncate text to max length
        truncated_text = text[: self.max_text_length]
        
        # Format prompt with text
        user_prompt = self.extraction_prompt_template.format(
            context=truncated_text
        )
        
        # Generate with LLM
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.0,  # Use deterministic generation for extraction
            )
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            return {}
        
        # Parse response
        return self.response_parser(response)

    @staticmethod
    def _default_json_parser(raw_response: str) -> Dict[str, Any]:
        """Default parser that cleans and parses JSON from LLM response.
        
        Handles common LLM output formats:
        - Wrapped in ```json``` code blocks
        - Wrapped in ``` code blocks
        - Raw JSON
        
        Args:
            raw_response: Raw LLM response string
        
        Returns:
            Parsed dictionary or empty dict if parsing fails
        """
        # Clean code block markers
        cleaned = raw_response.strip()
        
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
        
        # Parse JSON
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Failed to parse JSON from LLM response: {exc}\n"
                f"Response: {raw_response[:200]}..."
            )
            return {}


__all__ = ["LLMMetadataExtractor"]
```

### Task 1.2: Create `MetadataEnricher`

**File:** `rag_toolkit/src/rag_toolkit/core/chunking/enrichment.py`

```python
"""Metadata enrichment for chunk text.

This module provides utilities for enriching chunk text with inline metadata,
improving retrieval quality by making metadata searchable within the text itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from rag_toolkit.core.chunking.types import TokenChunkLike


class MetadataEnricher:
    """Enrich chunk text with inline metadata for better retrieval.
    
    Adds metadata as inline annotations to chunk text, making metadata fields
    searchable in both vector and keyword retrieval. This is particularly useful
    for improving recall on specific entity searches (e.g., searching for documents
    from a specific author, date range, or category).
    
    Examples:
        Basic Usage:
            >>> enricher = MetadataEnricher()
            >>> text = "The contract duration is 24 months."
            >>> metadata = {"author": "Legal Dept", "contract_id": "C-2024-001"}
            >>> enriched = enricher.enrich_text(text, metadata)
            >>> print(enriched)
            'The contract duration is 24 months. [author: Legal Dept] [contract_id: C-2024-001]'
        
        Custom Format:
            >>> enricher = MetadataEnricher(
            ...     format_template="({key}={value})",
            ...     excluded_keys=["internal_id", "chunk_id"]
            ... )
            >>> enriched = enricher.enrich_text(text, metadata)
            >>> print(enriched)
            'The contract duration is 24 months. (author=Legal Dept) (contract_id=C-2024-001)'
        
        Batch Enrichment for Embedding:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> enriched_texts = enricher.enrich_chunks(chunks)
            >>> embeddings = embed_client.embed_batch(enriched_texts)
    
    Attributes:
        excluded_keys: Set of metadata keys to exclude from enrichment
        format_template: Template for formatting metadata (must have {key} and {value})
        separator: String to separate metadata annotations (default: single space)
    """

    DEFAULT_EXCLUDED_KEYS = {"file_name", "chunk_id", "id", "source_chunk_id"}

    def __init__(
        self,
        *,
        excluded_keys: List[str] | None = None,
        format_template: str = "[{key}: {value}]",
        separator: str = " ",
    ) -> None:
        """Initialize metadata enricher.
        
        Args:
            excluded_keys: List of metadata keys to exclude (default: file_name, chunk_id, id)
            format_template: Template for formatting metadata with {key} and {value} placeholders
            separator: String to separate metadata annotations
        
        Raises:
            ValueError: If format_template doesn't contain both {key} and {value}
        """
        if "{key}" not in format_template or "{value}" not in format_template:
            raise ValueError(
                "format_template must contain both {key} and {value} placeholders"
            )
        
        self.excluded_keys: Set[str] = (
            set(excluded_keys) if excluded_keys else self.DEFAULT_EXCLUDED_KEYS
        )
        self.format_template = format_template
        self.separator = separator

    def enrich_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add metadata inline to text.
        
        Args:
            text: Original chunk text
            metadata: Metadata dictionary to add inline
        
        Returns:
            Text with metadata annotations appended
        
        Example:
            >>> enriched = enricher.enrich_text(
            ...     "Contract terms...",
            ...     {"client": "Acme Corp", "year": "2024"}
            ... )
            >>> print(enriched)
            'Contract terms... [client: Acme Corp] [year: 2024]'
        """
        enriched_parts = [text]
        
        for key, value in metadata.items():
            # Skip excluded keys
            if key in self.excluded_keys:
                continue
            
            # Only add non-empty string values
            if isinstance(value, str) and value.strip():
                formatted = self.format_template.format(
                    key=key, value=value.strip()
                )
                enriched_parts.append(formatted)
        
        return self.separator.join(enriched_parts)

    def enrich_chunks(
        self, chunks: List[TokenChunkLike]
    ) -> List[str]:
        """Enrich multiple chunks, returning enriched text list.
        
        Useful for batch embedding where you want to embed enriched text
        while keeping original chunk objects intact.
        
        Args:
            chunks: List of token chunks (must have .text and .metadata)
        
        Returns:
            List of enriched text strings (same order as input chunks)
        
        Example:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> enriched_texts = enricher.enrich_chunks(chunks)
            >>> # Use enriched texts for embedding
            >>> embeddings = embed_client.embed_batch(enriched_texts)
            >>> # Store embeddings with original chunks
            >>> for chunk, embedding in zip(chunks, embeddings):
            ...     store_chunk(chunk, embedding)
        """
        return [
            self.enrich_text(chunk.text, chunk.metadata)
            for chunk in chunks
        ]


__all__ = ["MetadataEnricher"]
```

### Task 1.3: Add `embed_batch()` to Embedding Clients

**File:** `rag_toolkit/src/rag_toolkit/core/embedding.py` (update Protocol)

```python
# Add to EmbeddingClient Protocol
from typing import List, Protocol

class EmbeddingClient(Protocol):
    """Protocol for embedding models."""
    
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (optional optimization).
        
        Default implementation calls embed() for each text.
        Subclasses can override for batch API optimization.
        """
        return [self.embed(text) for text in texts]
```

**File:** `rag_toolkit/src/rag_toolkit/infra/embedding/ollama.py`

```python
# Add method to OllamaEmbeddingClient
class OllamaEmbeddingClient:
    # ... existing code ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Note: Ollama doesn't have native batch API, so this calls
        embed() for each text. Override in subclass if batch API exists.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (same order as input)
        
        Example:
            >>> client = OllamaEmbeddingClient()
            >>> embeddings = client.embed_batch(["text1", "text2", "text3"])
            >>> len(embeddings)
            3
        """
        return [self.embed(text) for text in texts]
```

**File:** `rag_toolkit/src/rag_toolkit/infra/embedding/openai.py`

```python
# Add method to OpenAIEmbeddingClient
class OpenAIEmbeddingClient:
    # ... existing code ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI batch API.
        
        Uses OpenAI's native batch embedding endpoint for efficiency.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (same order as input)
        
        Example:
            >>> client = OpenAIEmbeddingClient(model="text-embedding-3-small")
            >>> embeddings = client.embed_batch(["text1", "text2", "text3"])
            >>> len(embeddings)
            3
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            # Fallback to sequential if batch fails
            return [self.embed(text) for text in texts]
```

### Task 1.4: Update Module Exports

**File:** `rag_toolkit/src/rag_toolkit/core/metadata/__init__.py`

```python
"""Metadata extraction utilities."""

from .llm_extractor import LLMMetadataExtractor

__all__ = ["LLMMetadataExtractor"]
```

**File:** `rag_toolkit/src/rag_toolkit/core/chunking/__init__.py`

```python
# Add to existing exports
from .enrichment import MetadataEnricher

__all__ = [
    # ... existing exports ...
    "MetadataEnricher",
]
```

---

## 📚 Agent 2: Documentation

### Task 2.1: User Guide

**File:** `rag_toolkit/docs/guides/metadata-extraction.md`

```markdown
# Metadata Extraction & Enrichment

Learn how to extract structured metadata from documents and enrich chunks for better retrieval.

## Overview

**Metadata extraction** uses LLMs to extract structured information from unstructured text (e.g., extracting contract parties, dates, case numbers from legal documents).

**Metadata enrichment** adds extracted metadata inline to chunk text, making it searchable in both vector and keyword retrieval.

## When to Use

- **Legal Documents**: Extract case numbers, parties, courts, filing dates
- **Medical Records**: Extract patient IDs, diagnosis codes, procedure dates
- **Tender Documents**: Extract contracting authority, CIG codes, deadlines
- **Academic Papers**: Extract authors, institutions, publication dates
- **Contracts**: Extract parties, effective dates, termination clauses

## Basic Example: Legal Document Extraction

```python
from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.core.chunking import MetadataEnricher, TokenChunker
from rag_toolkit.infra.llm import OllamaLLMClient
from rag_toolkit.infra.embedding import OllamaEmbeddingClient

# Step 1: Define extraction schema (domain-specific)
LEGAL_SYSTEM_PROMPT = """
You are a legal document analyzer. Extract metadata in JSON format:
{
  "case_number": "",
  "court": "",
  "filing_date": "",
  "parties": []
}
If a field is not found, use empty string or empty array.
"""

LEGAL_EXTRACTION_PROMPT = """
Given this legal document excerpt:
{context}

Extract: case number, court name, filing date, and party names.
Return ONLY valid JSON matching the schema.
"""

# Step 2: Initialize extractor
llm = OllamaLLMClient(model="llama3.2")
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=LEGAL_SYSTEM_PROMPT,
    extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
    max_text_length=8000,
)

# Step 3: Extract metadata from document
document_text = load_document("contract.pdf")
metadata = extractor.extract(document_text)

print(metadata)
# Output: {"case_number": "2024-CV-12345", "court": "Superior Court", ...}
```

## Full Pipeline: Extract → Chunk → Enrich → Index

```python
from rag_toolkit.core.chunking import DynamicChunker, TokenChunker, MetadataEnricher
from rag_toolkit.infra.parsers import create_ingestion_service

# 1. Parse document
ingestion = create_ingestion_service()
parsed_pages = ingestion.parse_file("contract.pdf")
full_text = " ".join([page["text"] for page in parsed_pages])

# 2. Extract metadata from full document
metadata = extractor.extract(full_text)
print(f"Extracted: {metadata}")

# 3. Chunk document
dynamic_chunker = DynamicChunker()
token_chunker = TokenChunker(max_tokens=512)
structured_chunks = dynamic_chunker.build_chunks(parsed_pages)
token_chunks = token_chunker.chunk(structured_chunks)

# 4. Add extracted metadata to all chunks
for chunk in token_chunks:
    chunk.metadata.update(metadata)

# 5. Enrich chunk text with metadata
enricher = MetadataEnricher()
enriched_texts = enricher.enrich_chunks(token_chunks)

# 6. Embed enriched text
embed_client = OllamaEmbeddingClient()
embeddings = embed_client.embed_batch(enriched_texts)

# 7. Index with original chunks + embeddings
for chunk, embedding in zip(token_chunks, embeddings):
    index_service.index(chunk, embedding)
```

## Domain-Specific Examples

### Tender/Procurement Documents

```python
TENDER_SYSTEM_PROMPT = """
Extract tender metadata in JSON:
{
  "contracting_authority": "",
  "tender_id": "",
  "deadline": "",
  "budget": ""
}
"""

TENDER_EXTRACTION_PROMPT = """
From this tender document:
{context}

Extract: contracting authority, tender ID (CIG), submission deadline, budget.
Return JSON only.
"""

extractor = LLMMetadataExtractor(llm, TENDER_SYSTEM_PROMPT, TENDER_EXTRACTION_PROMPT)
metadata = extractor.extract(tender_text)
```

### Academic Papers

```python
PAPER_SYSTEM_PROMPT = """
Extract paper metadata in JSON:
{
  "title": "",
  "authors": [],
  "institution": "",
  "year": ""
}
"""

extractor = LLMMetadataExtractor(llm, PAPER_SYSTEM_PROMPT, ...)
metadata = extractor.extract(paper_text)
```

## Custom Response Parsing

If your LLM outputs non-JSON formats, provide a custom parser:

```python
def custom_parser(response: str) -> dict:
    """Parse custom format: KEY1=value1\nKEY2=value2"""
    metadata = {}
    for line in response.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata

extractor = LLMMetadataExtractor(
    llm,
    system_prompt,
    prompt_template,
    response_parser=custom_parser,
)
```

## Best Practices

1. **Truncate Long Documents**: Use `max_text_length` to avoid context limits
2. **Use Temperature=0**: Deterministic generation for consistent extraction
3. **Validate Schema**: Check extracted metadata matches expected fields
4. **Handle Failures**: Extractor returns `{}` if parsing fails (check for empty dict)
5. **Domain-Specific Prompts**: Tailor prompts to your document type
6. **Test with Real Docs**: Validate extraction quality on representative samples

## Performance Tips

- **Batch Embedding**: Use `embed_batch()` instead of sequential `embed()`
- **Cache Metadata**: Extract once per document, reuse for all chunks
- **Async LLM Calls**: Use async LLM clients for parallel extraction
- **Selective Enrichment**: Only enrich fields relevant for retrieval

## Migration from StudIA

If you're migrating from StudIA's `MetadataExtractor`:

```python
# OLD (StudIA)
from services.pdf_parser.preprocessor import MetadataExtractor
extractor = MetadataExtractor(model_name="phi4-mini")
metadata = extractor.extract(text)

# NEW (rag-toolkit)
from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.infra.llm import OllamaLLMClient

llm = OllamaLLMClient(model="phi4-mini")
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=YOUR_SYSTEM_PROMPT,
    extraction_prompt_template=YOUR_PROMPT_TEMPLATE,
)
metadata = extractor.extract(text)
```

## See Also

- [API Reference: LLMMetadataExtractor](../api/metadata.md)
- [Chunking Guide](./chunking.md)
- [RAG Pipeline Guide](./rag-pipeline.md)
```

### Task 2.2: API Reference

**File:** `rag_toolkit/docs/api/metadata.md`

```markdown
# Metadata API Reference

## LLMMetadataExtractor

::: rag_toolkit.core.metadata.LLMMetadataExtractor
    options:
      show_source: true
      show_signature: true
      show_root_heading: true
      heading_level: 3

## MetadataEnricher

::: rag_toolkit.core.chunking.MetadataEnricher
    options:
      show_source: true
      show_signature: true
      show_root_heading: true
      heading_level: 3
```

---

## 🧪 Agent 3: Testing

### Task 3.1: Test `LLMMetadataExtractor`

**File:** `rag_toolkit/tests/core/metadata/test_llm_extractor.py`

```python
"""Tests for LLMMetadataExtractor."""

import pytest
from unittest.mock import Mock

from rag_toolkit.core.metadata import LLMMetadataExtractor


class TestLLMMetadataExtractor:
    """Test LLM-based metadata extraction."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        mock = Mock()
        mock.generate.return_value = '{"key": "value"}'
        return mock
    
    @pytest.fixture
    def extractor(self, mock_llm_client):
        """Create extractor with mock LLM."""
        return LLMMetadataExtractor(
            llm_client=mock_llm_client,
            system_prompt="Extract metadata",
            extraction_prompt_template="From text: {context}\nExtract metadata.",
        )
    
    def test_extract_valid_json(self, extractor, mock_llm_client):
        """Test extraction with valid JSON response."""
        mock_llm_client.generate.return_value = '{"author": "John Doe", "year": "2024"}'
        
        metadata = extractor.extract("Some document text")
        
        assert metadata == {"author": "John Doe", "year": "2024"}
        mock_llm_client.generate.assert_called_once()
    
    def test_extract_json_with_code_blocks(self, extractor, mock_llm_client):
        """Test extraction with JSON in code blocks."""
        mock_llm_client.generate.return_value = '```json\n{"key": "value"}\n```'
        
        metadata = extractor.extract("Text")
        
        assert metadata == {"key": "value"}
    
    def test_extract_malformed_json(self, extractor, mock_llm_client):
        """Test extraction with malformed JSON returns empty dict."""
        mock_llm_client.generate.return_value = 'Not valid JSON {{'
        
        metadata = extractor.extract("Text")
        
        assert metadata == {}
    
    def test_extract_truncates_long_text(self, extractor, mock_llm_client):
        """Test that long text is truncated."""
        long_text = "x" * 10000
        extractor.max_text_length = 100
        
        extractor.extract(long_text)
        
        # Check prompt contains truncated text
        call_args = mock_llm_client.generate.call_args
        assert len(call_args.kwargs["prompt"]) < 200  # Prompt + template overhead
    
    def test_extract_uses_temperature_zero(self, extractor, mock_llm_client):
        """Test that extraction uses temperature=0."""
        extractor.extract("Text")
        
        call_args = mock_llm_client.generate.call_args
        assert call_args.kwargs.get("temperature") == 0.0
    
    def test_custom_response_parser(self, mock_llm_client):
        """Test custom response parser."""
        def custom_parser(response: str) -> dict:
            # Parse "KEY=value" format
            return {k: v for k, v in [line.split("=") for line in response.split("\n") if "=" in line]}
        
        extractor = LLMMetadataExtractor(
            llm_client=mock_llm_client,
            system_prompt="Extract",
            extraction_prompt_template="{context}",
            response_parser=custom_parser,
        )
        
        mock_llm_client.generate.return_value = "author=John\nyear=2024"
        metadata = extractor.extract("Text")
        
        assert metadata == {"author": "John", "year": "2024"}
    
    def test_missing_context_placeholder_raises_error(self, mock_llm_client):
        """Test that missing {context} raises ValueError."""
        with pytest.raises(ValueError, match="must contain {context}"):
            LLMMetadataExtractor(
                llm_client=mock_llm_client,
                system_prompt="Extract",
                extraction_prompt_template="No placeholder here",
            )
    
    def test_llm_exception_returns_empty_dict(self, extractor, mock_llm_client):
        """Test that LLM exceptions are handled gracefully."""
        mock_llm_client.generate.side_effect = RuntimeError("LLM error")
        
        metadata = extractor.extract("Text")
        
        assert metadata == {}
```

### Task 3.2: Test `MetadataEnricher`

**File:** `rag_toolkit/tests/core/chunking/test_enrichment.py`

```python
"""Tests for MetadataEnricher."""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List

from rag_toolkit.core.chunking import MetadataEnricher


@dataclass
class MockTokenChunk:
    """Mock token chunk for testing."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        metadata = {"author": "John", "count": 123, "flag": True}
        
        enriched = enricher.enrich_text(text, metadata)
        
        assert "[author: John]" in enriched
        assert "[count:" not in enriched
        assert "[flag:" not in enriched
    
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
            MockTokenChunk(text=f"Text {i}", metadata={"id": str(i)})
            for i in range(10)
        ]
        
        enriched_texts = enricher.enrich_chunks(chunks)
        
        for i, enriched in enumerate(enriched_texts):
            assert f"Text {i}" in enriched
    
    def test_invalid_format_template_raises_error(self):
        """Test that invalid format template raises ValueError."""
        with pytest.raises(ValueError, match="must contain both {key} and {value}"):
            MetadataEnricher(format_template="[{key}]")
        
        with pytest.raises(ValueError, match="must contain both {key} and {value}"):
            MetadataEnricher(format_template="[{value}]")
```

### Task 3.3: Integration Tests

**File:** `rag_toolkit/tests/integration/test_metadata_workflow.py`

```python
"""Integration tests for metadata extraction workflow."""

import pytest
from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.core.chunking import MetadataEnricher, TokenChunker, DynamicChunker
from rag_toolkit.infra.llm import OllamaLLMClient
from rag_toolkit.infra.embedding import OllamaEmbeddingClient


@pytest.mark.integration
class TestMetadataWorkflow:
    """Test full metadata extraction and enrichment workflow."""
    
    @pytest.fixture
    def sample_document(self):
        """Sample legal document text."""
        return """
        SUPERIOR COURT OF CALIFORNIA
        COUNTY OF LOS ANGELES
        
        Case No. 2024-CV-12345
        
        John Doe, Plaintiff
        v.
        Acme Corporation, Defendant
        
        Filed: January 15, 2024
        
        This case concerns a contract dispute between the parties...
        """
    
    def test_full_extraction_and_enrichment_workflow(self, sample_document):
        """Test complete workflow: extract → chunk → enrich → embed."""
        # 1. Extract metadata
        llm = OllamaLLMClient(model="llama3.2")
        
        SYSTEM_PROMPT = """
        Extract legal metadata in JSON:
        {"case_number": "", "court": "", "parties": [], "filing_date": ""}
        """
        
        EXTRACTION_PROMPT = """
        From document:
        {context}
        
        Extract case number, court, parties, and filing date.
        Return JSON only.
        """
        
        extractor = LLMMetadataExtractor(
            llm_client=llm,
            system_prompt=SYSTEM_PROMPT,
            extraction_prompt_template=EXTRACTION_PROMPT,
        )
        
        metadata = extractor.extract(sample_document)
        
        # Verify extraction (may vary based on LLM)
        assert "case_number" in metadata
        assert "court" in metadata
        
        # 2. Create mock chunks
        from dataclasses import dataclass, field
        from typing import Dict, Any
        
        @dataclass
        class MockChunk:
            text: str
            metadata: Dict[str, Any] = field(default_factory=dict)
        
        chunks = [
            MockChunk(
                text="This case concerns a contract dispute...",
                metadata=metadata,
            ),
            MockChunk(
                text="The plaintiff alleges breach of contract...",
                metadata=metadata,
            ),
        ]
        
        # 3. Enrich chunks
        enricher = MetadataEnricher()
        enriched_texts = enricher.enrich_chunks(chunks)
        
        # Verify enrichment
        assert len(enriched_texts) == 2
        for enriched in enriched_texts:
            # Check metadata is inline
            if metadata.get("case_number"):
                assert metadata["case_number"] in enriched
        
        # 4. Embed enriched text
        embed_client = OllamaEmbeddingClient()
        embeddings = embed_client.embed_batch(enriched_texts)
        
        # Verify embeddings
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)
```

---

## 🔗 Agent 4: Integration & Examples

### Task 4.1: Working Example

**File:** `rag_toolkit/examples/metadata_extraction_example.py`

```python
"""Example: Extract metadata from legal documents and enrich chunks.

This example demonstrates:
1. LLM-based metadata extraction from documents
2. Chunking with DynamicChunker and TokenChunker
3. Metadata enrichment for better retrieval
4. Batch embedding of enriched chunks

Domain: Legal documents (contracts, court filings)
"""

from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.core.chunking import DynamicChunker, TokenChunker, MetadataEnricher
from rag_toolkit.infra.llm import OllamaLLMClient
from rag_toolkit.infra.embedding import OllamaEmbeddingClient
from rag_toolkit.infra.parsers import create_ingestion_service


# ============================================================================
# Step 1: Define Domain-Specific Extraction Prompts
# ============================================================================

LEGAL_SYSTEM_PROMPT = """
You are a legal document analyzer. Your task is to extract structured metadata
from legal documents and return it in the following JSON format:

{
  "case_number": "",
  "court": "",
  "filing_date": "",
  "parties": [],
  "document_type": ""
}

Rules:
- If a field is not found, use empty string ("") or empty array ([])
- Return ONLY valid JSON, no additional text
- Be precise with extracted values
"""

LEGAL_EXTRACTION_PROMPT = """
Given the following legal document excerpt:

{context}

Extract the following information:
- case_number: The case identifier (e.g., "2024-CV-12345")
- court: The name of the court (e.g., "Superior Court of California")
- filing_date: The filing or effective date
- parties: List of party names (plaintiffs, defendants)
- document_type: Type of document (e.g., "complaint", "motion", "contract")

Return ONLY the JSON object with extracted metadata.
"""


# ============================================================================
# Step 2: Initialize Components
# ============================================================================

def main():
    # LLM for metadata extraction
    llm = OllamaLLMClient(model="llama3.2")
    
    # Embedding for vector indexing
    embed_client = OllamaEmbeddingClient(model="nomic-embed-text")
    
    # Document parser
    ingestion_service = create_ingestion_service()
    
    # Metadata extractor
    metadata_extractor = LLMMetadataExtractor(
        llm_client=llm,
        system_prompt=LEGAL_SYSTEM_PROMPT,
        extraction_prompt_template=LEGAL_EXTRACTION_PROMPT,
        max_text_length=8000,  # Truncate long documents
    )
    
    # Chunkers
    dynamic_chunker = DynamicChunker(include_tables=True)
    token_chunker = TokenChunker(max_tokens=512, overlap_tokens=50)
    
    # Metadata enricher
    enricher = MetadataEnricher(
        format_template="[{key}: {value}]",
        excluded_keys=["file_name", "chunk_id", "id"],
    )
    
    # ============================================================================
    # Step 3: Process Document
    # ============================================================================
    
    # Parse document
    print("📄 Parsing document...")
    document_path = "examples/data/sample_contract.pdf"
    parsed_pages = ingestion_service.parse_file(document_path)
    
    # Get full text for metadata extraction
    full_text = " ".join([page["text"] for page in parsed_pages])
    print(f"📝 Extracted {len(full_text)} characters")
    
    # ============================================================================
    # Step 4: Extract Metadata
    # ============================================================================
    
    print("\n🔍 Extracting metadata with LLM...")
    document_metadata = metadata_extractor.extract(full_text)
    
    print("✅ Extracted metadata:")
    for key, value in document_metadata.items():
        print(f"  {key}: {value}")
    
    # ============================================================================
    # Step 5: Chunk Document
    # ============================================================================
    
    print("\n✂️ Chunking document...")
    structured_chunks = dynamic_chunker.build_chunks(parsed_pages)
    token_chunks = token_chunker.chunk(structured_chunks)
    print(f"📦 Created {len(token_chunks)} chunks")
    
    # ============================================================================
    # Step 6: Add Metadata to Chunks
    # ============================================================================
    
    print("\n📎 Adding metadata to chunks...")
    for chunk in token_chunks:
        # Add document-level metadata to all chunks
        chunk.metadata.update(document_metadata)
    
    # ============================================================================
    # Step 7: Enrich Chunk Text
    # ============================================================================
    
    print("\n✨ Enriching chunk text with metadata...")
    enriched_texts = enricher.enrich_chunks(token_chunks)
    
    # Show example of enriched text
    print("\n📋 Example enriched chunk:")
    print(f"Original: {token_chunks[0].text[:100]}...")
    print(f"Enriched: {enriched_texts[0][:200]}...")
    
    # ============================================================================
    # Step 8: Embed Enriched Text
    # ============================================================================
    
    print("\n🔢 Generating embeddings for enriched chunks...")
    embeddings = embed_client.embed_batch(enriched_texts)
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    
    # ============================================================================
    # Step 9: Index (Mock - replace with actual index service)
    # ============================================================================
    
    print("\n💾 Indexing chunks...")
    for i, (chunk, embedding) in enumerate(zip(token_chunks, embeddings)):
        # In production, store in Milvus/Pinecone/etc.
        print(f"   Chunk {i+1}: {len(chunk.text)} chars, metadata keys: {list(chunk.metadata.keys())}")
    
    print("\n✅ Pipeline complete!")
    print(f"\n📊 Summary:")
    print(f"   - Parsed pages: {len(parsed_pages)}")
    print(f"   - Extracted metadata fields: {len(document_metadata)}")
    print(f"   - Total chunks: {len(token_chunks)}")
    print(f"   - Enriched chunks: {len(enriched_texts)}")
    print(f"   - Embeddings: {len(embeddings)}")


if __name__ == "__main__":
    main()
```

### Task 4.2: Update CHANGELOG

**File:** `rag_toolkit/CHANGELOG.md`

```markdown
# Changelog

## [Unreleased]

### Added

- **Metadata Extraction**: New `LLMMetadataExtractor` for extracting structured metadata from text using LLM prompts
  - Generic implementation with customizable prompt templates
  - Support for domain-specific extraction (legal, medical, tender, academic, etc.)
  - Automatic JSON parsing with code block cleanup
  - Graceful error handling (returns empty dict on failure)
  
- **Metadata Enrichment**: New `MetadataEnricher` for adding metadata inline to chunk text
  - Improves retrieval quality by making metadata searchable
  - Customizable format templates and excluded keys
  - Batch enrichment for efficient processing
  
- **Batch Embedding**: Added `embed_batch()` method to all embedding clients
  - `OllamaEmbeddingClient.embed_batch()` for sequential batch processing
  - `OpenAIEmbeddingClient.embed_batch()` using native batch API
  - Improves performance for large-scale indexing

- **Documentation**:
  - New user guide: `docs/guides/metadata-extraction.md`
  - API reference: `docs/api/metadata.md`
  - Working example: `examples/metadata_extraction_example.py`

### Changed

- `EmbeddingClient` protocol now includes optional `embed_batch()` method

## [Previous versions...]
```

### Task 4.3: Migration Guide for StudIA Users

**File:** `rag_toolkit/docs/guides/migration-from-studia.md`

```markdown
# Migration Guide: From StudIA to rag-toolkit

This guide helps StudIA users migrate to rag-toolkit's generic metadata extraction.

## Before (StudIA)

```python
from services.pdf_parser.preprocessor import MetadataExtractor, Preprocessor
from services.generative.consts import SYS_PROMPT_META, QUERY_CONTEXT_META

# StudIA-specific extractor
extractor = MetadataExtractor(model_name="phi4-mini:3.8b")
metadata = extractor.extract(text)

# StudIA-specific preprocessor
preprocessor = Preprocessor(
    output_dir="data/output",
    embedding_model="nomic-embed-text",
    llm_model="phi4-mini:3.8b"
)
```

## After (rag-toolkit)

```python
from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.core.chunking import MetadataEnricher, TokenChunker
from rag_toolkit.infra.llm import OllamaLLMClient
from rag_toolkit.infra.embedding import OllamaEmbeddingClient

# Generic extractor with custom prompts
llm = OllamaLLMClient(model="phi4-mini:3.8b")
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=YOUR_SYSTEM_PROMPT,  # Same as SYS_PROMPT_META
    extraction_prompt_template=YOUR_EXTRACTION_PROMPT,  # Same as QUERY_CONTEXT_META
)
metadata = extractor.extract(text)

# Separate components (better architecture)
embed_client = OllamaEmbeddingClient(model="nomic-embed-text")
token_chunker = TokenChunker(max_tokens=512)
enricher = MetadataEnricher()
```

## Key Changes

1. **Separation of Concerns**: Metadata extraction, chunking, and embedding are now separate components
2. **Protocol-Based**: Use `LLMClient` protocol instead of direct Ollama client
3. **Generic Prompts**: System prompts are now parameters (not hardcoded constants)
4. **Batch Processing**: Use `embed_batch()` for efficiency

## Step-by-Step Migration

### 1. Replace MetadataExtractor

```python
# OLD
from services.pdf_parser.preprocessor import MetadataExtractor
extractor = MetadataExtractor(model_name="phi4-mini")
metadata = extractor.extract(text)

# NEW
from rag_toolkit.core.metadata import LLMMetadataExtractor
from rag_toolkit.infra.llm import OllamaLLMClient

llm = OllamaLLMClient(model="phi4-mini")
extractor = LLMMetadataExtractor(
    llm_client=llm,
    system_prompt=YOUR_SYSTEM_PROMPT,
    extraction_prompt_template=YOUR_PROMPT,
)
metadata = extractor.extract(text)
```

### 2. Replace Preprocessor

```python
# OLD
preprocessor = Preprocessor(output_dir, embedding_model, llm_model)
doc_results = preprocessor.process_document(doc)

# NEW (separate concerns)
# a) Chunking
from rag_toolkit.core.chunking import TokenChunker
chunker = TokenChunker(max_tokens=1300, overlap_tokens=130)
chunks = chunker.chunk(structured_chunks)

# b) Metadata extraction
metadata = extractor.extract(full_text)

# c) Add metadata to chunks
for chunk in chunks:
    chunk.metadata.update(metadata)

# d) Enrichment
from rag_toolkit.core.chunking import MetadataEnricher
enricher = MetadataEnricher()
enriched_texts = enricher.enrich_chunks(chunks)

# e) Embedding
from rag_toolkit.infra.embedding import OllamaEmbeddingClient
embed_client = OllamaEmbeddingClient(model="nomic-embed-text")
embeddings = embed_client.embed_batch(enriched_texts)
```

### 3. Update Prompt Constants

Move prompt constants from `services/generative/consts.py` to config:

```python
# config/rag_prompts.py
SYSTEM_PROMPT = """
Sei un assistente per gare d'appalto...
"""

EXTRACTION_PROMPT = """
Dato il testo:
{context}

Estrai: ente_appaltante, CIG, importo, ...
"""
```

## Benefits of Migration

- ✅ **Generic Components**: Reusable across projects
- ✅ **Better Testing**: Mock LLM clients easily
- ✅ **Type Safety**: Protocol-based architecture
- ✅ **Batch Processing**: `embed_batch()` for performance
- ✅ **Clean Architecture**: Separation of concerns
```

---

## ✅ Success Criteria

### Agent 1 (Implementation)
- [ ] `LLMMetadataExtractor` implemented with tests passing
- [ ] `MetadataEnricher` implemented with tests passing
- [ ] `embed_batch()` added to all embedding clients
- [ ] All type hints correct (mypy passes)
- [ ] No circular imports

### Agent 2 (Documentation)
- [ ] User guide complete with 3+ domain examples
- [ ] API reference generated (mkdocs)
- [ ] All code examples tested and working
- [ ] Migration guide for StudIA users

### Agent 3 (Testing)
- [ ] Unit tests: >90% coverage for new modules
- [ ] Integration tests pass with real LLM (Ollama)
- [ ] Edge cases covered (malformed JSON, empty metadata, etc.)
- [ ] Performance benchmarks documented

### Agent 4 (Integration)
- [ ] `examples/metadata_extraction_example.py` runs successfully
- [ ] CHANGELOG.md updated
- [ ] `__init__.py` exports updated
- [ ] README.md updated with feature mention

---

## 🚀 Execution Plan

### Phase 1: Core Implementation (Agent 1)
**Estimated time:** 2-3 hours

1. Create `rag_toolkit/core/metadata/` directory
2. Implement `LLMMetadataExtractor` with docstrings
3. Implement `MetadataEnricher` with docstrings
4. Add `embed_batch()` to embedding clients
5. Update `__init__.py` exports
6. Run type checking (`mypy`)

### Phase 2: Documentation (Agent 2)
**Estimated time:** 2-3 hours

1. Write `docs/guides/metadata-extraction.md`
2. Add 3+ domain-specific examples
3. Create API reference pages
4. Write migration guide for StudIA
5. Generate docs (`mkdocs build`)

### Phase 3: Testing (Agent 3)
**Estimated time:** 2-3 hours

1. Write unit tests for `LLMMetadataExtractor`
2. Write unit tests for `MetadataEnricher`
3. Write integration tests with real LLM
4. Add edge case tests
5. Run coverage report (`pytest --cov`)
6. Ensure >90% coverage

### Phase 4: Integration & Examples (Agent 4)
**Estimated time:** 1-2 hours

1. Create `examples/metadata_extraction_example.py`
2. Test example end-to-end
3. Update CHANGELOG.md
4. Update README.md
5. Create PR with all changes

**Total estimated time:** 7-11 hours

---

## 🎯 Final Checklist

- [ ] All agents completed their tasks
- [ ] Tests pass (`pytest tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation builds (`mkdocs build`)
- [ ] Example runs successfully
- [ ] Code review completed
- [ ] CHANGELOG.md updated
- [ ] PR created and merged

---

## 📞 Support

For questions or issues during implementation, reference:
- Original StudIA code: `/Users/gianmarcomottola/Desktop/projects/StudIA/backend/app/services/`
- rag-toolkit architecture: `rag-toolkit/docs/architecture/`
- Tender-RAG-Lab integration example: `/Users/gianmarcomottola/Desktop/projects/Tender-RAG-Lab/`

---

**Ready to execute? Start with Agent 1! 🚀**
