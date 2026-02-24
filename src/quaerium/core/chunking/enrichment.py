"""Metadata enrichment for chunk text.

This module provides utilities for enriching chunk text with inline metadata,
improving retrieval quality by making metadata searchable within the text itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from quaerium.core.chunking.types import TokenChunkLike


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
            set(excluded_keys) if excluded_keys is not None else self.DEFAULT_EXCLUDED_KEYS
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
