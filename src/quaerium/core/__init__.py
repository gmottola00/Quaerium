"""
Core layer: Protocol definitions and type abstractions.

This layer contains only abstract interfaces (Protocols) with zero external dependencies.
All concrete implementations live in the infra layer.

Modules:
    - chunking: Document chunking protocols
    - embedding: Embedding client protocol
    - llm: LLM client protocol
    - vectorstore: Vector store protocol
    - types: Common type definitions
"""

from __future__ import annotations

# Re-export core protocols for easy access
from quaerium.core.chunking import Chunk, TokenChunk
from quaerium.core.embedding import EmbeddingClient
from quaerium.core.llm import LLMClient
from quaerium.core.types import (
    CollectionInfo,
    EmbeddingVector,
    SearchResult,
    VectorMetadata,
)
from quaerium.core.vectorstore import VectorStoreClient

__all__ = [
    # Chunking protocols
    "Chunk",
    "TokenChunk",
    # Client protocols
    "EmbeddingClient",
    "LLMClient",
    "VectorStoreClient",
    # Common types
    "SearchResult",
    "CollectionInfo",
    "VectorMetadata",
    "EmbeddingVector",
]
