"""
Common type definitions used across the library.

This module provides shared type aliases and data structures that are used
by multiple modules to ensure consistency and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

# ============================================================================
# Type Aliases
# ============================================================================

# Metadata can be any JSON-serializable dict
VectorMetadata: TypeAlias = dict[str, Any]

# Embedding vector
EmbeddingVector: TypeAlias = list[float]

# Graph metadata can be any JSON-serializable dict
GraphMetadata: TypeAlias = dict[str, Any]


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class SearchResult:
    """
    Result from a vector search operation.
    
    Attributes:
        id: Unique identifier of the vector
        score: Similarity score (higher = more similar)
        text: Original text associated with the vector
        metadata: Additional metadata stored with the vector
        vector: Optional embedding vector (not always returned)
    
    Example:
        >>> result = SearchResult(
        ...     id="doc_123",
        ...     score=0.95,
        ...     text="Installation guide for Python",
        ...     metadata={"source": "manual", "page": 5}
        ... )
        >>> print(f"Found: {result.text} (score: {result.score:.2f})")
    """

    id: str
    score: float
    text: str
    metadata: VectorMetadata
    vector: EmbeddingVector | None = None

    def __post_init__(self) -> None:
        """Validate score is in reasonable range."""
        if not 0.0 <= self.score <= 1.0:
            # Some metrics (like inner product) can exceed 1.0
            # This is just a warning, not an error
            pass


@dataclass
class CollectionInfo:
    """
    Information about a vector collection.

    Attributes:
        name: Collection name
        dimension: Vector dimension
        count: Number of vectors in collection
        metric: Distance metric used
        description: Optional description
    """

    name: str
    dimension: int
    count: int
    metric: str
    description: str | None = None


@dataclass(frozen=True)
class GraphNode:
    """
    Represents a node in a graph database.

    Attributes:
        id: Unique identifier for the node (can be internal DB ID or custom ID)
        label: Node label/type (e.g., "Document", "Chunk", "Entity")
        properties: Node properties as key-value pairs

    Example:
        >>> node = GraphNode(
        ...     id="doc_123",
        ...     label="Document",
        ...     properties={
        ...         "title": "User Manual",
        ...         "source": "manual.pdf",
        ...         "page_count": 42
        ...     }
        ... )
        >>> print(f"{node.label}: {node.properties['title']}")
    """

    id: str
    label: str
    properties: GraphMetadata


@dataclass(frozen=True)
class GraphRelationship:
    """
    Represents a relationship between two nodes in a graph database.

    Attributes:
        id: Unique identifier for the relationship (internal DB ID)
        type: Relationship type (e.g., "HAS_CHUNK", "RELATES_TO", "MENTIONS")
        from_node_id: Source node ID
        to_node_id: Target node ID
        properties: Relationship properties as key-value pairs

    Example:
        >>> rel = GraphRelationship(
        ...     id="rel_456",
        ...     type="HAS_CHUNK",
        ...     from_node_id="doc_123",
        ...     to_node_id="chunk_1",
        ...     properties={"position": 0, "confidence": 0.95}
        ... )
        >>> print(f"{rel.type}: {rel.from_node_id} -> {rel.to_node_id}")
    """

    id: str
    type: str
    from_node_id: str
    to_node_id: str
    properties: GraphMetadata


__all__ = [
    "VectorMetadata",
    "EmbeddingVector",
    "SearchResult",
    "CollectionInfo",
    "GraphMetadata",
    "GraphNode",
    "GraphRelationship",
]
