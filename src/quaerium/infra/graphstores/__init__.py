"""
Graph store implementations for rag-toolkit.

This module provides graph database integrations for building Graph RAG systems
that combine vector search with knowledge graph capabilities.

Available implementations:
    - Neo4j: Production-ready Neo4j 5.x client with async support

Example:
    >>> from quaerium.infra.graphstores import create_neo4j_service
    >>>
    >>> # Create Neo4j service
    >>> service = create_neo4j_service(
    ...     uri="bolt://localhost:7687",
    ...     user="neo4j",
    ...     password="password"
    ... )
    >>>
    >>> # Create nodes and relationships
    >>> await service.create_node(
    ...     label="Document",
    ...     properties={"id": "doc_1", "title": "User Manual"}
    ... )
    >>> await service.create_relationship(
    ...     from_label="Document",
    ...     from_properties={"id": "doc_1"},
    ...     to_label="Chunk",
    ...     to_properties={"id": "chunk_1"},
    ...     relationship_type="HAS_CHUNK"
    ... )
"""

from .factory import create_neo4j_service
from .neo4j import (
    Neo4jClient,
    Neo4jConfig,
    Neo4jService,
    Neo4jError,
    ConnectionError,
    QueryError,
    NodeNotFoundError,
    ConstraintError,
    IndexError,
)

__all__ = [
    "create_neo4j_service",
    "Neo4jClient",
    "Neo4jConfig",
    "Neo4jService",
    "Neo4jError",
    "ConnectionError",
    "QueryError",
    "NodeNotFoundError",
    "ConstraintError",
    "IndexError",
]
