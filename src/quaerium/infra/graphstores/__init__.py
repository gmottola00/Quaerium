"""
Graph store implementations for Quaerium.

This module provides graph database integrations for building Graph RAG systems
that combine vector search with knowledge graph capabilities.

Available implementations:
    - Neo4j: Production-ready Neo4j 5.x client with async support (pip install quaerium[neo4j])

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
from __future__ import annotations

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

_LAZY = {
    "create_neo4j_service": ".factory",
    "Neo4jClient": ".neo4j",
    "Neo4jConfig": ".neo4j",
    "Neo4jService": ".neo4j",
    "Neo4jError": ".neo4j",
    "ConnectionError": ".neo4j",
    "QueryError": ".neo4j",
    "NodeNotFoundError": ".neo4j",
    "ConstraintError": ".neo4j",
    "IndexError": ".neo4j",
}


def __getattr__(name: str):
    """Lazy import so optional dependencies are loaded only when used."""
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))
