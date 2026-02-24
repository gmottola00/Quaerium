"""
Factory functions for creating graph store clients.
"""

import os
from typing import Optional

from quaerium.core.graphstore import GraphStoreClient

from .neo4j.config import Neo4jConfig
from .neo4j.service import Neo4jService


def create_neo4j_service(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "neo4j",
    max_connection_lifetime: int = 3600,
    max_connection_pool_size: int = 50,
    encrypted: bool | None = None,
) -> GraphStoreClient:
    """
    Create a Neo4j graph store client.

    Args:
        uri: Neo4j connection URI. If None, reads from NEO4J_URI env var.
            Examples: "bolt://localhost:7687", "neo4j+s://xxxxx.databases.neo4j.io"
        user: Username for authentication. If None, reads from NEO4J_USER env var.
        password: Password for authentication. If None, reads from NEO4J_PASSWORD env var.
        database: Database name (default: "neo4j")
        max_connection_lifetime: Maximum connection lifetime in seconds
        max_connection_pool_size: Maximum connections in pool
        encrypted: Force TLS encryption, auto-detected from URI if None

    Returns:
        Neo4jService instance implementing GraphStoreClient protocol

    Raises:
        ValueError: If required connection parameters are missing

    Example:
        >>> # Using explicit parameters
        >>> service = create_neo4j_service(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>>
        >>> # Using environment variables
        >>> # NEO4J_URI=bolt://localhost:7687
        >>> # NEO4J_USER=neo4j
        >>> # NEO4J_PASSWORD=password
        >>> service = create_neo4j_service()
        >>>
        >>> # Use the service
        >>> await service.create_node(
        ...     label="Document",
        ...     properties={"id": "doc_1", "title": "Guide"}
        ... )
    """
    # Read from environment variables if not provided
    uri = uri or os.getenv("NEO4J_URI")
    user = user or os.getenv("NEO4J_USER")
    password = password or os.getenv("NEO4J_PASSWORD")

    # Validate required parameters
    if not uri:
        raise ValueError(
            "Neo4j URI is required. Provide 'uri' parameter or set NEO4J_URI environment variable."
        )
    if not user:
        raise ValueError(
            "Neo4j user is required. Provide 'user' parameter or set NEO4J_USER environment variable."
        )
    if not password:
        raise ValueError(
            "Neo4j password is required. Provide 'password' parameter or set NEO4J_PASSWORD environment variable."
        )

    config = Neo4jConfig(
        uri=uri,
        user=user,
        password=password,
        database=database,
        max_connection_lifetime=max_connection_lifetime,
        max_connection_pool_size=max_connection_pool_size,
        encrypted=encrypted,
    )

    return Neo4jService(config)


__all__ = ["create_neo4j_service"]
