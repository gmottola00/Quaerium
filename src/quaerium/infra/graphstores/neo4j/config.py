"""
Configuration for Neo4j graph store.
"""

from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """
    Configuration for Neo4j connection.

    Attributes:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687" or "neo4j+s://xxxxx.databases.neo4j.io")
        user: Username for authentication
        password: Password for authentication
        database: Database name (default: "neo4j")
        max_connection_lifetime: Maximum connection lifetime in seconds (default: 3600)
        max_connection_pool_size: Maximum connections in pool (default: 50)
        encrypted: Force TLS encryption, auto-detected from URI if None

    Example:
        >>> config = Neo4jConfig(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password",
        ...     database="neo4j"
        ... )
    """

    uri: str
    user: str
    password: str
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    encrypted: bool | None = None


__all__ = ["Neo4jConfig"]
