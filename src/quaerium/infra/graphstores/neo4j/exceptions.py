"""
Exception classes for Neo4j graph store operations.
"""


class Neo4jError(Exception):
    """Base exception for Neo4j operations."""

    pass


class ConnectionError(Neo4jError):
    """Exception raised when connection to Neo4j fails."""

    pass


class QueryError(Neo4jError):
    """Exception raised when query execution fails."""

    pass


class NodeNotFoundError(Neo4jError):
    """Exception raised when a node is not found."""

    pass


class ConstraintError(Neo4jError):
    """Exception raised when constraint operations fail."""

    pass


class IndexError(Neo4jError):
    """Exception raised when index operations fail."""

    pass


__all__ = [
    "Neo4jError",
    "ConnectionError",
    "QueryError",
    "NodeNotFoundError",
    "ConstraintError",
    "IndexError",
]
