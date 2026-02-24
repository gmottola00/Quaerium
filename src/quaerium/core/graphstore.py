"""
Core graph store abstraction.

This module defines the Protocol interface that all graph store implementations
must satisfy, enabling seamless switching between different graph databases
(Neo4j, ArangoDB, TigerGraph, etc.) without changing application code.

Design Philosophy:
    - Protocol-based: No inheritance required, duck typing with type safety
    - Provider-agnostic: Works with any graph database
    - Minimal interface: Only essential operations
    - Async-first: All operations are async for better performance
    - Flexible schema: Support for dynamic properties and relationships
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStoreClient(Protocol):
    """
    Protocol for graph store operations.

    This defines the interface that all graph store implementations must provide.
    It abstracts common operations across different graph databases, focusing on
    nodes, relationships, and querying capabilities.

    Implementations:
        - Neo4jService: Neo4j 5.x implementation
        - ArangoDBService: ArangoDB implementation (future)
        - TigerGraphService: TigerGraph implementation (future)

    Example:
        >>> store: GraphStoreClient = create_neo4j_service()
        >>> await store.create_node(
        ...     label="Document",
        ...     properties={"id": "doc_1", "title": "Installation Guide"}
        ... )
        >>> await store.create_relationship(
        ...     from_label="Document",
        ...     from_properties={"id": "doc_1"},
        ...     to_label="Chunk",
        ...     to_properties={"id": "chunk_1"},
        ...     relationship_type="HAS_CHUNK"
        ... )
        >>> results = await store.query(
        ...     "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d, c LIMIT 5"
        ... )
    """

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
        *,
        merge: bool = False,
    ) -> dict[str, Any]:
        """
        Create a node in the graph database.

        Args:
            label: Node label/type (e.g., "Document", "Chunk", "Entity")
            properties: Node properties as key-value pairs
            merge: If True, use MERGE instead of CREATE (upsert behavior)

        Returns:
            Created/merged node data including any auto-generated properties

        Raises:
            GraphError: If node creation fails
            ConnectionError: If database connection is unavailable

        Example:
            >>> node = await store.create_node(
            ...     label="Document",
            ...     properties={
            ...         "id": "doc_123",
            ...         "title": "User Manual",
            ...         "source": "manual.pdf",
            ...         "page_count": 42
            ...     },
            ...     merge=True  # Update if exists
            ... )
            >>> print(node["id"])  # "doc_123"
        """
        ...

    async def create_relationship(
        self,
        from_label: str,
        from_properties: dict[str, Any],
        to_label: str,
        to_properties: dict[str, Any],
        relationship_type: str,
        relationship_properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            from_label: Source node label
            from_properties: Properties to identify source node
            to_label: Target node label
            to_properties: Properties to identify target node
            relationship_type: Type of relationship (e.g., "HAS_CHUNK", "RELATES_TO")
            relationship_properties: Optional properties for the relationship

        Returns:
            Created relationship data

        Raises:
            GraphError: If relationship creation fails
            NodeNotFoundError: If source or target node doesn't exist

        Example:
            >>> rel = await store.create_relationship(
            ...     from_label="Document",
            ...     from_properties={"id": "doc_123"},
            ...     to_label="Chunk",
            ...     to_properties={"id": "chunk_1"},
            ...     relationship_type="HAS_CHUNK",
            ...     relationship_properties={"position": 0, "confidence": 0.95}
            ... )
        """
        ...

    async def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a graph query using the native query language.

        Args:
            query: Query string in native language (e.g., Cypher for Neo4j)
            parameters: Optional query parameters for parameterized queries

        Returns:
            List of result records as dictionaries

        Raises:
            QueryError: If query execution fails
            SyntaxError: If query syntax is invalid

        Example:
            >>> # Find all chunks of a document
            >>> results = await store.query(
            ...     "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
            ...     "RETURN c.id as chunk_id, c.text as text ORDER BY c.position",
            ...     parameters={"doc_id": "doc_123"}
            ... )
            >>> for record in results:
            ...     print(f"Chunk {record['chunk_id']}: {record['text'][:50]}...")
        """
        ...

    async def create_constraint(
        self,
        constraint_name: str,
        node_label: str,
        property_name: str,
        constraint_type: str = "UNIQUE",
    ) -> None:
        """
        Create a constraint on node properties.

        Args:
            constraint_name: Unique name for the constraint
            node_label: Node label to constrain
            property_name: Property to constrain
            constraint_type: Type of constraint (e.g., "UNIQUE", "EXISTS", "NODE_KEY")

        Raises:
            GraphError: If constraint creation fails
            ConstraintExistsError: If constraint already exists

        Example:
            >>> # Ensure document IDs are unique
            >>> await store.create_constraint(
            ...     constraint_name="unique_document_id",
            ...     node_label="Document",
            ...     property_name="id",
            ...     constraint_type="UNIQUE"
            ... )
        """
        ...

    async def create_index(
        self,
        index_name: str,
        node_label: str,
        property_name: str,
        index_type: str = "RANGE",
    ) -> None:
        """
        Create an index on node properties for faster queries.

        Args:
            index_name: Unique name for the index
            node_label: Node label to index
            property_name: Property to index
            index_type: Type of index (e.g., "RANGE", "TEXT", "POINT")

        Raises:
            GraphError: If index creation fails
            IndexExistsError: If index already exists

        Example:
            >>> # Index document titles for text search
            >>> await store.create_index(
            ...     index_name="document_title_index",
            ...     node_label="Document",
            ...     property_name="title",
            ...     index_type="TEXT"
            ... )
        """
        ...

    async def get_stats(self) -> dict[str, int]:
        """
        Get graph database statistics.

        Returns:
            Dictionary with database stats:
                - "node_count": Total number of nodes
                - "relationship_count": Total number of relationships
                - Additional provider-specific stats

        Example:
            >>> stats = await store.get_stats()
            >>> print(f"Graph has {stats['node_count']} nodes")
            >>> print(f"Graph has {stats['relationship_count']} relationships")
        """
        ...

    async def clear(self, confirm: bool = False) -> None:
        """
        Clear all data from the graph database.

        Args:
            confirm: Must be True to actually clear (safety measure)

        Raises:
            ValueError: If confirm is not True
            GraphError: If clear operation fails

        Warning:
            This operation is irreversible! Use only for testing/development.

        Example:
            >>> # Clear test database
            >>> await store.clear(confirm=True)
        """
        ...


__all__ = ["GraphStoreClient"]
