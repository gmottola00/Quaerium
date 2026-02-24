"""
Neo4j service implementation of GraphStoreClient protocol.
"""

import logging
from typing import Any

from quaerium.core.graphstore import GraphStoreClient

from .client import Neo4jClient
from .config import Neo4jConfig
from .exceptions import Neo4jError, NodeNotFoundError
from .utils import convert_neo4j_types

logger = logging.getLogger(__name__)


class Neo4jService:
    """
    Neo4j implementation of GraphStoreClient protocol.

    This service wraps the Neo4jClient and provides a protocol-compliant interface
    for graph operations. It handles type conversions and provides higher-level
    operations on top of the raw client.

    Example:
        >>> config = Neo4jConfig(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>> service = Neo4jService(config)
        >>> await service.create_node(
        ...     label="Document",
        ...     properties={"id": "doc_1", "title": "Guide"}
        ... )
    """

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j service.

        Args:
            config: Neo4j configuration
        """
        self.config = config
        self.client = Neo4jClient(
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database,
            max_connection_lifetime=config.max_connection_lifetime,
            max_connection_pool_size=config.max_connection_pool_size,
            encrypted=config.encrypted,
        )

    async def verify_connectivity(self) -> bool:
        """
        Verify connection to Neo4j.

        Returns:
            True if connection successful, False otherwise
        """
        return await self.client.verify_connectivity()

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
            Neo4jError: If node creation fails
        """
        try:
            # Build property string for Cypher
            prop_params = {f"prop_{k}": v for k, v in properties.items()}
            prop_string = ", ".join(
                f"{k}: ${param}" for k, param in zip(properties.keys(), prop_params.keys())
            )

            operation = "MERGE" if merge else "CREATE"
            query = f"""
            {operation} (n:{label} {{{prop_string}}})
            RETURN n
            """

            result = await self.client.execute_query(query, parameters=prop_params)

            if not result:
                raise Neo4jError(f"Failed to create node with label {label}")

            # Extract node properties and convert Neo4j types
            node_data = result[0]["n"]
            return convert_neo4j_types(node_data)

        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise Neo4jError(f"Failed to create node: {e}") from e

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
            Neo4jError: If relationship creation fails
            NodeNotFoundError: If source or target node doesn't exist
        """
        try:
            # Build parameters
            from_params = {f"from_{k}": v for k, v in from_properties.items()}
            to_params = {f"to_{k}": v for k, v in to_properties.items()}

            # Build match patterns
            from_pattern = ", ".join(
                f"{k}: ${param}"
                for k, param in zip(from_properties.keys(), from_params.keys())
            )
            to_pattern = ", ".join(
                f"{k}: ${param}"
                for k, param in zip(to_properties.keys(), to_params.keys())
            )

            # Build relationship properties
            rel_props = relationship_properties or {}
            rel_params = {f"rel_{k}": v for k, v in rel_props.items()}
            rel_prop_string = (
                "{" + ", ".join(f"{k}: ${param}" for k, param in zip(rel_props.keys(), rel_params.keys())) + "}"
                if rel_props
                else ""
            )

            # Combine all parameters
            all_params = {**from_params, **to_params, **rel_params}

            query = f"""
            MATCH (a:{from_label} {{{from_pattern}}})
            MATCH (b:{to_label} {{{to_pattern}}})
            CREATE (a)-[r:{relationship_type} {rel_prop_string}]->(b)
            RETURN r, a, b
            """

            result = await self.client.execute_query(query, parameters=all_params)

            if not result:
                raise NodeNotFoundError(
                    f"Could not find nodes to create relationship: "
                    f"{from_label}({from_properties}) -> {to_label}({to_properties})"
                )

            # Extract relationship data and convert Neo4j types
            rel_data = result[0]["r"]
            return convert_neo4j_types(rel_data)

        except NodeNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise Neo4jError(f"Failed to create relationship: {e}") from e

    async def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a graph query using Cypher query language.

        Args:
            query: Cypher query string
            parameters: Optional query parameters for parameterized queries

        Returns:
            List of result records as dictionaries

        Raises:
            Neo4jError: If query execution fails
        """
        try:
            result = await self.client.execute_query(query, parameters)
            return [convert_neo4j_types(record) for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise Neo4jError(f"Query failed: {e}") from e

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
            Neo4jError: If constraint creation fails
        """
        try:
            await self.client.create_constraint(
                constraint_name, node_label, property_name, constraint_type
            )
        except Exception as e:
            logger.error(f"Failed to create constraint: {e}")
            raise Neo4jError(f"Failed to create constraint: {e}") from e

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
            Neo4jError: If index creation fails
        """
        try:
            await self.client.create_index(
                index_name, node_label, property_name, index_type
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise Neo4jError(f"Failed to create index: {e}") from e

    async def get_stats(self) -> dict[str, int]:
        """
        Get graph database statistics.

        Returns:
            Dictionary with database stats including node and relationship counts
        """
        try:
            stats = await self.client.get_database_stats()

            # Normalize stats to match protocol expectations
            node_count = sum(v for k, v in stats.items() if k.startswith("nodes_"))
            relationship_count = stats.get("relationships_total", 0)

            return {
                "node_count": node_count,
                "relationship_count": relationship_count,
                **stats,  # Include detailed stats as well
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise Neo4jError(f"Failed to get stats: {e}") from e

    async def clear(self, confirm: bool = False) -> None:
        """
        Clear all data from the graph database.

        Args:
            confirm: Must be True to actually clear (safety measure)

        Raises:
            ValueError: If confirm is not True
            Neo4jError: If clear operation fails

        Warning:
            This operation is irreversible! Use only for testing/development.
        """
        try:
            await self.client.clear_database(confirm=confirm)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise Neo4jError(f"Failed to clear database: {e}") from e

    async def close(self) -> None:
        """Close the Neo4j connection."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.verify_connectivity()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Verify that Neo4jService implements GraphStoreClient protocol
_: GraphStoreClient = Neo4jService  # type: ignore


__all__ = ["Neo4jService"]
