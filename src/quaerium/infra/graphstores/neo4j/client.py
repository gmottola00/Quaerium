"""
Generic Neo4j client for Knowledge Graph operations.

This is a generic, reusable Neo4j client that can work with any domain.
Domain-specific logic (schema, queries) should go in separate files.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from .exceptions import ConnectionError as Neo4jConnectionError, QueryError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Generic async Neo4j client for Knowledge Graph operations.

    Supports both:
    - Local development (Docker with bolt://)
    - Production cloud (Neo4j Aura with neo4j+s://)

    Handles connection pooling, query execution, and error handling.
    This class is domain-agnostic and can be reused across projects.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        encrypted: bool | None = None,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
                - Local: bolt://localhost:7687
                - Aura: neo4j+s://xxxxx.databases.neo4j.io
            user: Username for authentication
            password: Password for authentication
            database: Database name (default: neo4j)
            max_connection_lifetime: Max connection lifetime in seconds
            max_connection_pool_size: Max connections in pool
            encrypted: Force TLS encryption (auto-detected from URI if None)
        """
        self.uri = uri
        self.user = user
        self.database = database
        self.is_aura = uri.startswith("neo4j+s://")

        self._driver: Optional[AsyncDriver] = None
        self._max_connection_lifetime = max_connection_lifetime
        self._max_connection_pool_size = max_connection_pool_size

        # Create driver with appropriate settings
        try:
            driver_config = {
                "auth": (user, password),
                "max_connection_lifetime": max_connection_lifetime,
                "max_connection_pool_size": max_connection_pool_size,
            }

            # For bolt:// URIs, we can specify encryption settings
            # For neo4j+s:// URIs, encryption is already in the scheme
            if not self.is_aura and encrypted:
                driver_config["encrypted"] = encrypted

            if self.is_aura:
                logger.info(f"Connecting to Neo4j Aura (cloud): {uri}")
            else:
                logger.info(f"Connecting to Neo4j local: {uri}")

            self._driver = AsyncGraphDatabase.driver(uri, **driver_config)
            logger.info("Neo4j driver created successfully")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            raise Neo4jConnectionError(f"Failed to create Neo4j driver: {e}") from e

    async def verify_connectivity(self) -> bool:
        """
        Verify connection to Neo4j.

        Returns:
            True if connection successful, False otherwise
        """
        if not self._driver:
            return False

        try:
            await self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            return False
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """
        Context manager for Neo4j session.

        Usage:
            async with client.session() as session:
                result = await session.run("MATCH (n) RETURN n LIMIT 1")

        Yields:
            AsyncSession
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")

        async with self._driver.session(database=self.database) as session:
            yield session

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters (default: None)

        Returns:
            List of result records as dictionaries

        Raises:
            QueryError: If query execution fails

        Example:
            results = await client.execute_query(
                "MATCH (n:Node {id: $id}) RETURN n",
                parameters={"id": "123"}
            )
        """
        if not parameters:
            parameters = {}

        try:
            async with self.session() as session:
                result = await session.run(query, parameters)
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query[:200]}...")
            raise QueryError(f"Query execution failed: {e}") from e

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a write query (CREATE, MERGE, DELETE, etc.).

        Args:
            query: Cypher write query
            parameters: Query parameters

        Returns:
            Summary statistics from Neo4j

        Raises:
            QueryError: If write query fails

        Example:
            summary = await client.execute_write(
                "CREATE (n:Node {id: $id, name: $name})",
                parameters={"id": "123", "name": "Example"}
            )
        """
        if not parameters:
            parameters = {}

        try:
            async with self.session() as session:
                result = await session.run(query, parameters)
                summary = await result.consume()

                stats = {
                    "nodes_created": summary.counters.nodes_created,
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_created": summary.counters.relationships_created,
                    "relationships_deleted": summary.counters.relationships_deleted,
                    "properties_set": summary.counters.properties_set,
                }

                logger.debug(f"Write query executed: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Write query failed: {e}")
            logger.error(f"Query: {query}")
            raise QueryError(f"Write query failed: {e}") from e

    async def create_constraint(
        self,
        constraint_name: str,
        node_label: str,
        property_name: str,
        constraint_type: str = "UNIQUE",
    ) -> None:
        """
        Create a constraint on a node property.

        Args:
            constraint_name: Name for the constraint
            node_label: Node label (e.g., "Person", "Product")
            property_name: Property to constrain (e.g., "email", "code")
            constraint_type: Type of constraint ("UNIQUE", "EXISTS", "NODE_KEY")

        Example:
            await client.create_constraint(
                "user_email_unique",
                "User",
                "email",
                "UNIQUE"
            )
        """
        if constraint_type == "UNIQUE":
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{node_label})
            REQUIRE n.{property_name} IS UNIQUE
            """
        else:
            raise ValueError(f"Constraint type {constraint_type} not yet supported")

        try:
            await self.execute_write(query)
            logger.info(f"Created constraint: {constraint_name}")
        except Exception as e:
            logger.warning(f"Constraint creation failed (may already exist): {e}")

    async def create_index(
        self,
        index_name: str,
        node_label: str,
        property_name: str,
        index_type: str = "RANGE",
    ) -> None:
        """
        Create an index on a node property.

        Args:
            index_name: Name for the index
            node_label: Node label
            property_name: Property to index
            index_type: Type of index ("RANGE", "TEXT", "POINT")

        Example:
            await client.create_index(
                "product_name_idx",
                "Product",
                "name",
                "TEXT"
            )
        """
        query = f"""
        CREATE {index_type} INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON (n.{property_name})
        """

        try:
            await self.execute_write(query)
            logger.info(f"Created index: {index_name}")
        except Exception as e:
            logger.warning(f"Index creation failed (may already exist): {e}")

    async def clear_database(self, confirm: bool = False) -> None:
        """
        Delete all nodes and relationships.

        **WARNING**: This is destructive and irreversible!

        Args:
            confirm: Must be True to execute (safety check)
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear database")

        logger.warning("Clearing entire Neo4j database!")
        await self.execute_write("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")

    async def get_database_stats(self) -> dict[str, int]:
        """
        Get database statistics (node/relationship counts).

        Returns:
            Dictionary with counts per label/type
        """
        stats = {}

        # Node counts by label
        node_labels = await self.execute_query(
            "CALL db.labels() YIELD label RETURN label"
        )

        for record in node_labels:
            label = record["label"]
            count_result = await self.execute_query(
                f"MATCH (n:{label}) RETURN count(n) as count"
            )
            stats[f"nodes_{label}"] = count_result[0]["count"]

        # Total relationship count
        rel_count = await self.execute_query(
            "MATCH ()-[r]->() RETURN count(r) as count"
        )
        stats["relationships_total"] = rel_count[0]["count"] if rel_count else 0

        return stats

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j driver closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.verify_connectivity()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


__all__ = ["Neo4jClient"]
