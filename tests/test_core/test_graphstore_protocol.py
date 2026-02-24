"""
Unit tests for GraphStoreClient protocol.

These tests verify protocol compliance using mocks, without requiring a running
Neo4j instance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from quaerium.core.graphstore import GraphStoreClient
from quaerium.infra.graphstores import Neo4jService, Neo4jConfig


class TestGraphStoreProtocol:
    """Test that implementations satisfy the GraphStoreClient protocol."""

    def test_neo4j_service_implements_protocol(self):
        """Test that Neo4jService implements GraphStoreClient protocol."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Check that service implements the protocol
        assert isinstance(service, GraphStoreClient)

    @pytest.mark.asyncio
    async def test_create_node_interface(self):
        """Test create_node method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's execute_query method
        service.client.execute_query = AsyncMock(
            return_value=[{"n": {"id": "doc_1", "title": "Test"}}]
        )

        result = await service.create_node(
            label="Document",
            properties={"id": "doc_1", "title": "Test"},
        )

        assert result == {"id": "doc_1", "title": "Test"}
        service.client.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_node_with_merge(self):
        """Test create_node with merge option."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's execute_query method
        service.client.execute_query = AsyncMock(
            return_value=[{"n": {"id": "doc_1", "title": "Test"}}]
        )

        result = await service.create_node(
            label="Document",
            properties={"id": "doc_1", "title": "Test"},
            merge=True,
        )

        assert result == {"id": "doc_1", "title": "Test"}
        # Verify MERGE was used in the query
        call_args = service.client.execute_query.call_args
        assert "MERGE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_relationship_interface(self):
        """Test create_relationship method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's execute_query method
        service.client.execute_query = AsyncMock(
            return_value=[
                {
                    "r": {"type": "HAS_CHUNK", "position": 0},
                    "a": {"id": "doc_1"},
                    "b": {"id": "chunk_1"},
                }
            ]
        )

        result = await service.create_relationship(
            from_label="Document",
            from_properties={"id": "doc_1"},
            to_label="Chunk",
            to_properties={"id": "chunk_1"},
            relationship_type="HAS_CHUNK",
            relationship_properties={"position": 0},
        )

        assert result == {"type": "HAS_CHUNK", "position": 0}
        service.client.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_interface(self):
        """Test query method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's execute_query method
        service.client.execute_query = AsyncMock(
            return_value=[
                {"id": "doc_1", "title": "Test 1"},
                {"id": "doc_2", "title": "Test 2"},
            ]
        )

        results = await service.query(
            "MATCH (n:Document) RETURN n.id as id, n.title as title",
            parameters={},
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc_1"
        assert results[1]["id"] == "doc_2"

    @pytest.mark.asyncio
    async def test_create_constraint_interface(self):
        """Test create_constraint method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's create_constraint method
        service.client.create_constraint = AsyncMock()

        await service.create_constraint(
            constraint_name="unique_doc_id",
            node_label="Document",
            property_name="id",
            constraint_type="UNIQUE",
        )

        service.client.create_constraint.assert_called_once_with(
            "unique_doc_id", "Document", "id", "UNIQUE"
        )

    @pytest.mark.asyncio
    async def test_create_index_interface(self):
        """Test create_index method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's create_index method
        service.client.create_index = AsyncMock()

        await service.create_index(
            index_name="doc_title_idx",
            node_label="Document",
            property_name="title",
            index_type="TEXT",
        )

        service.client.create_index.assert_called_once_with(
            "doc_title_idx", "Document", "title", "TEXT"
        )

    @pytest.mark.asyncio
    async def test_get_stats_interface(self):
        """Test get_stats method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's get_database_stats method
        service.client.get_database_stats = AsyncMock(
            return_value={
                "nodes_Document": 10,
                "nodes_Chunk": 50,
                "relationships_total": 50,
            }
        )

        stats = await service.get_stats()

        assert "node_count" in stats
        assert "relationship_count" in stats
        assert stats["node_count"] == 60  # 10 + 50
        assert stats["relationship_count"] == 50

    @pytest.mark.asyncio
    async def test_clear_interface(self):
        """Test clear method interface."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's clear_database method
        service.client.clear_database = AsyncMock()

        await service.clear(confirm=True)

        service.client.clear_database.assert_called_once_with(confirm=True)

    @pytest.mark.asyncio
    async def test_clear_requires_confirmation(self):
        """Test that clear requires confirmation."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        service = Neo4jService(config)

        # Mock the client's clear_database method to raise ValueError
        service.client.clear_database = AsyncMock(
            side_effect=ValueError("Must set confirm=True to clear database")
        )

        with pytest.raises(ValueError, match="Must set confirm=True"):
            await service.clear(confirm=False)


class TestTypeConversions:
    """Test Neo4j type conversions."""

    def test_convert_neo4j_datetime(self):
        """Test conversion of Neo4j DateTime types."""
        from neo4j.time import DateTime
        from quaerium.infra.graphstores.neo4j import convert_neo4j_types

        dt = DateTime(2024, 1, 15, 10, 30, 0)
        result = convert_neo4j_types(dt)

        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_convert_neo4j_date(self):
        """Test conversion of Neo4j Date types."""
        from neo4j.time import Date
        from quaerium.infra.graphstores.neo4j import convert_neo4j_types

        date = Date(2024, 1, 15)
        result = convert_neo4j_types(date)

        assert isinstance(result, str)
        assert result == "2024-01-15"

    def test_convert_nested_dict(self):
        """Test conversion of nested dictionaries."""
        from neo4j.time import DateTime
        from quaerium.infra.graphstores.neo4j import convert_neo4j_types

        dt = DateTime(2024, 1, 15, 10, 30, 0)
        data = {
            "id": "doc_1",
            "created": dt,
            "metadata": {"nested_date": dt, "count": 42},
        }

        result = convert_neo4j_types(data)

        assert result["id"] == "doc_1"
        assert isinstance(result["created"], str)
        assert isinstance(result["metadata"]["nested_date"], str)
        assert result["metadata"]["count"] == 42

    def test_convert_list(self):
        """Test conversion of lists."""
        from neo4j.time import Date
        from quaerium.infra.graphstores.neo4j import convert_neo4j_types

        date = Date(2024, 1, 15)
        data = [{"date": date}, {"date": date}]

        result = convert_neo4j_types(data)

        assert len(result) == 2
        assert isinstance(result[0]["date"], str)
        assert isinstance(result[1]["date"], str)
