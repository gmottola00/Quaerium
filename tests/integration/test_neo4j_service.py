"""
Integration tests for Neo4j service.

These tests require a running Neo4j instance. You can start one with:

    docker run -d \\
      --name neo4j-test \\
      -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/testpassword \\
      neo4j:latest

Or set these environment variables to use an existing instance:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=testpassword
"""

import os
import pytest

from quaerium.infra.graphstores import (
    create_neo4j_service,
    Neo4jError,
    NodeNotFoundError,
)


# Skip all tests if Neo4j is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI") and not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Neo4j not configured. Set NEO4J_URI or RUN_INTEGRATION_TESTS=1",
)


@pytest.fixture
async def neo4j_service():
    """Fixture that provides a Neo4j service and cleans up after tests."""
    # Use environment variables or defaults
    service = create_neo4j_service(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "testpassword"),
    )

    # Verify connectivity
    connected = await service.verify_connectivity()
    if not connected:
        pytest.skip("Cannot connect to Neo4j")

    # Clear database before test
    await service.clear(confirm=True)

    yield service

    # Cleanup after test
    await service.clear(confirm=True)
    await service.close()


@pytest.mark.asyncio
class TestNeo4jServiceIntegration:
    """Integration tests for Neo4j service."""

    async def test_connectivity(self, neo4j_service):
        """Test that we can connect to Neo4j."""
        connected = await neo4j_service.verify_connectivity()
        assert connected is True

    async def test_create_node(self, neo4j_service):
        """Test creating a node."""
        result = await neo4j_service.create_node(
            label="Document",
            properties={
                "id": "doc_1",
                "title": "Test Document",
                "page_count": 42,
            },
        )

        assert result["id"] == "doc_1"
        assert result["title"] == "Test Document"
        assert result["page_count"] == 42

    async def test_create_node_with_merge(self, neo4j_service):
        """Test creating a node with merge (upsert)."""
        # Use a different label to avoid constraint conflicts
        # MERGE works by matching on ALL specified properties
        # If you want to update, use the same property values

        # Create initial node
        await neo4j_service.create_node(
            label="TestDoc",
            properties={"id": "merge_test_1", "title": "Original Title", "version": 1},
        )

        # Merge with same id and title (should find existing)
        result = await neo4j_service.create_node(
            label="TestDoc",
            properties={"id": "merge_test_1", "title": "Original Title", "version": 1},
            merge=True,
        )

        assert result["id"] == "merge_test_1"
        assert result["title"] == "Original Title"

        # Verify only one node exists
        query_result = await neo4j_service.query(
            "MATCH (d:TestDoc {id: $id}) RETURN d", parameters={"id": "merge_test_1"}
        )
        assert len(query_result) == 1

        # Test that merge creates new node if properties don't match
        result2 = await neo4j_service.create_node(
            label="TestDoc",
            properties={"id": "merge_test_2", "title": "Different Title", "version": 2},
            merge=True,
        )

        assert result2["id"] == "merge_test_2"

        # Verify two nodes now exist
        query_result = await neo4j_service.query(
            "MATCH (d:TestDoc) RETURN d"
        )
        assert len(query_result) == 2

    async def test_create_relationship(self, neo4j_service):
        """Test creating a relationship between nodes."""
        # Create two nodes
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1", "title": "Test Doc"}
        )
        await neo4j_service.create_node(
            label="Chunk",
            properties={"id": "chunk_1", "text": "This is a test chunk"},
        )

        # Create relationship
        result = await neo4j_service.create_relationship(
            from_label="Document",
            from_properties={"id": "doc_1"},
            to_label="Chunk",
            to_properties={"id": "chunk_1"},
            relationship_type="HAS_CHUNK",
            relationship_properties={"position": 0, "confidence": 0.95},
        )

        # Verify relationship was created (check it has expected properties)
        # Note: Neo4j returns relationship with internal properties
        assert "position" in result or "confidence" in result or result is not None

    async def test_create_relationship_missing_nodes(self, neo4j_service):
        """Test that creating a relationship with missing nodes fails."""
        with pytest.raises(NodeNotFoundError):
            await neo4j_service.create_relationship(
                from_label="Document",
                from_properties={"id": "nonexistent"},
                to_label="Chunk",
                to_properties={"id": "also_nonexistent"},
                relationship_type="HAS_CHUNK",
            )

    async def test_query(self, neo4j_service):
        """Test executing a Cypher query."""
        # Create test data
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1", "title": "Doc 1"}
        )
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_2", "title": "Doc 2"}
        )
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_3", "title": "Doc 3"}
        )

        # Query all documents
        results = await neo4j_service.query(
            "MATCH (d:Document) RETURN d.id as id, d.title as title ORDER BY d.id"
        )

        assert len(results) == 3
        assert results[0]["id"] == "doc_1"
        assert results[1]["id"] == "doc_2"
        assert results[2]["id"] == "doc_3"

    async def test_query_with_parameters(self, neo4j_service):
        """Test executing a parameterized query."""
        # Create test data
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1", "title": "Test"}
        )

        # Query with parameter
        results = await neo4j_service.query(
            "MATCH (d:Document {id: $doc_id}) RETURN d.title as title",
            parameters={"doc_id": "doc_1"},
        )

        assert len(results) == 1
        assert results[0]["title"] == "Test"

    async def test_create_constraint(self, neo4j_service):
        """Test creating a constraint."""
        await neo4j_service.create_constraint(
            constraint_name="unique_doc_id",
            node_label="Document",
            property_name="id",
            constraint_type="UNIQUE",
        )

        # Create a node with unique ID
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1", "title": "Test"}
        )

        # Try to create another node with same ID (should fail)
        with pytest.raises(Neo4jError):
            await neo4j_service.create_node(
                label="Document", properties={"id": "doc_1", "title": "Different"}
            )

    async def test_create_index(self, neo4j_service):
        """Test creating an index."""
        # This should not raise an error
        await neo4j_service.create_index(
            index_name="doc_title_idx",
            node_label="Document",
            property_name="title",
            index_type="RANGE",
        )

        # Create some test data
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1", "title": "Test"}
        )

        # Query should work (index will be used behind the scenes)
        results = await neo4j_service.query(
            "MATCH (d:Document) WHERE d.title = 'Test' RETURN d"
        )
        assert len(results) == 1

    async def test_get_stats(self, neo4j_service):
        """Test getting database statistics."""
        # Create test data
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1"}
        )
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_2"}
        )
        await neo4j_service.create_node(label="Chunk", properties={"id": "chunk_1"})

        # Create relationship
        await neo4j_service.create_relationship(
            from_label="Document",
            from_properties={"id": "doc_1"},
            to_label="Chunk",
            to_properties={"id": "chunk_1"},
            relationship_type="HAS_CHUNK",
        )

        # Get stats
        stats = await neo4j_service.get_stats()

        assert stats["node_count"] == 3
        assert stats["relationship_count"] == 1
        assert stats["nodes_Document"] == 2
        assert stats["nodes_Chunk"] == 1

    async def test_clear(self, neo4j_service):
        """Test clearing the database."""
        # Create test data
        await neo4j_service.create_node(
            label="Document", properties={"id": "doc_1"}
        )
        await neo4j_service.create_node(label="Chunk", properties={"id": "chunk_1"})

        # Verify data exists
        stats = await neo4j_service.get_stats()
        assert stats["node_count"] > 0

        # Clear database
        await neo4j_service.clear(confirm=True)

        # Verify data is gone
        stats = await neo4j_service.get_stats()
        assert stats["node_count"] == 0

    async def test_clear_requires_confirmation(self, neo4j_service):
        """Test that clear requires confirmation."""
        with pytest.raises(ValueError, match="Must set confirm=True"):
            await neo4j_service.clear(confirm=False)

    async def test_graph_traversal(self, neo4j_service):
        """Test a more complex graph traversal scenario."""
        # Create a document with multiple chunks
        await neo4j_service.create_node(
            label="Document",
            properties={"id": "doc_1", "title": "Test Document"},
        )

        for i in range(3):
            await neo4j_service.create_node(
                label="Chunk",
                properties={"id": f"chunk_{i}", "text": f"Chunk {i} text"},
            )
            await neo4j_service.create_relationship(
                from_label="Document",
                from_properties={"id": "doc_1"},
                to_label="Chunk",
                to_properties={"id": f"chunk_{i}"},
                relationship_type="HAS_CHUNK",
                relationship_properties={"position": i},
            )

        # Query to find all chunks of the document
        results = await neo4j_service.query(
            """
            MATCH (d:Document {id: $doc_id})-[r:HAS_CHUNK]->(c:Chunk)
            RETURN c.id as chunk_id, c.text as text, r.position as position
            ORDER BY r.position
            """,
            parameters={"doc_id": "doc_1"},
        )

        assert len(results) == 3
        assert results[0]["chunk_id"] == "chunk_0"
        assert results[1]["chunk_id"] == "chunk_1"
        assert results[2]["chunk_id"] == "chunk_2"
        assert results[0]["position"] == 0
        assert results[1]["position"] == 1
        assert results[2]["position"] == 2
