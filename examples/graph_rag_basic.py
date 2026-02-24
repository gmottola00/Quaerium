"""
Basic Graph RAG Example with Neo4j.

This example demonstrates:
1. Connecting to Neo4j
2. Creating a document with chunks
3. Linking chunks to entities
4. Querying the graph
5. Graph traversal for retrieval

Requirements:
    pip install rag-toolkit[neo4j]

Setup:
    docker run -d \\
      --name neo4j-dev \\
      -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/password \\
      neo4j:latest

Environment variables (optional):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password
"""

import asyncio
import os

from quaerium.infra.graphstores import create_neo4j_service


async def setup_schema(service):
    """Set up graph schema with constraints and indexes."""
    print("\n📋 Setting up graph schema...")

    # Create unique constraints
    await service.create_constraint(
        constraint_name="unique_document_id",
        node_label="Document",
        property_name="id",
        constraint_type="UNIQUE",
    )

    await service.create_constraint(
        constraint_name="unique_chunk_id",
        node_label="Chunk",
        property_name="id",
        constraint_type="UNIQUE",
    )

    await service.create_constraint(
        constraint_name="unique_entity_name",
        node_label="Entity",
        property_name="name",
        constraint_type="UNIQUE",
    )

    # Create indexes for text search
    await service.create_index(
        index_name="document_title_idx",
        node_label="Document",
        property_name="title",
        index_type="TEXT",
    )

    await service.create_index(
        index_name="chunk_text_idx",
        node_label="Chunk",
        property_name="text",
        index_type="TEXT",
    )

    print("✅ Schema created (constraints and indexes)")


async def index_document(service, doc_id, title, chunks):
    """Index a document with its chunks in the graph."""
    print(f"\n📄 Indexing document: {title}")

    # Create document node
    await service.create_node(
        label="Document",
        properties={
            "id": doc_id,
            "title": title,
            "source": "example.pdf",
            "page_count": 5,
        },
        merge=True,
    )
    print(f"  ✅ Created document node: {doc_id}")

    # Create chunks and link to document
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"

        # Create chunk node
        await service.create_node(
            label="Chunk",
            properties={"id": chunk_id, "text": chunk_text, "position": i},
            merge=True,
        )

        # Link document to chunk
        await service.create_relationship(
            from_label="Document",
            from_properties={"id": doc_id},
            to_label="Chunk",
            to_properties={"id": chunk_id},
            relationship_type="HAS_CHUNK",
            relationship_properties={"position": i},
        )

        print(f"  ✅ Created chunk {i}: {chunk_text[:50]}...")


async def extract_and_link_entities(service, chunk_id, entities):
    """Extract entities from a chunk and create relationships."""
    print(f"\n🔗 Linking entities for {chunk_id}...")

    for entity in entities:
        # Create or merge entity node
        await service.create_node(
            label="Entity",
            properties={
                "name": entity["name"],
                "type": entity["type"],
            },
            merge=True,
        )

        # Link chunk to entity
        await service.create_relationship(
            from_label="Chunk",
            from_properties={"id": chunk_id},
            to_label="Entity",
            to_properties={"name": entity["name"]},
            relationship_type="MENTIONS",
            relationship_properties={"confidence": entity.get("confidence", 1.0)},
        )

        print(f"  ✅ Linked entity: {entity['name']} ({entity['type']})")


async def query_document_structure(service, doc_id):
    """Query to see document structure."""
    print(f"\n🔍 Querying document structure for {doc_id}...")

    results = await service.query(
        """
        MATCH (d:Document {id: $doc_id})-[r:HAS_CHUNK]->(c:Chunk)
        RETURN d.title as title, c.id as chunk_id, c.text as text, r.position as position
        ORDER BY r.position
        """,
        parameters={"doc_id": doc_id},
    )

    print(f"\n📊 Found {len(results)} chunks:")
    for record in results:
        print(f"  • Chunk {record['position']}: {record['text'][:60]}...")


async def query_entities(service):
    """Query to find entities and their mentions."""
    print("\n🔍 Querying entities...")

    results = await service.query(
        """
        MATCH (e:Entity)<-[r:MENTIONS]-(c:Chunk)
        RETURN e.name as entity,
               e.type as type,
               count(c) as mention_count,
               collect(c.id) as mentioned_in
        ORDER BY mention_count DESC
        """
    )

    print(f"\n📊 Found {len(results)} entities:")
    for record in results:
        print(
            f"  • {record['entity']} ({record['type']}): "
            f"mentioned {record['mention_count']} times"
        )


async def graph_traversal_example(service, entity_name):
    """Example of graph traversal: find all documents mentioning an entity."""
    print(f"\n🔍 Finding documents mentioning '{entity_name}'...")

    results = await service.query(
        """
        MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        RETURN DISTINCT d.id as doc_id,
               d.title as title,
               collect(DISTINCT c.id) as chunks
        """,
        parameters={"entity_name": entity_name},
    )

    print(f"\n📊 Found {len(results)} documents:")
    for record in results:
        print(f"  • {record['title']} (ID: {record['doc_id']})")
        print(f"    Mentioned in {len(record['chunks'])} chunks")


async def get_statistics(service):
    """Get and display graph statistics."""
    print("\n📊 Database Statistics:")

    stats = await service.get_stats()

    print(f"  • Total nodes: {stats['node_count']}")
    print(f"  • Total relationships: {stats['relationship_count']}")
    print(f"  • Documents: {stats.get('nodes_Document', 0)}")
    print(f"  • Chunks: {stats.get('nodes_Chunk', 0)}")
    print(f"  • Entities: {stats.get('nodes_Entity', 0)}")


async def main():
    """Run the complete Graph RAG example."""
    print("🚀 Graph RAG Basic Example with Neo4j\n")
    print("=" * 60)

    # Create Neo4j service
    # Uses environment variables or defaults
    service = create_neo4j_service(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    try:
        # Verify connectivity
        print("\n🔌 Connecting to Neo4j...")
        connected = await service.verify_connectivity()
        if not connected:
            print("❌ Cannot connect to Neo4j. Is it running?")
            print("\nStart Neo4j with:")
            print("  docker run -d --name neo4j-dev -p 7474:7474 -p 7687:7687 \\")
            print("    -e NEO4J_AUTH=neo4j/password neo4j:latest")
            return

        print("✅ Connected to Neo4j")

        # Clear database for clean example
        print("\n🧹 Clearing database for fresh start...")
        await service.clear(confirm=True)

        # Set up schema
        await setup_schema(service)

        # Index a sample document
        doc_chunks = [
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
            "It enables language models to access external knowledge bases.",
            "Vector embeddings are used to represent text semantically. "
            "Libraries like OpenAI and Sentence-Transformers provide pre-trained models.",
            "Graph databases like Neo4j store relationships between entities. "
            "This enables traversing connections for richer context retrieval.",
        ]

        await index_document(
            service,
            doc_id="doc_rag_intro",
            title="Introduction to RAG Systems",
            chunks=doc_chunks,
        )

        # Extract and link entities
        await extract_and_link_entities(
            service,
            chunk_id="doc_rag_intro_chunk_0",
            entities=[
                {"name": "RAG", "type": "CONCEPT", "confidence": 0.95},
                {"name": "Language Models", "type": "CONCEPT", "confidence": 0.90},
            ],
        )

        await extract_and_link_entities(
            service,
            chunk_id="doc_rag_intro_chunk_1",
            entities=[
                {"name": "OpenAI", "type": "ORGANIZATION", "confidence": 0.98},
                {"name": "Sentence-Transformers", "type": "LIBRARY", "confidence": 0.95},
            ],
        )

        await extract_and_link_entities(
            service,
            chunk_id="doc_rag_intro_chunk_2",
            entities=[
                {"name": "Neo4j", "type": "DATABASE", "confidence": 0.99},
                {"name": "Graph Databases", "type": "CONCEPT", "confidence": 0.92},
            ],
        )

        # Query examples
        await query_document_structure(service, "doc_rag_intro")
        await query_entities(service)
        await graph_traversal_example(service, "Neo4j")

        # Get statistics
        await get_statistics(service)

        # Show how to access Neo4j Browser
        print("\n" + "=" * 60)
        print("\n🌐 Explore your graph in Neo4j Browser:")
        print("  URL: http://localhost:7474")
        print("  Connect: bolt://localhost:7687")
        print("  Credentials: neo4j / password")
        print("\n  Try these queries:")
        print("    MATCH (n) RETURN n LIMIT 25")
        print("    MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)")
        print("    RETURN d, r, c")
        print("    MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)")
        print("    RETURN e, c")

        print("\n✅ Example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Closing connection...")
        await service.close()
        print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
