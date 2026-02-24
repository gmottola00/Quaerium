# Graph RAG with Neo4j

Graph RAG combines traditional vector search with knowledge graph capabilities to build more powerful retrieval systems. This guide shows you how to use quaerium's Neo4j integration to add graph capabilities to your RAG applications.

## What is Graph RAG?

Traditional RAG systems use **vector similarity search** to find relevant documents. While powerful, this approach has limitations:

- **Lost structure**: Vector embeddings don't capture relationships between entities, documents, and chunks
- **Limited context**: Each chunk is independent, missing important connections
- **No graph reasoning**: Can't traverse relationships like "Document → Chunks → Entities → Related Documents"

**Graph RAG** addresses these limitations by:

1. **Storing structured data**: Documents, chunks, and entities as nodes with relationships
2. **Enabling graph traversal**: Follow relationships to discover connected information
3. **Combining vector + graph**: Use both similarity search and relationship-based retrieval

## When to Use Graph RAG

Graph RAG is particularly valuable when:

- ✅ Your data has inherent **relationships** (e.g., papers citing papers, products with components)
- ✅ You need **multi-hop reasoning** (e.g., "Find papers citing works by Author X")
- ✅ You want **structured metadata** alongside vector embeddings
- ✅ You're building **knowledge bases** that benefit from explicit connections

Graph RAG may be **overkill** if:

- ❌ Your documents are independent with no relationships
- ❌ Simple vector search already meets your needs
- ❌ You want to minimize infrastructure complexity (vector DB is simpler)

## Installation

Install quaerium with Neo4j support:

```bash
pip install quaerium[neo4j]
```

Or with uv:

```bash
uv pip install quaerium[neo4j]
```

## Quick Start

### 1. Start Neo4j

The easiest way to get started is with Docker:

```bash
docker run -d \
  --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

This starts Neo4j with:
- **Browser UI**: http://localhost:7474
- **Bolt protocol**: bolt://localhost:7687
- **Credentials**: neo4j / password

For production, consider [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) (managed cloud).

### 2. Create a Graph Store Client

```python
from quaerium.infra.graphstores import create_neo4j_service

# Using explicit credentials
service = create_neo4j_service(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Or use environment variables
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password
service = create_neo4j_service()

# Verify connection
if await service.verify_connectivity():
    print("Connected to Neo4j!")
```

### 3. Create Nodes and Relationships

```python
# Create a document node
doc = await service.create_node(
    label="Document",
    properties={
        "id": "doc_123",
        "title": "Introduction to RAG",
        "source": "paper.pdf",
        "page_count": 15
    }
)

# Create chunk nodes
chunk1 = await service.create_node(
    label="Chunk",
    properties={
        "id": "chunk_1",
        "text": "RAG combines retrieval with generation...",
        "position": 0
    }
)

chunk2 = await service.create_node(
    label="Chunk",
    properties={
        "id": "chunk_2",
        "text": "Vector embeddings enable semantic search...",
        "position": 1
    }
)

# Create relationships
await service.create_relationship(
    from_label="Document",
    from_properties={"id": "doc_123"},
    to_label="Chunk",
    to_properties={"id": "chunk_1"},
    relationship_type="HAS_CHUNK",
    relationship_properties={"position": 0}
)

await service.create_relationship(
    from_label="Document",
    from_properties={"id": "doc_123"},
    to_label="Chunk",
    to_properties={"id": "chunk_2"},
    relationship_type="HAS_CHUNK",
    relationship_properties={"position": 1}
)
```

### 4. Query the Graph

Use Cypher (Neo4j's query language) to traverse the graph:

```python
# Find all chunks of a document
results = await service.query(
    """
    MATCH (d:Document {id: $doc_id})-[r:HAS_CHUNK]->(c:Chunk)
    RETURN c.id, c.text, r.position
    ORDER BY r.position
    """,
    parameters={"doc_id": "doc_123"}
)

for record in results:
    print(f"Chunk {record['c.id']}: {record['c.text'][:50]}...")
```

## Common Patterns

### Pattern 1: Document → Chunks

The most basic graph structure for RAG:

```python
async def index_document_with_chunks(service, doc_id, title, chunks):
    """Index a document and its chunks in the graph."""

    # Create document node
    await service.create_node(
        label="Document",
        properties={"id": doc_id, "title": title},
        merge=True  # Update if exists
    )

    # Create chunks and relationships
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"

        # Create chunk node
        await service.create_node(
            label="Chunk",
            properties={
                "id": chunk_id,
                "text": chunk_text,
                "doc_id": doc_id
            },
            merge=True
        )

        # Link document to chunk
        await service.create_relationship(
            from_label="Document",
            from_properties={"id": doc_id},
            to_label="Chunk",
            to_properties={"id": chunk_id},
            relationship_type="HAS_CHUNK",
            relationship_properties={"position": i}
        )

# Usage
await index_document_with_chunks(
    service,
    doc_id="doc_123",
    title="RAG Guide",
    chunks=["Chunk 1 text...", "Chunk 2 text...", "Chunk 3 text..."]
)
```

### Pattern 2: Entity Extraction and Linking

Extract entities from chunks and create relationships:

```python
async def link_entities(service, chunk_id, entities):
    """Link a chunk to extracted entities."""

    for entity in entities:
        # Create or merge entity node
        await service.create_node(
            label="Entity",
            properties={
                "name": entity["name"],
                "type": entity["type"]  # e.g., "PERSON", "ORG", "LOCATION"
            },
            merge=True
        )

        # Link chunk to entity
        await service.create_relationship(
            from_label="Chunk",
            from_properties={"id": chunk_id},
            to_label="Entity",
            to_properties={"name": entity["name"]},
            relationship_type="MENTIONS",
            relationship_properties={"confidence": entity.get("confidence", 1.0)}
        )

# Usage
await link_entities(
    service,
    chunk_id="doc_123_chunk_0",
    entities=[
        {"name": "RAG", "type": "CONCEPT", "confidence": 0.95},
        {"name": "OpenAI", "type": "ORG", "confidence": 0.98}
    ]
)
```

### Pattern 3: Graph-Enhanced Retrieval

Combine vector search with graph traversal:

```python
async def graph_enhanced_retrieval(vector_results, service):
    """Enrich vector search results with graph context."""

    enriched_results = []

    for result in vector_results:
        chunk_id = result["id"]

        # Get document and neighboring chunks
        context = await service.query(
            """
            MATCH (c:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(d:Document)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(neighbor:Chunk)
            WHERE neighbor.position >= c.position - 1
              AND neighbor.position <= c.position + 1
            RETURN d.title as doc_title,
                   d.source as doc_source,
                   collect(DISTINCT neighbor.text) as context_chunks
            """,
            parameters={"chunk_id": chunk_id}
        )

        enriched_results.append({
            **result,
            "document": context[0]["doc_title"],
            "source": context[0]["doc_source"],
            "context": context[0]["context_chunks"]
        })

    return enriched_results
```

### Pattern 4: Constraint and Index Setup

Set up constraints and indexes for better performance:

```python
async def setup_graph_schema(service):
    """Create constraints and indexes for optimal performance."""

    # Unique constraints (also create indexes)
    await service.create_constraint(
        constraint_name="unique_document_id",
        node_label="Document",
        property_name="id",
        constraint_type="UNIQUE"
    )

    await service.create_constraint(
        constraint_name="unique_chunk_id",
        node_label="Chunk",
        property_name="id",
        constraint_type="UNIQUE"
    )

    await service.create_constraint(
        constraint_name="unique_entity_name",
        node_label="Entity",
        property_name="name",
        constraint_type="UNIQUE"
    )

    # Additional indexes for common queries
    await service.create_index(
        index_name="document_title_idx",
        node_label="Document",
        property_name="title",
        index_type="TEXT"
    )

    await service.create_index(
        index_name="chunk_text_idx",
        node_label="Chunk",
        property_name="text",
        index_type="TEXT"
    )

# Run once at setup
await setup_graph_schema(service)
```

## Database Management

### Get Statistics

```python
stats = await service.get_stats()
print(f"Total nodes: {stats['node_count']}")
print(f"Total relationships: {stats['relationship_count']}")
print(f"Documents: {stats.get('nodes_Document', 0)}")
print(f"Chunks: {stats.get('nodes_Chunk', 0)}")
```

### Clear Database (Development Only)

```python
# WARNING: This deletes ALL data!
await service.clear(confirm=True)
```

### Cleanup

```python
# Close connection when done
await service.close()

# Or use as context manager
async with create_neo4j_service() as service:
    # Do work...
    pass
# Automatically closes
```

## Neo4j Browser

Neo4j Browser is a powerful tool for visualizing and exploring your graph:

1. **Open**: http://localhost:7474
2. **Connect**: bolt://localhost:7687 with neo4j/password
3. **Query**: Try these commands:

```cypher
// View all nodes
MATCH (n) RETURN n LIMIT 25

// View document structure
MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)
RETURN d, r, c

// Find entities mentioned in multiple chunks
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
WHERE c <> c2
RETURN e.name, count(DISTINCT c) as mention_count
ORDER BY mention_count DESC
```

## Best Practices

1. **Use merge for updates**: Use `merge=True` when creating nodes that might already exist
2. **Set up constraints early**: Define unique constraints before bulk loading
3. **Batch operations**: For large datasets, batch your create operations
4. **Index frequently queried properties**: Add TEXT indexes for properties you search on
5. **Close connections**: Always close the service when done or use context managers
6. **Parameterize queries**: Use parameters instead of string interpolation to prevent injection

## Next Steps

- See [API Reference](../api/core/graphstore.md) for complete API documentation
- Check out [examples/graph_rag_basic.py](../../examples/graph_rag_basic.py) for a working example
- Learn about [combining vector and graph search](./hybrid_search.md) (future)
- Explore [advanced graph patterns](./advanced_graph_patterns.md) (future)

## Troubleshooting

### Cannot connect to Neo4j

```python
# Check if Neo4j is running
docker ps | grep neo4j

# Check connectivity
connected = await service.verify_connectivity()
if not connected:
    print("Neo4j is not reachable")
```

### Authentication failed

Make sure your credentials match:

```bash
# In docker run command
-e NEO4J_AUTH=neo4j/your_password

# In Python code or env vars
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Constraint violations

If you get "already exists" errors, you may have duplicate data. Either:

1. Use `merge=True` to update instead of create
2. Clear the database: `await service.clear(confirm=True)`
3. Use different IDs for your nodes

## Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Graph Data Modeling](https://neo4j.com/developer/guide-data-modeling/)
- [Neo4j Aura (Cloud)](https://neo4j.com/cloud/platform/aura-graph-database/)
