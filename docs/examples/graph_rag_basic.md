# Graph RAG Basic Example

This example demonstrates how to use quaerium's Neo4j integration to build a Graph RAG system that combines vector search with knowledge graph capabilities.

## Overview

The example shows:

1. **Setting up Neo4j** with constraints and indexes
2. **Indexing documents** with chunks in a graph structure
3. **Extracting and linking entities** from text
4. **Querying the graph** to explore relationships
5. **Graph traversal** for enhanced retrieval

## Source Code

The complete example is available at: [`examples/graph_rag_basic.py`](https://github.com/gmottola00/quaerium/blob/main/examples/graph_rag_basic.py)

## Prerequisites

### 1. Install Dependencies

```bash
pip install quaerium[neo4j]
```

Or with uv:

```bash
uv pip install quaerium[neo4j]
```

### 2. Start Neo4j

Using Docker:

```bash
docker run -d \
  --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

Or use the Makefile:

```bash
make docker-up-neo4j
```

### 3. Set Environment Variables (Optional)

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

## Running the Example

### Quick Run

```bash
python examples/graph_rag_basic.py
```

Or use the Makefile:

```bash
make run-graph-example
```

### Expected Output

```
🚀 Graph RAG Basic Example with Neo4j

============================================================

🔌 Connecting to Neo4j...
✅ Connected to Neo4j

🧹 Clearing database for fresh start...

📋 Setting up graph schema...
✅ Schema created (constraints and indexes)

📄 Indexing document: Introduction to RAG Systems
  ✅ Created document node: doc_rag_intro
  ✅ Created chunk 0: Retrieval-Augmented Generation (RAG) combines...
  ✅ Created chunk 1: Vector embeddings are used to represent text...
  ✅ Created chunk 2: Graph databases like Neo4j store relationships...

🔗 Linking entities for doc_rag_intro_chunk_0...
  ✅ Linked entity: RAG (CONCEPT)
  ✅ Linked entity: Language Models (CONCEPT)

🔗 Linking entities for doc_rag_intro_chunk_1...
  ✅ Linked entity: OpenAI (ORGANIZATION)
  ✅ Linked entity: Sentence-Transformers (LIBRARY)

🔗 Linking entities for doc_rag_intro_chunk_2...
  ✅ Linked entity: Neo4j (DATABASE)
  ✅ Linked entity: Graph Databases (CONCEPT)

🔍 Querying document structure for doc_rag_intro...

📊 Found 3 chunks:
  • Chunk 0: Retrieval-Augmented Generation (RAG) combines...
  • Chunk 1: Vector embeddings are used to represent text...
  • Chunk 2: Graph databases like Neo4j store relationships...

🔍 Querying entities...

📊 Found 6 entities:
  • RAG (CONCEPT): mentioned 1 times
  • Language Models (CONCEPT): mentioned 1 times
  • OpenAI (ORGANIZATION): mentioned 1 times
  • Sentence-Transformers (LIBRARY): mentioned 1 times
  • Neo4j (DATABASE): mentioned 1 times
  • Graph Databases (CONCEPT): mentioned 1 times

🔍 Finding documents mentioning 'Neo4j'...

📊 Found 1 documents:
  • Introduction to RAG Systems (ID: doc_rag_intro)
    Mentioned in 1 chunks

📊 Database Statistics:
  • Total nodes: 10
  • Total relationships: 9
  • Documents: 1
  • Chunks: 3
  • Entities: 6

============================================================

🌐 Explore your graph in Neo4j Browser:
  URL: http://localhost:7474
  Connect: bolt://localhost:7687
  Credentials: neo4j / password

  Try these queries:
    MATCH (n) RETURN n LIMIT 25
    MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)
    RETURN d, r, c
    MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
    RETURN e, c

✅ Example completed successfully!

🧹 Closing connection...
✅ Done!
```

## Code Walkthrough

### 1. Connect to Neo4j

```python
from quaerium.infra.graphstores import create_neo4j_service

service = create_neo4j_service(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Verify connectivity
connected = await service.verify_connectivity()
if not connected:
    print("❌ Cannot connect to Neo4j")
    return
```

### 2. Set Up Schema

Create constraints for data integrity and indexes for performance:

```python
async def setup_schema(service):
    # Unique constraints (also create indexes)
    await service.create_constraint(
        constraint_name="unique_document_id",
        node_label="Document",
        property_name="id",
        constraint_type="UNIQUE"
    )

    # Additional indexes for text search
    await service.create_index(
        index_name="document_title_idx",
        node_label="Document",
        property_name="title",
        index_type="TEXT"
    )
```

### 3. Index Document with Chunks

```python
async def index_document(service, doc_id, title, chunks):
    # Create document node
    await service.create_node(
        label="Document",
        properties={
            "id": doc_id,
            "title": title,
            "source": "example.pdf"
        },
        merge=True
    )

    # Create chunks and link to document
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"

        # Create chunk node
        await service.create_node(
            label="Chunk",
            properties={"id": chunk_id, "text": chunk_text, "position": i},
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
```

### 4. Extract and Link Entities

```python
async def extract_and_link_entities(service, chunk_id, entities):
    for entity in entities:
        # Create or merge entity node
        await service.create_node(
            label="Entity",
            properties={
                "name": entity["name"],
                "type": entity["type"]
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
```

### 5. Query the Graph

```python
# Find all chunks of a document (ordered)
results = await service.query(
    """
    MATCH (d:Document {id: $doc_id})-[r:HAS_CHUNK]->(c:Chunk)
    RETURN d.title as title, c.id as chunk_id, c.text as text, r.position as position
    ORDER BY r.position
    """,
    parameters={"doc_id": doc_id}
)

# Find documents mentioning an entity
results = await service.query(
    """
    MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
    RETURN DISTINCT d.id as doc_id, d.title as title, collect(DISTINCT c.id) as chunks
    """,
    parameters={"entity_name": entity_name}
)
```

### 6. Get Statistics

```python
stats = await service.get_stats()
print(f"Total nodes: {stats['node_count']}")
print(f"Total relationships: {stats['relationship_count']}")
print(f"Documents: {stats.get('nodes_Document', 0)}")
print(f"Chunks: {stats.get('nodes_Chunk', 0)}")
print(f"Entities: {stats.get('nodes_Entity', 0)}")
```

## Visualizing the Graph

After running the example, you can explore the graph visually:

1. **Open Neo4j Browser**: http://localhost:7474
2. **Connect**: bolt://localhost:7687 with neo4j/password
3. **Run queries**:

```cypher
// View all nodes and relationships
MATCH (n) RETURN n LIMIT 25

// View document structure
MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)
RETURN d, r, c

// View entity mentions
MATCH (e:Entity)<-[r:MENTIONS]-(c:Chunk)
RETURN e, r, c

// Find entities mentioned in multiple chunks
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
WHERE c <> c2
RETURN e.name, count(DISTINCT c) as mention_count
ORDER BY mention_count DESC
```

## Graph Structure

The example creates this graph structure:

```
Document (doc_rag_intro)
  ├─[HAS_CHUNK]→ Chunk (chunk_0)
  │   ├─[MENTIONS]→ Entity (RAG)
  │   └─[MENTIONS]→ Entity (Language Models)
  ├─[HAS_CHUNK]→ Chunk (chunk_1)
  │   ├─[MENTIONS]→ Entity (OpenAI)
  │   └─[MENTIONS]→ Entity (Sentence-Transformers)
  └─[HAS_CHUNK]→ Chunk (chunk_2)
      ├─[MENTIONS]→ Entity (Neo4j)
      └─[MENTIONS]→ Entity (Graph Databases)
```

## Next Steps

1. **Add vector embeddings**: Store embeddings in chunks for hybrid search
2. **Entity extraction**: Use NER models to automatically extract entities
3. **Complex relationships**: Add more relationship types (CITES, RELATED_TO, etc.)
4. **Graph algorithms**: Use PageRank, community detection for ranking
5. **Combine with vector search**: Hybrid retrieval using both graph and vectors

## Related Documentation

- [Graph RAG Guide](../guides/graph_rag.md) - Complete guide to Graph RAG
- [GraphStore API Reference](../api/core/graphstore.md) - API documentation
- [Neo4j Documentation](https://neo4j.com/docs/) - Official Neo4j docs

## Troubleshooting

### Cannot connect to Neo4j

Make sure Neo4j is running:

```bash
docker ps | grep neo4j
```

If not running:

```bash
make docker-up-neo4j
```

### Authentication failed

Check credentials match:

```bash
# Docker
docker run ... -e NEO4J_AUTH=neo4j/password ...

# Environment variables
export NEO4J_PASSWORD=password
```

### Database not empty

Clear the database:

```bash
# In Python
await service.clear(confirm=True)

# Or in Neo4j Browser
MATCH (n) DETACH DELETE n
```
