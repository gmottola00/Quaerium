# Graph Store API Reference

The graph store module provides protocol-based abstractions for graph database operations, enabling seamless integration with knowledge graph systems for Graph RAG applications.

## Core Protocol

::: quaerium.core.graphstore.GraphStoreClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members:
        - create_node
        - create_relationship
        - query
        - create_constraint
        - create_index
        - get_stats
        - clear

## Type Definitions

::: quaerium.core.types.GraphNode
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: quaerium.core.types.GraphRelationship
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: quaerium.core.types.GraphMetadata
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Neo4j Implementation

### Factory Function

::: quaerium.infra.graphstores.factory.create_neo4j_service
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Service

::: quaerium.infra.graphstores.neo4j.service.Neo4jService
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members:
        - __init__
        - verify_connectivity
        - create_node
        - create_relationship
        - query
        - create_constraint
        - create_index
        - get_stats
        - clear
        - close

### Configuration

::: quaerium.infra.graphstores.neo4j.config.Neo4jConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Client

::: quaerium.infra.graphstores.neo4j.client.Neo4jClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members:
        - __init__
        - verify_connectivity
        - session
        - execute_query
        - execute_write
        - create_constraint
        - create_index
        - clear_database
        - get_database_stats
        - close

### Utilities

::: quaerium.infra.graphstores.neo4j.utils.convert_neo4j_types
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Exceptions

::: quaerium.infra.graphstores.neo4j.exceptions
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members:
        - Neo4jError
        - ConnectionError
        - QueryError
        - NodeNotFoundError
        - ConstraintError
        - IndexError

## Usage Examples

### Basic Usage

```python
from quaerium.infra.graphstores import create_neo4j_service

# Create service
service = create_neo4j_service(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Create node
node = await service.create_node(
    label="Document",
    properties={"id": "doc_1", "title": "Example"}
)

# Create relationship
rel = await service.create_relationship(
    from_label="Document",
    from_properties={"id": "doc_1"},
    to_label="Chunk",
    to_properties={"id": "chunk_1"},
    relationship_type="HAS_CHUNK"
)

# Query
results = await service.query(
    "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d, c"
)

# Cleanup
await service.close()
```

### Using Environment Variables

```python
import os

# Set environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Create service (reads from env vars)
service = create_neo4j_service()
```

### Context Manager

```python
from quaerium.infra.graphstores import create_neo4j_service

async with create_neo4j_service() as service:
    # Service automatically connects
    await service.create_node(
        label="Document",
        properties={"id": "doc_1"}
    )
    # Service automatically closes
```

### Protocol Compliance

```python
from quaerium.core.graphstore import GraphStoreClient
from quaerium.infra.graphstores import Neo4jService, Neo4jConfig

# Neo4jService implements GraphStoreClient protocol
config = Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="password")
service = Neo4jService(config)

# Can use as GraphStoreClient
assert isinstance(service, GraphStoreClient)
```

## See Also

- [Graph RAG Guide](../../guides/graph_rag.md) - Complete guide to using Graph RAG
- [Vector Store API](./vectorstore.md) - Vector database operations
- [RAG Pipeline API](./pipeline.md) - End-to-end RAG pipelines
