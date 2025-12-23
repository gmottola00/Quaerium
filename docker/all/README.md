# All Services Docker Setup

Complete development environment with all RAG Toolkit services.

## Quick Start

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready (takes ~2 minutes)
docker-compose ps

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Services Included

| Service | Port(s) | Purpose | Dashboard |
|---------|---------|---------|-----------|
| **Ollama** | 11434 | Embeddings & LLM | - |
| **Qdrant** | 6333, 6334 | Vector DB | http://localhost:6333/dashboard |
| **Milvus** | 19530, 9091 | Vector DB | http://localhost:9091 |

## Initial Setup

After starting services, pull Ollama models:

```bash
# Pull embedding model
docker exec -it ollama ollama pull nomic-embed-text

# Pull LLM (optional, for chat)
docker exec -it ollama ollama pull llama3.2

# Verify models
docker exec -it ollama ollama list
```

## Health Checks

```bash
# Check all services
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:6333/healthz    # Qdrant
curl http://localhost:9091/healthz    # Milvus
```

## Usage Example

```python
from rag_toolkit.infra.embedding.ollama import OllamaEmbeddingClient
from rag_toolkit.infra.vectorstores.qdrant import QdrantService
from rag_toolkit.infra.vectorstores.qdrant.config import QdrantConfig

# Initialize embedding client
embedder = OllamaEmbeddingClient(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Initialize Qdrant
config = QdrantConfig(url="http://localhost:6333")
qdrant = QdrantService(config)

# Create collection
qdrant.ensure_collection(
    collection_name="my_docs",
    vector_size=768  # nomic-embed-text dimension
)

# Insert documents
texts = ["AI is amazing", "RAG systems are powerful"]
embeddings = [embedder.embed(text) for text in texts]

points = [
    {
        "id": f"doc-{i}",
        "vector": emb,
        "payload": {"text": text}
    }
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]

qdrant.upsert(collection_name="my_docs", points=points)

# Search
query_emb = embedder.embed("What is AI?")
results = qdrant.search(
    collection_name="my_docs",
    query_vector=query_emb,
    limit=5
)

for result in results:
    print(f"Score: {result['score']:.4f} - {result['payload']['text']}")
```

## Resource Requirements

Minimum system requirements:
- **RAM**: 8GB (16GB recommended)
- **Disk**: 10GB free space
- **CPU**: 4 cores (8 recommended for Ollama)

## Data Persistence

All data is stored in Docker volumes:
- `ollama` - Downloaded models (~1-5GB per model)
- `qdrant` - Vector data
- `milvus` - Vector data

## Performance Tips

1. **Use Qdrant for development** - Faster startup and lower memory
2. **Use Milvus for production** - Better for large-scale deployments
3. **Pull smaller Ollama models** - Use `nomic-embed-text` (274MB) instead of larger models

## Troubleshooting

### Services won't start

Check Docker resources:
```bash
docker system df
docker system prune  # Clean up if needed
```

### Ollama model pull fails

```bash
# Check Ollama logs
docker logs ollama

# Restart Ollama
docker restart ollama

# Try again
docker exec -it ollama ollama pull nomic-embed-text
```

### Port conflicts

Edit `docker-compose.yml` and change the first port number:

```yaml
ports:
  - "11435:11434"  # Change 11435 to any free port
```

### Reset everything

```bash
# Nuclear option: remove all data
docker-compose down -v
rm -rf volumes/
docker-compose up -d

# Re-pull Ollama models
docker exec -it ollama ollama pull nomic-embed-text
```

## Service-Specific Commands

### Ollama

```bash
# List models
docker exec -it ollama ollama list

# Remove a model
docker exec -it ollama ollama rm llama3.2

# Run interactive chat
docker exec -it ollama ollama run llama3.2
```

### Qdrant

```bash
# Open dashboard
open http://localhost:6333/dashboard

# List collections
curl http://localhost:6333/collections
```

### Milvus

```bash
# Check status
curl http://localhost:9091/healthz

# View metrics
curl http://localhost:9091/metrics
```

## Integration Tests

Use this setup for integration tests:

```bash
# Start services
docker-compose up -d

# Run integration tests
pytest tests/integration -v -m integration

# Cleanup
docker-compose down -v
```

## Production Migration

When moving to production:

1. **Ollama** → Use dedicated GPU server or cloud API
2. **Qdrant** → Use Qdrant Cloud or deploy cluster
3. **Milvus** → Deploy distributed Milvus cluster

See individual service READMEs for production configuration.

## Support

- Ollama: https://ollama.ai/
- Qdrant: https://qdrant.tech/
- Milvus: https://milvus.io/
