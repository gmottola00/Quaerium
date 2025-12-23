# Docker Infrastructure

Production-ready Docker setup for RAG Toolkit development and testing.

## Structure

```
docker/
├── docker.sh           # Helper script for all operations
├── milvus/            # Milvus setup (production-ready)
│   ├── docker-compose.yml
│   └── README.md
├── qdrant/            # Qdrant setup (lightweight)
│   ├── docker-compose.yml
│   └── README.md
└── all/               # All services together
    ├── docker-compose.yml
    └── README.md
```

## Quick Start

```bash
# Start all services
./docker/docker.sh up

# Or use Makefile
make docker-up

# Pull Ollama models
make docker-pull-models

# Check health
make docker-health

# View logs
make docker-logs

# Stop all
make docker-down
```

## Services

| Service | Ports | Purpose | Memory |
|---------|-------|---------|--------|
| Ollama | 11434 | Embeddings & LLM | ~2GB |
| Qdrant | 6333, 6334 | Vector DB (fast) | ~500MB |
| Milvus | 19530, 9091 | Vector DB (scalable) | ~2GB |

## Usage Examples

### Qdrant

```python
from rag_toolkit.infra.vectorstores.qdrant import QdrantService
from rag_toolkit.infra.vectorstores.qdrant.config import QdrantConfig

config = QdrantConfig(url="http://localhost:6333")
qdrant = QdrantService(config)
```

### Milvus

```python
from rag_toolkit.infra.vectorstores.factory import create_milvus_service

milvus = create_milvus_service(uri="http://localhost:19530")
```

### Ollama

```python
from rag_toolkit.infra.embedding.ollama import OllamaEmbeddingClient

embedder = OllamaEmbeddingClient(base_url="http://localhost:11434")
```

## Helper Script

The `docker.sh` script provides all Docker operations:

```bash
./docker/docker.sh <command> [service]

Commands:
  up [service]        Start services
  down [service]      Stop services
  restart [service]   Restart services
  logs [service]      View logs
  ps                  Show running services
  health              Check service health
  clean [service]     Remove all data (dangerous!)
  pull-models         Pull Ollama models

Services:
  all                 All services (default)
  milvus              Milvus only
  qdrant              Qdrant only

Examples:
  ./docker/docker.sh up              # Start everything
  ./docker/docker.sh up qdrant       # Start only Qdrant
  ./docker/docker.sh logs milvus     # View Milvus logs
  ./docker/docker.sh health          # Check all services
```

## Makefile Integration

All operations are available via Makefile:

```bash
# Docker commands
make docker-up                # Start all
make docker-up-qdrant        # Start Qdrant only
make docker-up-milvus        # Start Milvus only
make docker-down             # Stop all
make docker-restart          # Restart all
make docker-logs             # View logs
make docker-ps               # Show status
make docker-health           # Health check
make docker-clean            # Remove data
make docker-pull-models      # Pull Ollama models

# Development workflows
make dev-setup               # Complete setup
make dev-teardown           # Stop everything
make test-integration       # Run integration tests
```

## Development Setup

Complete development environment in one command:

```bash
make dev-setup
```

This will:
1. Install package with all dependencies
2. Start all Docker services
3. Pull required Ollama models
4. Verify everything is working

## Integration Tests

Run tests against real services:

```bash
# Start services and run tests
make test-integration

# Or manually
make docker-up
pytest tests/integration -v -m integration
make docker-down
```

See [tests/integration/README.md](../tests/integration/README.md) for details.

## Data Persistence

Each service stores data in local volumes:

```
docker/
├── milvus/volumes/     # Milvus data
├── qdrant/volumes/     # Qdrant data
└── all/volumes/        # All services data
```

### Backup Data

```bash
# Backup Qdrant
tar -czf qdrant-backup.tar.gz docker/qdrant/volumes/

# Restore
tar -xzf qdrant-backup.tar.gz
```

### Clean Data

```bash
# Remove all data (careful!)
make docker-clean

# Or specific service
./docker/docker.sh clean qdrant
```

## Troubleshooting

### Services won't start

```bash
# Check Docker
docker info

# Check ports
lsof -i :6333
lsof -i :11434
lsof -i :19530

# Check logs
make docker-logs
```

### Port conflicts

Edit `docker-compose.yml` and change ports:

```yaml
ports:
  - "6335:6333"  # Change first number
```

### Out of memory

Increase Docker Desktop memory:
- Docker Desktop → Settings → Resources → Memory → 8GB+

### Reset everything

```bash
# Nuclear option
make docker-clean
docker system prune -a --volumes
make docker-up
```

## Production Deployment

These Docker setups are optimized for **development**. For production:

### Qdrant
- Use [Qdrant Cloud](https://cloud.qdrant.io/)
- Or deploy cluster with Kubernetes
- Enable authentication
- Use HTTPS

### Milvus
- Use distributed deployment
- External etcd cluster
- S3 object storage
- Load balancer

### Ollama
- Dedicated GPU server
- Or use cloud APIs (OpenAI, Anthropic)
- Rate limiting
- Caching

See individual README files for production notes.

## Contributing

When adding new services:

1. Create `docker/<service>/docker-compose.yml`
2. Add `docker/<service>/README.md`
3. Update `docker.sh` script
4. Add Makefile targets
5. Create integration tests

## Support

- [Docker Documentation](https://docs.docker.com/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Milvus Docs](https://milvus.io/docs)
- [Ollama Docs](https://ollama.ai/)
