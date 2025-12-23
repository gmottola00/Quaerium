# Qdrant Docker Setup

Lightweight and fast Qdrant vector database setup for RAG Toolkit.

## Quick Start

```bash
# Start Qdrant
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f qdrant

# Stop
docker-compose down

# Stop and remove data
docker-compose down -v
```

## Services

- **Qdrant** - Vector database with HTTP and gRPC APIs

## Endpoints

- **HTTP API**: `http://localhost:6333`
- **gRPC API**: `http://localhost:6334`
- **Web Dashboard**: `http://localhost:6333/dashboard`

## Configuration

Set environment variables in `.env` file:

```env
DOCKER_VOLUME_DIRECTORY=./data
# Optional: Enable API authentication
# QDRANT_API_KEY=your-secret-key
```

### Enable Authentication

Uncomment in `docker-compose.yml`:

```yaml
environment:
  - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
```

Then use in code:

```python
from rag_toolkit.infra.vectorstores.factory import create_qdrant_service

service = create_qdrant_service(
    url="http://localhost:6333",
    api_key="your-secret-key"
)
```

## Usage in Code

```python
from rag_toolkit.infra.vectorstores.qdrant import QdrantService
from rag_toolkit.infra.vectorstores.qdrant.config import QdrantConfig

# Create service
config = QdrantConfig(url="http://localhost:6333")
service = QdrantService(config)

# Create collection
service.ensure_collection(
    collection_name="my_collection",
    vector_size=768
)

# Health check
print(service.health_check())  # True
```

## Health Check

```bash
# Check if Qdrant is ready
curl http://localhost:6333/healthz

# Get collections
curl http://localhost:6333/collections
```

## Data Persistence

Data is stored in `./volumes/qdrant/` directory.

## Features

- ✅ Fast startup (< 10 seconds)
- ✅ Low memory footprint
- ✅ Built-in web dashboard
- ✅ HNSW and quantization support
- ✅ Advanced payload filtering
- ✅ gRPC for high performance

## Troubleshooting

### Port conflicts

Change ports in `docker-compose.yml`:

```yaml
ports:
  - "6335:6333"  # Change HTTP port
  - "6336:6334"  # Change gRPC port
```

### Reset data

```bash
docker-compose down -v
rm -rf volumes/
docker-compose up -d
```

### Enable gRPC

For better performance, use gRPC in your code:

```python
config = QdrantConfig(
    url="http://localhost:6333",
    grpc_port=6334,
    prefer_grpc=True
)
```

## Production Notes

For production deployments:
1. Enable authentication
2. Use HTTPS
3. Configure snapshots
4. Set up backup strategy
5. Monitor with Prometheus

See [Qdrant documentation](https://qdrant.tech/documentation/) for details.
