# Milvus Docker Setup

Production-ready Milvus vector database setup for RAG Toolkit.

## Quick Start

```bash
# Start Milvus
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f standalone

# Stop
docker-compose down

# Stop and remove data
docker-compose down -v
```

## Services

- **Milvus Standalone** - Vector database (ports 19530, 9091)
- **etcd** - Metadata storage
- **MinIO** - Object storage

## Endpoints

- **Milvus API**: `http://localhost:19530`
- **Milvus Web UI**: `http://localhost:9091`
- **MinIO Console**: `http://localhost:9001` (admin/minioadmin)

## Configuration

Set environment variables in `.env` file:

```env
DOCKER_VOLUME_DIRECTORY=./data
```

## Usage in Code

```python
from rag_toolkit.infra.vectorstores.factory import create_milvus_service

# Create service
service = create_milvus_service(
    uri="http://localhost:19530"
)

# Create collection
service.ensure_collection("my_collection", dimension=768)
```

## Health Check

```bash
# Check if Milvus is ready
curl http://localhost:9091/healthz

# Expected: OK
```

## Data Persistence

Data is stored in `./volumes/` directory:
- `./volumes/milvus/` - Vector data
- `./volumes/etcd/` - Metadata
- `./volumes/minio/` - Object storage

## Troubleshooting

### Port conflicts

Change ports in `docker-compose.yml`:

```yaml
ports:
  - "19531:19530"  # Change first port
```

### Increase memory

Milvus requires at least 4GB RAM. Adjust Docker Desktop settings.

### Reset everything

```bash
docker-compose down -v
rm -rf volumes/
docker-compose up -d
```

## Production Notes

For production deployments:
1. Use external etcd cluster
2. Use external S3-compatible storage
3. Enable authentication
4. Configure resource limits

See [Milvus documentation](https://milvus.io/docs) for details.
