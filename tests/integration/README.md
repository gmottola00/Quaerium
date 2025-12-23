"""Integration tests README."""

# Integration Tests

Integration tests that run against real Docker services.

## Prerequisites

1. Docker and Docker Compose installed
2. At least 8GB RAM available for Docker
3. Ports 6333, 11434, 19530 available

## Running Tests

### Quick Start

```bash
# Start services and run tests
make test-integration

# Or manually
make docker-up
pytest tests/integration -v -m integration
make docker-down
```

### Individual Services

```bash
# Test Qdrant only
make docker-up-qdrant
pytest tests/integration/test_qdrant_integration.py -v
make docker-down-qdrant
```

## Test Markers

Integration tests are marked with `@pytest.mark.integration`:

```bash
# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"

# Run all tests including integration
pytest
```

## Writing Integration Tests

```python
import pytest

@pytest.mark.integration
def test_my_integration(qdrant_service, ollama_client, test_collection_name):
    """Test description."""
    # Your test code
    pass
```

### Available Fixtures

- `docker_services` - Ensures all services are running
- `ollama_client` - OllamaEmbeddingClient instance
- `qdrant_service` - QdrantService instance
- `milvus_service` - MilvusService instance
- `test_collection_name` - Unique collection name for each test

## Troubleshooting

### Services won't start

```bash
# Check Docker
docker ps

# View logs
docker-compose -f docker/all/docker-compose.yml logs

# Reset
make docker-clean
make docker-up
```

### Tests fail with connection errors

```bash
# Check service health
make docker-health

# Wait longer for services
sleep 30
pytest tests/integration -v
```

### Clean up after failed tests

```bash
# Stop all services
make docker-down

# Remove all data
make docker-clean
```

## CI/CD Integration

For GitHub Actions:

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: make docker-up
      
      - name: Wait for services
        run: sleep 30
      
      - name: Run integration tests
        run: pytest tests/integration -v -m integration
      
      - name: Cleanup
        if: always()
        run: make docker-down
```

## Performance Notes

- Integration tests are slower (~30s setup + test time)
- Run unit tests first for quick feedback
- Use `pytest-xdist` for parallel execution:
  ```bash
  pytest tests/integration -n auto
  ```

## Best Practices

1. **Always cleanup**: Use fixtures to ensure collection cleanup
2. **Unique names**: Use `test_collection_name` fixture
3. **Idempotent**: Tests should work regardless of execution order
4. **Isolated**: Don't rely on state from other tests
5. **Fast enough**: Keep integration tests under 5 minutes total
