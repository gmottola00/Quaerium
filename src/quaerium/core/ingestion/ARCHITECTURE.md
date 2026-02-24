# 📚 Ingestion Architecture - Technical Documentation

See full documentation in `src/core/ingestion/ARCHITECTURE.md`

## Quick Start

```python
from src.infra.parsers import create_ingestion_service

service = create_ingestion_service()
result = service.parse_document("document.pdf")
```

## Architecture

- **Core** (`src/core/ingestion/`) — Abstract protocols and orchestration
- **Infra** (`src/infra/parsers/`) — Concrete implementations

## Files

- `base.py` — Protocol definitions (interfaces)
- `service.py` — Generic orchestrator with dependency injection
- `../infra/parsers/factory.py` — Production factory

## See Also

- `MIGRATION_INGESTION.md` — Migration guide from old code
- `examples/ingestion_usage.py` — Usage examples
- `architecture.md` — Overall project architecture
