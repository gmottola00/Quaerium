# Migration Tools

Vector store migration engine with validation, retry logic, and progress tracking.

## VectorStoreMigrator

::: rag_toolkit.migration.migrator.VectorStoreMigrator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - migrate
        - estimate
      heading_level: 3

## Models

::: rag_toolkit.migration.models.MigrationResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: rag_toolkit.migration.models.MigrationEstimate
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: rag_toolkit.migration.models.MigrationProgress
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Exceptions

::: rag_toolkit.migration.exceptions.MigrationError
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: rag_toolkit.migration.exceptions.ValidationError
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Migration

```python
from rag_toolkit.migration import VectorStoreMigrator
from rag_toolkit.infra.vectorstores import get_qdrant_service, get_chromadb_service

# Initialize stores
source = get_qdrant_service(host="localhost", port=6333)
target = get_chromadb_service(host="localhost", port=8000)

# Create migrator
migrator = VectorStoreMigrator(
    source=source,
    target=target,
    validate=True
)

# Run migration
result = migrator.migrate(
    source_collection="docs",
    target_collection="docs_backup"
)

print(f"Migrated: {result.vectors_migrated}")
print(f"Success rate: {result.success_rate}%")
```

### Migration with Filtering

```python
# Migrate only published documents from 2025
result = migrator.migrate(
    source_collection="documents",
    target_collection="documents_2025",
    filter={
        "status": "published",
        "year": 2025
    }
)
```

### Dry-Run Mode

```python
# Test migration without writing
result = migrator.migrate(
    source_collection="large_dataset",
    target_collection="large_dataset_backup",
    dry_run=True  # No writes to target
)

print(f"Would migrate {result.vectors_migrated} vectors")
print(f"Estimated duration: {result.duration_seconds:.1f}s")

# If looks good, run for real
if result.vectors_migrated < 1_000_000:
    result = migrator.migrate(
        source_collection="large_dataset",
        target_collection="large_dataset_backup",
        dry_run=False
    )
```

### Migration with Retry Logic

```python
# Configure retry behavior
migrator = VectorStoreMigrator(
    source=source,
    target=target,
    max_retries=5,
    retry_delay=2.0,
    retry_backoff=1.5
)

# Migration will automatically retry on failures
result = migrator.migrate(source_collection="docs")
```

### Progress Tracking

```python
def on_progress(progress):
    print(
        f"Progress: {progress.percentage:.1f}% "
        f"({progress.vectors_processed}/{progress.total_vectors}) "
        f"ETA: {progress.eta_seconds:.0f}s"
    )

migrator = VectorStoreMigrator(
    source=source,
    target=target,
    on_progress=on_progress  # Callback for progress
)

result = migrator.migrate(source_collection="large_collection")
```

### Estimation

```python
# Estimate before migrating
estimate = migrator.estimate(
    collection_name="huge_collection",
    batch_size=1000
)

print(f"Total vectors: {estimate.total_vectors:,}")
print(f"Estimated duration: {estimate.estimated_duration_seconds:.0f}s")
print(f"Estimated batches: {estimate.estimated_batches}")
print(f"Throughput: {estimate.vectors_per_second:.1f} vectors/sec")

# Decide based on estimate
if estimate.estimated_duration_seconds > 3600:  # > 1 hour
    print("Migration will take long, consider splitting")
else:
    result = migrator.migrate(collection_name="huge_collection")
```

## See Also

- [User Guide: Migration](../../tools/migration.md) - Comprehensive guide
- [Examples: Production Setup](../../examples/production_setup.md) - Production patterns
