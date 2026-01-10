# :material-swap-horizontal: Vector Store Migration

Transfer vector data seamlessly between different vector store implementations with validation, progress tracking, and intelligent error handling.

---

## :material-information: Overview

!!! abstract "Migration Capabilities"
    Professional-grade tools for moving data between vector stores.

**What You Can Do:**

<div class="grid cards" markdown>

- :material-database-sync: **Migrate Data**

    ---

    Transfer between Milvus, Qdrant, and ChromaDB

- :material-check-circle: **Validate Integrity**

    ---

    Automatic data integrity verification

- :material-progress-check: **Track Progress**

    ---

    Real-time progress with ETA calculation

- :material-alert-decagram: **Handle Errors**

    ---

    Retry logic with exponential backoff

- :material-clock-outline: **Estimate Resources**

    ---

    Predict migration time and requirements

</div>

---

## :material-rocket-launch: Quick Start

### :material-numeric-1-box: Basic Migration

```python title="simple_migration.py" linenums="1" hl_lines="6-14 17-21 24"
from rag_toolkit.migration import VectorStoreMigrator
from rag_toolkit.infra.vectorstores import (
    get_chromadb_service,
    get_qdrant_service,
)

# Initialize source and target stores
source = get_chromadb_service(
    host="localhost",
    port=8000,
)

target = get_qdrant_service(
    host="localhost",
    port=6333,
)

# Create migrator
migrator = VectorStoreMigrator(
    source=source,
    target=target,
    validate=True,  # Enable validation
)

# Run migration
result = migrator.migrate(
    source_collection="my_documents",
    target_collection="my_documents",
    batch_size=1000,
)

# Check results
print(f"Success: {result.success}")
print(f"Migrated: {result.vectors_migrated}")
print(f"Failed: {result.vectors_failed}")
print(f"Duration: {result.duration_seconds}s")
print(f"Success Rate: {result.success_rate}%")
```

!!! success "Migration Complete"
    ```
    Success: True
    Migrated: 125,430 vectors
    Failed: 0
    Duration: 127.5s
    Success Rate: 100.0%
    ```

### :material-numeric-2-box: Migration with Progress Tracking

!!! example "Real-Time Progress Monitoring"
    Track migration progress with live updates and ETA.

```python title="progress_tracking.py" linenums="1" hl_lines="2-7 10-12"
# Define progress callback
def on_progress(progress):
    print(
        f"Progress: {progress.percentage:.1f}% "
        f"({progress.vectors_processed}/{progress.total_vectors}) "
        f"- ETA: {progress.eta_seconds:.0f}s"
    )

migrator = VectorStoreMigrator(
    source=source,
    target=target,
    on_progress=on_progress,  # Add callback
)

result = migrator.migrate(
    source_collection="my_docs",
    batch_size=500,
)
```

!!! quote "Progress Output"
    ```
    Progress: 12.5% (12,500/100,000) - ETA: 245s
    Progress: 25.0% (25,000/100,000) - ETA: 210s
    Progress: 37.5% (37,500/100,000) - ETA: 175s
    Progress: 50.0% (50,000/100,000) - ETA: 140s
    ...
    Progress: 100.0% (100,000/100,000) - ETA: 0s
    ```

---

## :material-star: Features

### :material-calculator: Estimation

!!! tip "Plan Before You Migrate"
    Estimate time and resources before starting a large migration.

```python title="estimate_migration.py" hl_lines="2-5"
# Get migration estimate
estimate = migrator.estimate(
    collection_name="large_collection",
    batch_size=1000,
)

print(f"Total vectors: {estimate.total_vectors:,}")
print(f"Estimated duration: {estimate.estimated_duration_seconds:.1f}s")
print(f"Estimated batches: {estimate.estimated_batches}")
print(f"Throughput: {estimate.vectors_per_second:.0f} vectors/sec")
```

!!! example "Estimate Output"
    ```
    Total vectors: 1,250,000
    Estimated duration: 625.0s (~10.4 minutes)
    Estimated batches: 1,250
    Throughput: 2,000 vectors/sec
    ```

### :material-filter: Filtered Migration

!!! info "Selective Migration"
    Migrate only vectors matching specific metadata criteria.

=== "By Status & Year"

    ```python title="filter_by_status.py" linenums="1" hl_lines="4-7"
    # Migrate only published articles from 2025
    result = migrator.migrate(
        source_collection="articles",
        target_collection="articles_published_2025",
        filter={
            "status": "published",
            "year": 2025,
        },
    )

    print(f"Migrated {result.vectors_migrated} filtered vectors")
    ```

=== "Complex Criteria"

    ```python title="complex_filter.py" linenums="1" hl_lines="4-9"
    # Multiple filter criteria
    result = migrator.migrate(
        source_collection="documents",
        target_collection="filtered_docs",
        filter={
            "document_type": "report",
            "department": "engineering",
            "confidential": False,
            "version": "2.0",
        },
    )
    ```

=== "Use Cases"

    <div class="grid cards" markdown>

    -   :material-filter-check: **Migrate Subsets**

        ---

        Migrate only production-ready documents

    -   :material-backup-restore: **Filtered Backups**

        ---

        Create backups of non-sensitive data only

    -   :material-format-list-group: **Split Collections**

        ---

        Organize by customer, region, or date

    -   :material-test-tube: **Test Samples**

        ---

        Test migrations with representative data

    </div>

---

### :material-test-tube: Dry-Run Mode

!!! warning "Test Before Executing"
    Simulate migrations without writing to target store.

```python title="dry_run.py" linenums="1" hl_lines="5"
# Test migration before executing
result = migrator.migrate(
    source_collection="production_data",
    target_collection="production_backup",
    dry_run=True,  # No writes to target
)

print(f"Would migrate {result.vectors_migrated} vectors")
print(f"Estimated duration: {result.duration_seconds}s")
print(f"No data written to target store")
```

=== "With Filters"

    ```python title="dry_run_with_filter.py" linenums="1" hl_lines="3 4"
    # Test filtered migration
    result = migrator.migrate(
        source_collection="documents",
        filter={"status": "active"},
        dry_run=True,
    )

    if result.vectors_migrated > 1000000:
        print("⚠ Warning: Large migration, consider splitting")
    else:
        print("✓ Safe to proceed")
        
        # Execute real migration
        real_result = migrator.migrate(
            source_collection="documents",
            filter={"status": "active"},
            dry_run=False,
        )
    ```

=== "Benefits"

    <div class="grid cards" markdown>

    -   :material-shield-check: **Zero Risk**

        ---

        Test without modifying target

    -   :material-check-decagram: **Validation**

        ---

        Verify filters and counts

    -   :material-speedometer: **Performance**

        ---

        Measure actual throughput

    -   :material-calculator-variant: **Planning**

        ---

        Calculate storage & windows

    </div>

!!! note "Dry-Run Behavior"
    Dry-run mode skips target writes and validation but still reads from source to provide accurate counts.

---

### :material-restart: Retry Logic

!!! success "Automatic Error Recovery"
    Built-in exponential backoff for transient failures.

```python title="retry_config.py" linenums="1" hl_lines="4-6"
# Configure retry behavior
migrator = VectorStoreMigrator(
    source=source,
    target=target,
    max_retries=5,         # Up to 5 retries (default: 3)
    retry_delay=2.0,       # Initial delay 2s (default: 1.0s)
    retry_backoff=2.0,     # Double delay each retry (default: 2.0)
)

result = migrator.migrate(
    source_collection="unreliable_network_migration",
)
```

!!! example "Automatic Retry On"
    - ⚡ Network timeouts
    - 🔌 Temporary connection failures
    - 🚦 Rate limit errors (429)
    - ⏱️ Transient store unavailability

=== "Retry Timing"

    ```python title="retry_timing.py"
    # Exponential backoff example
    Attempt 1: Immediate
    Attempt 2: Wait 2.0s
    Attempt 3: Wait 4.0s (2.0 * 2.0)
    Attempt 4: Wait 8.0s (4.0 * 2.0)
    Attempt 5: Wait 16.0s (8.0 * 2.0)
    ```

=== "Conservative (Cloud)"

    ```python title="conservative_retry.py" linenums="1" hl_lines="4-6"
    # Longer delays for rate-limited APIs
    migrator = VectorStoreMigrator(
        source=source,
        target=target,
        max_retries=10,
        retry_delay=5.0,
        retry_backoff=1.5,  # Slower backoff
    )
    ```

=== "Aggressive (Local)"

    ```python title="aggressive_retry.py" linenums="1" hl_lines="4-6"
    # Fast retry for local migrations
    migrator = VectorStoreMigrator(
        source=source,
        target=target,
        max_retries=3,
        retry_delay=0.5,
        retry_backoff=2.0,
    )
    ```

=== "Error Handling"

    ```python title="error_handling.py" linenums="1" hl_lines="1 4-7"
    from rag_toolkit.migration import MigrationError

    try:
        result = migrator.migrate(source_collection="docs")
    except MigrationError as e:
        # All retries exhausted
        print(f"Failed after {migrator.max_retries} retries: {e}")
    ```

---

### :material-check-all: Validation

!!! tip "Data Integrity"
    Automatic validation ensures successful migration.

```python title="validation.py" linenums="1" hl_lines="4"
# Validation enabled by default
migrator = VectorStoreMigrator(
    source=source,
    target=target,
    validate=True,  # Enable validation
)

result = migrator.migrate(source_collection="docs")

# Check validation status
if result.metadata.get("validated"):
    print("✓ Migration validated successfully")
else:
    print("⚠ Validation warnings:", result.errors)
```

---

### :material-alert-circle: Error Handling

!!! warning "Resilient Migration"
    Migration continues even if individual batches fail.

```python title="error_handling_batch.py" linenums="1" hl_lines="7-10"
result = migrator.migrate(
    source_collection="docs",
    batch_size=1000,
)

if not result.success:
    print(f"Completed with {result.vectors_failed} failures")
    for error in result.errors:
        print(f"  - {error}")
```

---

## :material-file-document: Migration Models

### :material-check-circle: MigrationResult

!!! info "Complete Migration Information"
    Detailed results from a migration operation.

```python title="migration_result.py" linenums="1"
@dataclass
class MigrationResult:
    success: bool                    # Overall success
    vectors_migrated: int            # Successfully migrated
    vectors_failed: int              # Failed vectors
    duration_seconds: float          # Total time
    source_collection: str           # Source name
    target_collection: str           # Target name
    started_at: datetime             # Start timestamp
    completed_at: datetime           # End timestamp
    errors: List[str]                # Error messages
    metadata: Dict[str, Any]         # Additional info
    
    @property
    def total_vectors(self) -> int:
        """Total vectors processed"""
        
    @property
    def success_rate(self) -> float:
        """Percentage successfully migrated"""
```

---

### :material-progress-clock: MigrationProgress

!!! example "Real-Time Progress"
    Live progress information during migration.

```python title="migration_progress.py" linenums="1"
@dataclass
class MigrationProgress:
    vectors_processed: int           # Processed so far
    total_vectors: int               # Total to migrate
    current_batch: int               # Current batch number
    total_batches: int               # Total batches
    elapsed_seconds: float           # Time elapsed
    errors: int                      # Errors encountered
    
    @property
    def percentage(self) -> float:
        """Completion percentage"""
        
    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining"""
```

---

### :material-calculator: MigrationEstimate

!!! tip "Pre-Migration Planning"
    Estimate migration requirements before execution.

```python title="migration_estimate.py" linenums="1"
@dataclass
class MigrationEstimate:
    total_vectors: int               # Vectors to migrate
    estimated_duration_seconds: float # Estimated time
    estimated_batches: int           # Number of batches
    source_dimension: int            # Vector dimension
    target_dimension: Optional[int]  # If different
    compatible: bool                 # Schema compatible
    warnings: List[str]              # Potential issues
    
    @property
    def vectors_per_second(self) -> float:
        """Estimated throughput"""
```

---

## :material-lightbulb: Best Practices

### :material-package-variant: Batch Sizing

!!! tip "Optimal Batch Sizes"
    Choose batch size based on your dataset size and constraints.

<div class="grid cards" markdown>

-   :material-memory: **Memory**

    ---

    Larger batches use more memory

-   :material-network: **Network**

    ---

    Larger batches reduce overhead

-   :material-database: **Store Limits**

    ---

    Some stores have payload limits (Qdrant: 33MB)

</div>

=== "Small Datasets"

    ```python title="small_dataset.py"
    # < 10K vectors
    batch_size = 500
    ```

=== "Medium Datasets"

    ```python title="medium_dataset.py"
    # 10K - 100K vectors
    batch_size = 1000
    ```

=== "Large Datasets"

    ```python title="large_dataset.py"
    # > 100K vectors
    batch_size = 2000
    ```

---

### :material-progress-check: Progress Monitoring

!!! example "Robust Progress Tracking"
    Implement detailed logging for large migrations.

```python title="progress_monitoring.py" linenums="1" hl_lines="6-14 17-19"
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def detailed_progress(progress):
    logger.info(
        f"[{datetime.now().isoformat()}] "
        f"Batch {progress.current_batch}/{progress.total_batches} | "
        f"{progress.vectors_processed:,}/{progress.total_vectors:,} vectors | "
        f"{progress.percentage:.1f}% | "
        f"ETA: {progress.eta_seconds:.0f}s | "
        f"Errors: {progress.errors}"
    )

migrator = VectorStoreMigrator(
    source=source,
    target=target,
    on_progress=detailed_progress,
)
```

---

### :material-restart-alert: Error Recovery

!!! failure "Graceful Failure Handling"
    Handle migration failures with proper logging and recovery.

```python title="error_recovery.py" linenums="1" hl_lines="7-20"
result = migrator.migrate(
    source_collection="production_data",
    target_collection="production_data_v2",
    batch_size=1000,
)

if not result.success:
    # Log errors
    for error in result.errors:
        logger.error(f"Migration error: {error}")
    
    # Decide on recovery strategy
    if result.success_rate > 95:
        logger.info("Migration mostly successful, proceeding")
    else:
        logger.error("Migration failed, rolling back")
        # Implement rollback logic
```

---

## :material-rocket-launch: Advanced Use Cases

### :material-shield-check: Pre-Production Validation

!!! success "Production-Grade Migration"
    Full pipeline with dry-run, validation, and verification.

```python title="production_migration.py" linenums="1" hl_lines="7-10 17-21 26-31 37-45"
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def safe_production_migration(
    source, target, collection_name, filter_criteria=None
):
    """Production-grade migration with validation and dry-run."""
    
    migrator = VectorStoreMigrator(
        source=source,
        target=target,
        max_retries=5,
        retry_delay=2.0,
    )
    
    # Step 1: Dry-run to estimate
    logger.info("Running dry-run estimation...")
    dry_result = migrator.migrate(
        source_collection=collection_name,
        filter=filter_criteria,
        dry_run=True,
    )
    
    logger.info(
        f"Dry-run: {dry_result.vectors_migrated} vectors, "
        f"{dry_result.duration_seconds:.1f}s estimated"
    )
    
    # Step 2: Validate counts
    if dry_result.vectors_migrated == 0:
        raise ValueError("No vectors to migrate, check filter criteria")
    
    if dry_result.vectors_migrated > 10_000_000:
        logger.warning("Large migration detected, consider incremental approach")
    
    # Step 3: Execute real migration
    logger.info("Starting real migration...")
    result = migrator.migrate(
        source_collection=collection_name,
        target_collection=f"{collection_name}_prod_{datetime.now().strftime('%Y%m%d')}",
        filter=filter_criteria,
        validate=True,
    )
    
    # Step 4: Verify success
    if result.success_rate < 99.9:
        raise MigrationError(
            f"Migration success rate too low: {result.success_rate}%"
        )
    
    logger.info(
        f"✓ Migration successful: {result.vectors_migrated} vectors, "
        f"{result.success_rate:.2f}% success rate"
    )
    
    return result

# Usage
result = safe_production_migration(
    source=dev_store,
    target=prod_store,
    collection_name="user_documents",
    filter_criteria={"verified": True, "active": True},
)
```

---

### :material-file-tree: Selective Migration

!!! info "Multi-Target Strategy"
    Migrate different data subsets to optimized targets.

=== "Fast Store"

    ```python title="fast_store.py" linenums="1" hl_lines="1-4 6-10"
    # High-priority to fast store
fast_migrator = VectorStoreMigrator(
    source=chromadb_source,
    target=qdrant_fast,
)

high_priority_result = fast_migrator.migrate(
    source_collection="documents",
    target_collection="documents_priority",
    filter={"priority": "high", "status": "active"},
)

# Migrate archived documents to cold storage
archive_migrator = VectorStoreMigrator(
    source=chromadb_source,
    target=s3_backed_store,
)

archive_result = archive_migrator.migrate(
    source_collection="documents",
    target_collection="documents_archive",
    filter={"status": "archived", "year": {"$lt": 2024}},
)

print(f"Fast store: {high_priority_result.vectors_migrated} vectors")
print(f"Archive: {archive_result.vectors_migrated} vectors")
```

### Multi-Environment Deployment

Deploy validated data across environments:

```python
def deploy_to_environment(env_name: str, target_store, filter_override=None):
    """Deploy vectors to specific environment with environment-specific filters."""
    
    base_filter = {"validated": True, "status": "active"}
    if filter_override:
        base_filter.update(filter_override)
    
    migrator = VectorStoreMigrator(
        source=staging_store,
        target=target_store,
        max_retries=5,
    )
    
    # Dry-run first
    dry_result = migrator.migrate(
        source_collection="product_embeddings",
        filter=base_filter,
        dry_run=True,
    )
    
    print(f"{env_name} dry-run: {dry_result.vectors_migrated} vectors")
    
    # Require approval for production
    if env_name == "production":
        approval = input(f"Deploy {dry_result.vectors_migrated} vectors to prod? [y/N]: ")
        if approval.lower() != 'y':
            print("Deployment cancelled")
            return None
    
    # Execute migration
    result = migrator.migrate(
        source_collection="product_embeddings",
        target_collection=f"product_embeddings_{env_name}",
        filter=base_filter,
        validate=True,
    )
    
    return result

# Deploy pipeline
dev_result = deploy_to_environment("dev", dev_store)
staging_result = deploy_to_environment("staging", staging_store)
prod_result = deploy_to_environment("production", prod_store)
```

---

## :material-application-brackets: Common Use Cases

### :material-cloud-check: Development to Production

!!! example "ChromaDB to Qdrant"
    Migrate from local dev to production cloud.

```python title="dev_to_prod.py" linenums="1" hl_lines="2 5-9 11-15 17-22"
# Development setup
dev_store = get_chromadb_service(host="localhost")

# Production setup
prod_store = get_qdrant_service(
    host="production.qdrant.com",
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY"),
)

migrator = VectorStoreMigrator(
    source=dev_store,
    target=prod_store,
    validate=True,
)

result = migrator.migrate(
    source_collection="dev_documents",
    target_collection="prod_documents",
    batch_size=1000,
)
```

### Store Comparison

Migrate data to test different stores:

```python
source = get_chromadb_service()

for target_name, target_store in [
    ("Milvus", get_milvus_service()),
    ("Qdrant", get_qdrant_service()),
]:
    migrator = VectorStoreMigrator(source=source, target=target_store)
    
    print(f"\nMigrating to {target_name}...")
    result = migrator.migrate(
        source_collection="test_data",
        target_collection="test_data",
    )
    
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Success rate: {result.success_rate:.1f}%")
```

### Backup and Restore

Create backups of vector data:

```python
# Backup from Qdrant to ChromaDB
backup_migrator = VectorStoreMigrator(
    source=qdrant_service,
    target=chromadb_backup,
)

backup_result = backup_migrator.migrate(
    source_collection="critical_data",
    target_collection=f"backup_{datetime.now().isoformat()}",
)

# Later: Restore from backup
restore_migrator = VectorStoreMigrator(
    source=chromadb_backup,
    target=qdrant_service,
)

restore_result = restore_migrator.migrate(
    source_collection="backup_2025_12_27",
    target_collection="critical_data_restored",
)
```

## Exceptions

### MigrationError

Base exception for all migration errors:

```python
from rag_toolkit.migration import MigrationError

try:
    result = migrator.migrate(source_collection="docs")
except MigrationError as e:
    print(f"Migration failed: {e}")
```

### CollectionNotFoundError

Raised when source collection doesn't exist:

```python
from rag_toolkit.migration import CollectionNotFoundError

try:
    estimate = migrator.estimate(collection_name="missing")
except CollectionNotFoundError:
    print("Collection doesn't exist in source store")
```

### ValidationError

Raised when post-migration validation fails:

```python
from rag_toolkit.migration import ValidationError

try:
    result = migrator.migrate(
        source_collection="docs",
        validate=True,
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Performance Tips

### Optimize Batch Size

Test different batch sizes to find optimal throughput:

```python
for batch_size in [500, 1000, 2000, 5000]:
    migrator = VectorStoreMigrator(source=source, target=target)
    
    result = migrator.migrate(
        source_collection="test",
        target_collection=f"test_{batch_size}",
        batch_size=batch_size,
    )
    
    throughput = result.vectors_migrated / result.duration_seconds
    print(f"Batch {batch_size}: {throughput:.0f} vectors/sec")
```

### Parallel Migrations

For multiple collections, consider parallel processing:

```python
from concurrent.futures import ThreadPoolExecutor

collections = ["docs", "images", "audio"]

def migrate_collection(collection_name):
    migrator = VectorStoreMigrator(source=source, target=target)
    return migrator.migrate(source_collection=collection_name)

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(migrate_collection, collections)
    
for result in results:
    print(f"{result.source_collection}: {result.success_rate}% success")
```

## Troubleshooting

### Slow Migration

If migration is slow:

1. **Increase batch size** (if memory allows)
2. **Check network latency** between stores
3. **Monitor resource usage** (CPU, memory, disk)
4. **Verify store performance** (check individual store metrics)

### Validation Failures

If validation consistently fails:

1. **Check store connectivity**
2. **Verify write permissions**
3. **Inspect error messages** in `result.errors`
4. **Try smaller batch size**

### Memory Issues

If running out of memory:

1. **Reduce batch size**
2. **Disable progress tracking** (if not needed)
3. **Close other applications**
4. **Consider streaming approaches** for very large datasets

## Next Steps & Roadmap

### Implemented Features (Phase 1 & 2 Priority 1)

✅ **Core Migration Engine**
- Batch processing with configurable sizes
- Progress tracking with callbacks
- Automatic validation
- Comprehensive error handling
- Migration estimation

✅ **Advanced Filtering & Safety (Phase 2 Priority 1)**
- Metadata-based filtering for selective migration
- Dry-run mode for zero-risk testing
- Retry logic with exponential backoff
- Production-grade error recovery

### Planned Enhancements

#### Phase 2 Priority 2: Data Continuity & Schema Evolution

**1. Incremental Migration**

*Problem:* Large migrations (millions of vectors) can take hours/days. If interrupted, must restart from scratch.

*Solution:* Checkpoint-based incremental migration with resume capability.

```python
# Technical implementation:
# - Maintain .migration_checkpoint.json with migrated IDs
# - Support resume=True to skip already-migrated vectors
# - Implement conflict resolution strategies (skip, overwrite, fail)
# - Enable continuous sync workflows

migrator.migrate(
    source_collection="large_dataset",
    incremental=True,
    resume=True,  # Resume from last checkpoint
    checkpoint_file=".migration_state.json",
)
```

**Engineering considerations:**
- Checkpoint persistence: Local file, Redis, or database?
- Checkpoint granularity: Per-batch, per-vector, or hybrid?
- Distributed migration: How to handle concurrent migrators?
- Checkpoint cleanup: TTL, manual deletion, or auto-cleanup?
- Error recovery: Partial batch failures in incremental mode

**2. Schema Mapping & Transformation**

*Problem:* Different vector stores use different schemas. Migration between Qdrant, Pinecone, and Weaviate requires manual field mapping.

*Solution:* Declarative schema mapping with transformation functions.

```python
# Technical implementation:
# - Field name mapping (dict-based)
# - Type conversion pipeline (callable transformers)
# - Default value injection
# - Field exclusion/inclusion lists
# - Pre/post-migration hooks

schema_mapping = {
    "field_mapping": {"doc_text": "text", "doc_id": "id"},
    "transforms": {
        "timestamp": lambda x: int(x.timestamp()),
        "tags": lambda x: ",".join(x) if isinstance(x, list) else x,
    },
    "exclude_fields": ["internal_cache", "temp_data"],
    "defaults": {"version": "2.0", "migrated_at": datetime.now()},
}

migrator.migrate(
    source_collection="docs",
    schema_mapping=schema_mapping,
)
```

**Engineering considerations:**
- Transform error handling: Skip vector, use default, or fail fast?
- Type safety: Runtime validation vs. static type checking?
- Performance: Transform overhead on large datasets (vectorization?)
- Bidirectional mapping: Support reverse migrations?
- Schema versioning: Track mapping versions for rollback?

#### Phase 3: Enterprise Features

**1. Command-Line Interface (CLI)**

*Problem:* Python API requires coding. DevOps teams need CLI for scripts/CI-CD.

*Solution:* Rich CLI with YAML configuration support.

```bash
# Technical implementation:
# - Click or Typer-based CLI
# - YAML/JSON config file support
# - Environment variable injection
# - Progress bars (rich library)
# - Exit codes for CI/CD integration

rag-migrate \
  --source qdrant://localhost:6333/docs \
  --target pinecone://api-key@index-name \
  --filter 'status=active,year=2025' \
  --batch-size 1000 \
  --dry-run \
  --retry 5 \
  --config migration.yaml
```

**Engineering considerations:**
- Connection string parsing: URL-based or explicit parameters?
- Secret management: Env vars, keyring, or external vault?
- Config file schema: YAML, TOML, JSON, or all three?
- Output formats: JSON, YAML, table, or all three?
- Logging: Structured logging (JSON) vs. human-readable?

**2. Parallel Migration (Multi-threaded/Multi-process)**

*Problem:* Single-threaded migration is slow for large datasets.

*Solution:* Parallel batch processing with configurable workers.

```python
# Technical implementation:
# - ThreadPoolExecutor for I/O-bound operations
# - ProcessPoolExecutor for CPU-bound transforms
# - Batch-level parallelism (not vector-level)
# - Graceful degradation on errors
# - Resource throttling

migrator.migrate(
    source_collection="large_dataset",
    parallel=True,
    max_workers=8,  # Auto-detect by default
    worker_type="thread",  # or "process"
)
```

**Engineering considerations:**
- GIL limitations: When to use threads vs. processes?
- Memory management: Worker memory limits to prevent OOM?
- Error propagation: How to handle worker failures?
- Progress tracking: Aggregate progress from multiple workers?
- Rate limiting: Coordinate workers to respect API limits?
- Batch ordering: Preserve order or allow out-of-order completion?

**3. Metrics & Observability**

*Problem:* Large migrations lack visibility. Need monitoring, alerting, and debugging.

*Solution:* Prometheus metrics, OpenTelemetry tracing, and structured logging.

```python
# Technical implementation:
# - Prometheus client for metrics export
# - OpenTelemetry spans for distributed tracing
# - Structured logging (JSON) with correlation IDs
# - Pluggable exporters (Prometheus, Grafana, Datadog)

from rag_toolkit.migration import MetricsExporter

migrator = VectorStoreMigrator(
    source=source,
    target=target,
    metrics_exporter=MetricsExporter(
        backend="prometheus",
        port=9090,
    ),
)

# Metrics exposed:
# - migration_vectors_total
# - migration_duration_seconds
# - migration_batch_size
# - migration_errors_total
# - migration_retry_attempts
```

**Engineering considerations:**
- Metric cardinality: Avoid label explosion
- Sampling: Full tracing vs. sampled for large migrations?
- Overhead: Metrics/tracing impact on throughput?
- Retention: How long to keep metrics/traces?
- Alerting: Built-in alerts or external alertmanager?

#### Phase 4: Advanced Capabilities

**1. Schema Version Migration**

- Migrate between different schema versions (v1 → v2)
- Automatic field deprecation handling
- Version compatibility checks

**2. Cross-Cloud Migration**

- AWS → GCP → Azure vector store migrations
- Network optimization for inter-cloud transfers
- Cost estimation for data egress

**3. Zero-Downtime Migration**

- Dual-write pattern during migration
- Automatic cutover with validation
- Rollback support

**4. Data Quality Validation**

- Embedding drift detection
- Outlier detection in migrated vectors
- Semantic similarity preservation checks

### Contributing

Interested in implementing these features? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Priority feature requests:
1. Incremental migration (high demand)
2. Schema mapping (cross-store compatibility)
3. CLI tool (DevOps workflows)
4. Parallel migration (performance)

### Architecture Decisions

**Retry logic choice:** Exponential backoff chosen over fixed delay for:
- Better handling of transient failures (network, rate limits)
- Reduced load on struggling services
- Industry standard pattern (AWS SDK, Google APIs)

**Dry-run implementation:** Read-only approach (no mock writes) because:
- True performance measurement
- Accurate filter validation
- Simpler implementation (no mocking layer)
- Predictable behavior

**Filter design:** Metadata-based (not content-based) because:
- Performance: No vector similarity computation needed
- Clarity: Explicit metadata fields are self-documenting
- Scalability: Metadata filtering supported by all stores
- Future: Can extend to vector-based filtering later

**Checkpoint format (planned):** JSON file (not database) because:
- Zero dependencies
- Human-readable for debugging
- Easy to version control
- Simple to backup/restore
- Trade-off: Not suitable for distributed scenarios (Phase 4)

## API Reference

For detailed API documentation, see:

- [VectorStoreMigrator API](autoapi/src/rag_toolkit/migration/migrator/index.html)
- [Migration Models API](autoapi/src/rag_toolkit/migration/models/index.html)
- [Migration Exceptions API](autoapi/src/rag_toolkit/migration/exceptions/index.html)
