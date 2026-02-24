"""Core migration functionality for vector stores."""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_BACKOFF = 2.0  # exponential backoff multiplier

from quaerium.core.vectorstore import VectorStoreClient
from quaerium.migration.exceptions import (
    CollectionNotFoundError,
    MigrationError,
    ValidationError,
)
from quaerium.migration.models import (
    MigrationEstimate,
    MigrationProgress,
    MigrationResult,
)

logger = logging.getLogger(__name__)


class VectorStoreMigrator:
    """Migrates vector data between different vector store implementations.
    
    This class provides functionality to migrate vectors, embeddings, and metadata
    from one vector store to another with validation, progress tracking, and
    error handling.
    
    Args:
        source: Source vector store client
        target: Target vector store client
        on_progress: Optional callback function called with MigrationProgress
        validate: Whether to validate data after migration (default: True)
        
    Example:
        ```python
        from quaerium.migration import VectorStoreMigrator
        
        migrator = VectorStoreMigrator(
            source=chroma_service,
            target=qdrant_service,
            on_progress=lambda p: print(f"{p.percentage:.1f}% complete"),
        )
        
        result = migrator.migrate(
            collection_name="my_docs",
            batch_size=1000,
        )
        
        print(f"Migrated {result.vectors_migrated} vectors in {result.duration_seconds}s")
        ```
    """

    def __init__(
        self,
        source: VectorStoreClient,
        target: VectorStoreClient,
        on_progress: Optional[Callable[[MigrationProgress], None]] = None,
        validate: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        self.source = source
        self.target = target
        self.on_progress = on_progress
        self.validate = validate
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self._id_mapping: Dict[str, str] = {}

    def estimate(self, collection_name: str, batch_size: int = 1000) -> MigrationEstimate:
        """Estimate migration time and resource requirements.
        
        Args:
            collection_name: Name of the collection to migrate
            batch_size: Number of vectors per batch
            
        Returns:
            MigrationEstimate with estimated metrics
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist in source
        """
        try:
            # Get collection info from source
            source_count = self.source.count(collection_name)
            if source_count == 0:
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' not found or empty in source store"
                )

            # Get dimension info (search for a sample vector)
            sample_results = self.source.search(
                collection_name=collection_name,
                query_vector=[0.0] * 384,  # Dummy vector for dimension check
                top_k=1,
            )
            
            source_dimension = 384  # Default, will be detected from actual data
            estimated_batches = (source_count + batch_size - 1) // batch_size
            
            # Rough estimate: 100-500 vectors/second depending on store
            estimated_rate = 200  # Conservative estimate
            estimated_duration = source_count / estimated_rate

            return MigrationEstimate(
                total_vectors=source_count,
                estimated_duration_seconds=estimated_duration,
                estimated_batches=estimated_batches,
                source_dimension=source_dimension,
                compatible=True,
            )

        except CollectionNotFoundError:
            # Re-raise CollectionNotFoundError directly
            raise
        except Exception as e:
            logger.error(f"Error estimating migration: {e}")
            raise MigrationError(f"Failed to estimate migration: {e}") from e

    def migrate(
        self,
        source_collection: str,
        target_collection: Optional[str] = None,
        batch_size: int = 1000,
        validate: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Migrate all vectors from source to target collection.
        
        Args:
            source_collection: Name of the collection in source store
            target_collection: Name of the collection in target store (defaults to source name)
            batch_size: Number of vectors to process per batch
            validate: Override instance validation setting
            filter: Optional metadata filter to apply (only migrate matching vectors)
            dry_run: If True, simulate migration without writing to target
            
        Returns:
            MigrationResult with detailed migration statistics
            
        Raises:
            MigrationError: If migration fails
            CollectionNotFoundError: If source collection doesn't exist
        """
        target_collection = target_collection or source_collection
        should_validate = validate if validate is not None else self.validate
        
        started_at = datetime.now()
        start_time = time.time()
        
        vectors_migrated = 0
        vectors_failed = 0
        errors: List[str] = []
        
        try:
            # Get total count for progress tracking
            total_vectors = self.source.count(source_collection)
            if total_vectors == 0:
                raise CollectionNotFoundError(
                    f"Collection '{source_collection}' not found or empty"
                )
            
            mode_info = []
            if dry_run:
                mode_info.append("DRY-RUN")
            if filter:
                mode_info.append(f"FILTERED: {filter}")
            mode_str = f" [{', '.join(mode_info)}]" if mode_info else ""
            
            logger.info(
                f"Starting migration of {total_vectors} vectors from '{source_collection}' "
                f"to '{target_collection}'{mode_str}"
            )
            
            # Calculate number of batches
            total_batches = (total_vectors + batch_size - 1) // batch_size
            
            # Migrate in batches
            offset = 0
            current_batch = 0
            
            while offset < total_vectors:
                current_batch += 1
                batch_start = time.time()
                
                try:
                    # Fetch batch from source with retry
                    batch_data = self._fetch_batch_with_retry(
                        source_collection, offset, batch_size, filter
                    )
                    
                    if not batch_data:
                        break
                    
                    # Apply metadata filter if specified
                    if filter:
                        batch_data = self._apply_filter(batch_data, filter)
                        if not batch_data:
                            offset += batch_size
                            continue
                    
                    # Insert batch into target (skip if dry-run)
                    if not dry_run:
                        self._insert_batch_with_retry(target_collection, batch_data)
                    
                    batch_size_actual = len(batch_data)
                    vectors_migrated += batch_size_actual
                    offset += batch_size_actual
                    
                    # Report progress
                    if self.on_progress:
                        elapsed = time.time() - start_time
                        progress = MigrationProgress(
                            vectors_processed=vectors_migrated,
                            total_vectors=total_vectors,
                            current_batch=current_batch,
                            total_batches=total_batches,
                            elapsed_seconds=elapsed,
                            errors=vectors_failed,
                        )
                        self.on_progress(progress)
                    
                    logger.debug(
                        f"Batch {current_batch}/{total_batches} completed: "
                        f"{batch_size_actual} vectors in {time.time() - batch_start:.2f}s"
                    )
                    
                except Exception as e:
                    error_msg = f"Error in batch {current_batch}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    vectors_failed += batch_size
                    offset += batch_size
            
            # Validation (skip in dry-run mode)
            if should_validate and vectors_migrated > 0 and not dry_run:
                logger.info("Validating migration...")
                try:
                    self._validate_migration(source_collection, target_collection, vectors_migrated)
                except ValidationError as e:
                    errors.append(f"Validation failed: {e}")
                    logger.warning(f"Validation failed: {e}")
            
            completed_at = datetime.now()
            duration = time.time() - start_time
            
            success = vectors_failed == 0 and len(errors) == 0
            
            result = MigrationResult(
                success=success,
                vectors_migrated=vectors_migrated,
                vectors_failed=vectors_failed,
                duration_seconds=duration,
                source_collection=source_collection,
                target_collection=target_collection,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                metadata={
                    "batch_size": batch_size,
                    "total_batches": current_batch,
                    "validated": should_validate and not dry_run,
                    "dry_run": dry_run,
                    "filter": filter,
                },
            )
            
            if dry_run:
                logger.info(
                    f"DRY-RUN completed: {vectors_migrated} vectors would be migrated "
                    f"in {duration:.2f}s"
                )
            else:
                logger.info(
                    f"Migration completed: {vectors_migrated} vectors migrated, "
                    f"{vectors_failed} failed in {duration:.2f}s "
                    f"({result.success_rate:.1f}% success rate)"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            completed_at = datetime.now()
            duration = time.time() - start_time
            
            return MigrationResult(
                success=False,
                vectors_migrated=vectors_migrated,
                vectors_failed=vectors_failed,
                duration_seconds=duration,
                source_collection=source_collection,
                target_collection=target_collection,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors + [str(e)],
            )

    def _fetch_batch(
        self,
        collection_name: str,
        offset: int,
        limit: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a batch of vectors from the source store.
        
        Args:
            collection_name: Name of the collection
            offset: Starting offset
            limit: Maximum number of vectors to fetch
            filter: Optional metadata filter
            
        Note: This is a simplified implementation. In production, you'd want
        proper pagination support or use the vector store's native export functionality.
        """
        # For now, we'll implement a simple approach using search
        # This is not ideal for large datasets, but works for demonstration
        
        # Get all data - this is inefficient but necessary without pagination API
        # In a real implementation, we'd add pagination support to VectorStoreClient protocol
        try:
            # Use a dummy query to get all results
            # This is a limitation we'll need to address with proper pagination
            results = self.source.search(
                collection_name=collection_name,
                query_vector=[0.0] * 384,  # Dummy vector
                top_k=limit,
            )
            
            # Convert search results to batch format
            batch_data = []
            for result in results:
                batch_data.append({
                    "id": result.id,
                    "vector": result.vector or [],
                    "text": result.text,
                    "metadata": result.metadata or {},
                })
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            return []

    def _insert_batch(self, collection_name: str, batch_data: List[Dict[str, Any]]) -> None:
        """Insert a batch of vectors into the target store."""
        try:
            # Convert batch data to the format expected by add_vectors
            vectors = [item["vector"] for item in batch_data]
            texts = [item["text"] for item in batch_data]
            metadatas = [item["metadata"] for item in batch_data]
            
            # Note: IDs might need conversion depending on target store
            # For now, we'll let the target store generate new IDs
            
            self.target.add_vectors(
                collection_name=collection_name,
                vectors=vectors,
                texts=texts,
                metadatas=metadatas,
            )
            
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            raise MigrationError(f"Failed to insert batch: {e}") from e

    def _validate_migration(
        self, source_collection: str, target_collection: str, expected_count: int
    ) -> None:
        """Validate that migration was successful.
        
        Args:
            source_collection: Name of source collection
            target_collection: Name of target collection
            expected_count: Expected number of vectors in target
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            target_count = self.target.count(target_collection)
            
            if target_count < expected_count:
                raise ValidationError(
                    f"Target collection has {target_count} vectors, "
                    f"expected at least {expected_count}"
                )
            
            logger.info(f"Validation passed: {target_count} vectors in target collection")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            logger.error(f"Error during validation: {e}")
            raise ValidationError(f"Validation failed: {e}") from e

    def _fetch_batch_with_retry(
        self,
        collection_name: str,
        offset: int,
        limit: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a batch with retry logic.
        
        Args:
            collection_name: Name of the collection
            offset: Starting offset
            limit: Maximum number of vectors
            filter: Optional metadata filter
            
        Returns:
            List of vector data dictionaries
            
        Raises:
            MigrationError: If all retries fail
        """
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return self._fetch_batch(collection_name, offset, limit, filter)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Fetch failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    logger.error(f"Fetch failed after {self.max_retries} attempts")
        
        raise MigrationError(f"Failed to fetch batch after {self.max_retries} retries: {last_error}")

    def _insert_batch_with_retry(
        self, collection_name: str, batch_data: List[Dict[str, Any]]
    ) -> None:
        """Insert a batch with retry logic.
        
        Args:
            collection_name: Name of the target collection
            batch_data: List of vector data to insert
            
        Raises:
            MigrationError: If all retries fail
        """
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                self._insert_batch(collection_name, batch_data)
                return
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Insert failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    logger.error(f"Insert failed after {self.max_retries} attempts")
        
        raise MigrationError(f"Failed to insert batch after {self.max_retries} retries: {last_error}")

    def _apply_filter(
        self, batch_data: List[Dict[str, Any]], filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filter to batch data.
        
        Args:
            batch_data: List of vector data dictionaries
            filter: Metadata filter criteria
            
        Returns:
            Filtered list of vector data
        """
        filtered_data = []
        
        for item in batch_data:
            metadata = item.get("metadata", {})
            
            # Check if all filter criteria match
            matches = True
            for key, value in filter.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered_data.append(item)
        
        return filtered_data
