"""Data models for migration operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MigrationResult:
    """Result of a migration operation.
    
    Attributes:
        success: Whether the migration completed successfully
        vectors_migrated: Total number of vectors successfully migrated
        vectors_failed: Number of vectors that failed to migrate
        duration_seconds: Total time taken for migration
        source_collection: Name of the source collection
        target_collection: Name of the target collection
        started_at: Timestamp when migration started
        completed_at: Timestamp when migration completed
        errors: List of error messages encountered during migration
        metadata: Additional metadata about the migration
    """

    success: bool
    vectors_migrated: int
    vectors_failed: int
    duration_seconds: float
    source_collection: str
    target_collection: str
    started_at: datetime
    completed_at: datetime
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_vectors(self) -> int:
        """Total number of vectors processed."""
        return self.vectors_migrated + self.vectors_failed

    @property
    def success_rate(self) -> float:
        """Percentage of successfully migrated vectors."""
        if self.total_vectors == 0:
            return 0.0
        return (self.vectors_migrated / self.total_vectors) * 100


@dataclass
class MigrationEstimate:
    """Estimated metrics for a migration operation.
    
    Attributes:
        total_vectors: Total number of vectors to migrate
        estimated_duration_seconds: Estimated time to complete
        estimated_batches: Number of batches required
        source_dimension: Vector dimension in source store
        target_dimension: Vector dimension in target store (if different)
        compatible: Whether schemas are compatible
        warnings: List of potential issues or warnings
    """

    total_vectors: int
    estimated_duration_seconds: float
    estimated_batches: int
    source_dimension: int
    target_dimension: Optional[int] = None
    compatible: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def vectors_per_second(self) -> float:
        """Estimated migration throughput."""
        if self.estimated_duration_seconds == 0:
            return 0.0
        return self.total_vectors / self.estimated_duration_seconds


@dataclass
class MigrationProgress:
    """Current progress of an ongoing migration.
    
    Attributes:
        vectors_processed: Number of vectors processed so far
        total_vectors: Total number of vectors to migrate
        current_batch: Current batch number
        total_batches: Total number of batches
        elapsed_seconds: Time elapsed since start
        errors: Number of errors encountered
    """

    vectors_processed: int
    total_vectors: int
    current_batch: int
    total_batches: int
    elapsed_seconds: float
    errors: int = 0

    @property
    def percentage(self) -> float:
        """Completion percentage."""
        if self.total_vectors == 0:
            return 0.0
        return (self.vectors_processed / self.total_vectors) * 100

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.vectors_processed == 0 or self.elapsed_seconds == 0:
            return 0.0
        rate = self.vectors_processed / self.elapsed_seconds
        remaining = self.total_vectors - self.vectors_processed
        return remaining / rate
