"""Tests for migration models and data structures."""

import pytest
from datetime import datetime, timedelta

from rag_toolkit.migration.models import (
    MigrationResult,
    MigrationEstimate,
    MigrationProgress,
)


class TestMigrationResult:
    """Test suite for MigrationResult model."""

    def test_initialization(self):
        """Test basic initialization."""
        started = datetime.now()
        completed = started + timedelta(seconds=10)
        
        result = MigrationResult(
            success=True,
            vectors_migrated=100,
            vectors_failed=0,
            duration_seconds=10.5,
            source_collection="source",
            target_collection="target",
            started_at=started,
            completed_at=completed,
        )
        
        assert result.success is True
        assert result.vectors_migrated == 100
        assert result.vectors_failed == 0
        assert result.duration_seconds == 10.5

    def test_total_vectors_property(self):
        """Test total_vectors computed property."""
        result = MigrationResult(
            success=True,
            vectors_migrated=85,
            vectors_failed=15,
            duration_seconds=5.0,
            source_collection="src",
            target_collection="tgt",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        
        assert result.total_vectors == 100

    def test_success_rate_property(self):
        """Test success_rate computed property."""
        result = MigrationResult(
            success=True,
            vectors_migrated=90,
            vectors_failed=10,
            duration_seconds=5.0,
            source_collection="src",
            target_collection="tgt",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        
        assert result.success_rate == 90.0

    def test_success_rate_zero_vectors(self):
        """Test success_rate when no vectors processed."""
        result = MigrationResult(
            success=False,
            vectors_migrated=0,
            vectors_failed=0,
            duration_seconds=0.0,
            source_collection="src",
            target_collection="tgt",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        
        assert result.success_rate == 0.0

    def test_with_errors(self):
        """Test result with error messages."""
        result = MigrationResult(
            success=False,
            vectors_migrated=50,
            vectors_failed=50,
            duration_seconds=5.0,
            source_collection="src",
            target_collection="tgt",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=["Error 1", "Error 2"],
        )
        
        assert len(result.errors) == 2
        assert result.success is False

    def test_with_metadata(self):
        """Test result with custom metadata."""
        result = MigrationResult(
            success=True,
            vectors_migrated=100,
            vectors_failed=0,
            duration_seconds=5.0,
            source_collection="src",
            target_collection="tgt",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            metadata={"batch_size": 50, "validated": True},
        )
        
        assert result.metadata["batch_size"] == 50
        assert result.metadata["validated"] is True


class TestMigrationEstimate:
    """Test suite for MigrationEstimate model."""

    def test_initialization(self):
        """Test basic initialization."""
        estimate = MigrationEstimate(
            total_vectors=1000,
            estimated_duration_seconds=10.0,
            estimated_batches=10,
            source_dimension=384,
        )
        
        assert estimate.total_vectors == 1000
        assert estimate.estimated_duration_seconds == 10.0
        assert estimate.estimated_batches == 10
        assert estimate.source_dimension == 384

    def test_vectors_per_second_property(self):
        """Test vectors_per_second computed property."""
        estimate = MigrationEstimate(
            total_vectors=1000,
            estimated_duration_seconds=10.0,
            estimated_batches=10,
            source_dimension=384,
        )
        
        assert estimate.vectors_per_second == 100.0

    def test_vectors_per_second_zero_duration(self):
        """Test vectors_per_second when duration is zero."""
        estimate = MigrationEstimate(
            total_vectors=1000,
            estimated_duration_seconds=0.0,
            estimated_batches=10,
            source_dimension=384,
        )
        
        assert estimate.vectors_per_second == 0.0

    def test_with_dimension_mismatch(self):
        """Test estimate with different source and target dimensions."""
        estimate = MigrationEstimate(
            total_vectors=1000,
            estimated_duration_seconds=10.0,
            estimated_batches=10,
            source_dimension=384,
            target_dimension=768,
            compatible=False,
        )
        
        assert estimate.source_dimension == 384
        assert estimate.target_dimension == 768
        assert estimate.compatible is False

    def test_with_warnings(self):
        """Test estimate with warnings."""
        estimate = MigrationEstimate(
            total_vectors=1000,
            estimated_duration_seconds=10.0,
            estimated_batches=10,
            source_dimension=384,
            warnings=["Large dataset", "Schema differences detected"],
        )
        
        assert len(estimate.warnings) == 2
        assert "Large dataset" in estimate.warnings


class TestMigrationProgress:
    """Test suite for MigrationProgress model."""

    def test_initialization(self):
        """Test basic initialization."""
        progress = MigrationProgress(
            vectors_processed=50,
            total_vectors=100,
            current_batch=5,
            total_batches=10,
            elapsed_seconds=5.0,
        )
        
        assert progress.vectors_processed == 50
        assert progress.total_vectors == 100
        assert progress.current_batch == 5
        assert progress.total_batches == 10

    def test_percentage_property(self):
        """Test percentage computed property."""
        progress = MigrationProgress(
            vectors_processed=75,
            total_vectors=100,
            current_batch=7,
            total_batches=10,
            elapsed_seconds=7.5,
        )
        
        assert progress.percentage == 75.0

    def test_percentage_zero_total(self):
        """Test percentage when total is zero."""
        progress = MigrationProgress(
            vectors_processed=0,
            total_vectors=0,
            current_batch=0,
            total_batches=0,
            elapsed_seconds=0.0,
        )
        
        assert progress.percentage == 0.0

    def test_eta_seconds_property(self):
        """Test eta_seconds computed property."""
        progress = MigrationProgress(
            vectors_processed=50,
            total_vectors=100,
            current_batch=5,
            total_batches=10,
            elapsed_seconds=10.0,
        )
        
        # Rate: 50 vectors / 10 seconds = 5 vectors/second
        # Remaining: 50 vectors
        # ETA: 50 / 5 = 10 seconds
        assert progress.eta_seconds == 10.0

    def test_eta_seconds_zero_processed(self):
        """Test eta_seconds when no vectors processed yet."""
        progress = MigrationProgress(
            vectors_processed=0,
            total_vectors=100,
            current_batch=0,
            total_batches=10,
            elapsed_seconds=0.0,
        )
        
        assert progress.eta_seconds == 0.0

    def test_with_errors(self):
        """Test progress with errors."""
        progress = MigrationProgress(
            vectors_processed=50,
            total_vectors=100,
            current_batch=5,
            total_batches=10,
            elapsed_seconds=5.0,
            errors=3,
        )
        
        assert progress.errors == 3
