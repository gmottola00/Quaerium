"""Unit tests for VectorStoreMigrator."""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime

from quaerium.migration import VectorStoreMigrator, MigrationError, ValidationError
from quaerium.migration.exceptions import CollectionNotFoundError
from quaerium.migration.models import MigrationProgress


class TestVectorStoreMigrator:
    """Test suite for VectorStoreMigrator class."""

    def test_init(self, mock_source_store, mock_target_store):
        """Test migrator initialization."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        assert migrator.source == mock_source_store
        assert migrator.target == mock_target_store
        assert migrator.validate is True
        assert migrator.on_progress is None

    def test_init_with_progress_callback(self, mock_source_store, mock_target_store):
        """Test migrator with progress callback."""
        callback = Mock()
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            on_progress=callback,
        )
        
        assert migrator.on_progress == callback

    def test_estimate_basic(self, mock_source_store, mock_target_store):
        """Test migration estimation."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        estimate = migrator.estimate(collection_name="test_collection", batch_size=10)
        
        assert estimate.total_vectors == 100
        assert estimate.estimated_batches == 10
        assert estimate.estimated_duration_seconds > 0
        assert estimate.compatible is True

    def test_estimate_empty_collection(self, mock_source_store, mock_target_store):
        """Test estimation with empty collection."""
        mock_source_store.count.return_value = 0
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        with pytest.raises(CollectionNotFoundError):
            migrator.estimate(collection_name="empty_collection")

    def test_migrate_basic(self, mock_source_store, mock_target_store):
        """Test basic migration."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,  # Disable validation for simpler test
        )
        
        result = migrator.migrate(
            source_collection="source_col",
            target_collection="target_col",
            batch_size=50,
        )
        
        assert result.success is True
        assert result.vectors_migrated > 0
        assert result.vectors_failed == 0
        assert result.source_collection == "source_col"
        assert result.target_collection == "target_col"
        assert result.duration_seconds > 0

    def test_migrate_with_validation(self, mock_source_store, mock_target_store):
        """Test migration with validation enabled."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=True,
        )
        
        result = migrator.migrate(
            source_collection="source_col",
            batch_size=50,
        )
        
        # Validation should pass
        assert result.success is True
        assert result.metadata.get("validated") is True

    def test_migrate_progress_callback(self, mock_source_store, mock_target_store):
        """Test that progress callback is called during migration."""
        progress_calls = []
        
        def on_progress(progress: MigrationProgress):
            progress_calls.append(progress)
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            on_progress=on_progress,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
        )
        
        # Progress callback should have been called
        assert len(progress_calls) > 0
        
        # Check progress data
        last_progress = progress_calls[-1]
        assert last_progress.vectors_processed > 0
        assert last_progress.total_vectors == 100
        assert last_progress.percentage > 0

    def test_migrate_empty_collection(self, mock_source_store, mock_target_store):
        """Test migration of empty collection."""
        mock_source_store.count.return_value = 0
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        result = migrator.migrate(source_collection="empty_col")
        
        assert result.success is False
        assert result.vectors_migrated == 0
        assert len(result.errors) > 0

    def test_migrate_batch_error_handling(self, mock_source_store, mock_target_store):
        """Test error handling during batch processing."""
        # Make target store fail on add_vectors
        mock_target_store.add_vectors.side_effect = Exception("Insert failed")
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
        )
        
        assert result.success is False
        assert len(result.errors) > 0

    def test_migrate_default_target_collection(self, mock_source_store, mock_target_store):
        """Test that target collection defaults to source collection name."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(source_collection="my_collection")
        
        assert result.source_collection == "my_collection"
        assert result.target_collection == "my_collection"

    def test_migrate_custom_target_collection(self, mock_source_store, mock_target_store):
        """Test migration with custom target collection name."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="source",
            target_collection="custom_target",
        )
        
        assert result.source_collection == "source"
        assert result.target_collection == "custom_target"

    def test_validation_failure(self, mock_source_store, mock_target_store):
        """Test behavior when validation fails."""
        # Override the count to always return less than expected during validation
        def mock_count_with_failure(collection_name: str):
            return 50  # Always return 50, less than expected 100
        
        mock_target_store.count = Mock(side_effect=mock_count_with_failure)
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=True,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
        )
        
        # Migration should complete but with validation warning
        assert len(result.errors) > 0
        assert any("Validation" in error for error in result.errors)

    def test_migration_result_properties(self, mock_source_store, mock_target_store):
        """Test MigrationResult computed properties."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(source_collection="test_col")
        
        # Test computed properties
        assert result.total_vectors == result.vectors_migrated + result.vectors_failed
        assert 0 <= result.success_rate <= 100
        
        if result.total_vectors > 0:
            expected_rate = (result.vectors_migrated / result.total_vectors) * 100
            assert result.success_rate == pytest.approx(expected_rate, rel=1e-5)
