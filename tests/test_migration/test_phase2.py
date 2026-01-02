"""Tests for Phase 2 features: filtered migration, dry-run, and retry logic."""

import pytest
from unittest.mock import Mock, patch
from time import time

from rag_toolkit.migration import VectorStoreMigrator, MigrationError
from rag_toolkit.migration.migrator import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_BACKOFF,
)


class TestFilteredMigration:
    """Test suite for filtered migration functionality."""

    def test_migrate_with_filter(self, mock_source_store, mock_target_store):
        """Test migration with metadata filter."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        # Filter for specific category
        result = migrator.migrate(
            source_collection="test_col",
            filter={"category": "test"},
            batch_size=50,
        )
        
        assert result.success is True
        assert result.metadata.get("filter") == {"category": "test"}

    def test_migrate_with_multiple_filters(self, mock_source_store, mock_target_store):
        """Test migration with multiple filter criteria."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            filter={"category": "test", "status": "active"},
            batch_size=50,
        )
        
        assert result.success is True
        assert "filter" in result.metadata

    def test_apply_filter_basic(self, mock_source_store, mock_target_store):
        """Test _apply_filter method with basic matching."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        batch_data = [
            {"id": "1", "metadata": {"category": "test", "index": 1}},
            {"id": "2", "metadata": {"category": "prod", "index": 2}},
            {"id": "3", "metadata": {"category": "test", "index": 3}},
        ]
        
        filtered = migrator._apply_filter(batch_data, {"category": "test"})
        
        assert len(filtered) == 2
        assert filtered[0]["id"] == "1"
        assert filtered[1]["id"] == "3"

    def test_apply_filter_multiple_criteria(self, mock_source_store, mock_target_store):
        """Test _apply_filter with multiple criteria."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        batch_data = [
            {"id": "1", "metadata": {"category": "test", "year": 2024}},
            {"id": "2", "metadata": {"category": "test", "year": 2023}},
            {"id": "3", "metadata": {"category": "prod", "year": 2024}},
        ]
        
        filtered = migrator._apply_filter(
            batch_data, {"category": "test", "year": 2024}
        )
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "1"

    def test_apply_filter_no_matches(self, mock_source_store, mock_target_store):
        """Test _apply_filter when no items match."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        batch_data = [
            {"id": "1", "metadata": {"category": "test"}},
            {"id": "2", "metadata": {"category": "prod"}},
        ]
        
        filtered = migrator._apply_filter(batch_data, {"category": "staging"})
        
        assert len(filtered) == 0

    def test_apply_filter_missing_metadata(self, mock_source_store, mock_target_store):
        """Test _apply_filter with items missing metadata."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        batch_data = [
            {"id": "1", "metadata": {"category": "test"}},
            {"id": "2"},  # No metadata
            {"id": "3", "metadata": {}},  # Empty metadata
        ]
        
        filtered = migrator._apply_filter(batch_data, {"category": "test"})
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "1"


class TestDryRunMode:
    """Test suite for dry-run functionality."""

    def test_dry_run_basic(self, mock_source_store, mock_target_store):
        """Test basic dry-run migration."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
            dry_run=True,
        )
        
        # Should succeed without errors
        assert result.success is True
        assert result.metadata.get("dry_run") is True
        
        # Target store should not have been called for insertion
        mock_target_store.add_vectors.assert_not_called()

    def test_dry_run_with_filter(self, mock_source_store, mock_target_store):
        """Test dry-run with filtered migration."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            filter={"category": "test"},
            batch_size=50,
            dry_run=True,
        )
        
        assert result.success is True
        assert result.metadata.get("dry_run") is True
        assert result.metadata.get("filter") == {"category": "test"}

    def test_dry_run_skips_validation(self, mock_source_store, mock_target_store):
        """Test that dry-run skips validation even if enabled."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=True,  # Enable validation
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
            dry_run=True,
        )
        
        # Validation should be skipped in dry-run
        assert result.metadata.get("validated") is False

    def test_dry_run_counts_vectors(self, mock_source_store, mock_target_store):
        """Test that dry-run still counts vectors that would be migrated."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            batch_size=50,
            dry_run=True,
        )
        
        # Should still report how many vectors would be migrated
        assert result.vectors_migrated > 0
        assert result.vectors_failed == 0


class TestRetryLogic:
    """Test suite for retry logic with exponential backoff."""

    def test_retry_configuration(self, mock_source_store, mock_target_store):
        """Test that retry configuration is properly initialized."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=5,
            retry_delay=2.0,
            retry_backoff=3.0,
        )
        
        assert migrator.max_retries == 5
        assert migrator.retry_delay == 2.0
        assert migrator.retry_backoff == 3.0

    def test_default_retry_configuration(self, mock_source_store, mock_target_store):
        """Test default retry configuration."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        assert migrator.max_retries == DEFAULT_MAX_RETRIES
        assert migrator.retry_delay == DEFAULT_RETRY_DELAY
        assert migrator.retry_backoff == DEFAULT_RETRY_BACKOFF

    def test_fetch_batch_with_retry_success_first_attempt(
        self, mock_source_store, mock_target_store
    ):
        """Test successful fetch on first attempt."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        # Should succeed without retry
        batch = migrator._fetch_batch_with_retry("test_col", 0, 10)
        
        assert isinstance(batch, list)
        assert len(batch) > 0

    def test_fetch_batch_with_retry_success_after_failure(
        self, mock_source_store, mock_target_store
    ):
        """Test successful fetch after initial failures."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )
        
        # Mock to fail twice, then succeed
        call_count = [0]
        
        def mock_fetch_with_failure(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Temporary failure")
            return [{"id": "1", "vector": [0.1, 0.2]}]
        
        migrator._fetch_batch = Mock(side_effect=mock_fetch_with_failure)
        
        # Should succeed after retries
        batch = migrator._fetch_batch_with_retry("test_col", 0, 10)
        
        assert len(batch) == 1
        assert call_count[0] == 3

    def test_fetch_batch_with_retry_all_attempts_fail(
        self, mock_source_store, mock_target_store
    ):
        """Test failure after all retry attempts exhausted."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=3,
            retry_delay=0.01,
        )
        
        # Mock to always fail
        migrator._fetch_batch = Mock(side_effect=Exception("Persistent failure"))
        
        # Should raise MigrationError after all retries
        with pytest.raises(MigrationError) as exc_info:
            migrator._fetch_batch_with_retry("test_col", 0, 10)
        
        assert "Failed to fetch batch after 3 retries" in str(exc_info.value)

    def test_insert_batch_with_retry_success_first_attempt(
        self, mock_source_store, mock_target_store
    ):
        """Test successful insert on first attempt."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
        )
        
        batch_data = [{"id": "1", "vector": [0.1, 0.2], "text": "test", "metadata": {}}]
        
        # Should succeed without retry
        migrator._insert_batch_with_retry("test_col", batch_data)
        
        # Verify insert was called
        mock_target_store.add_vectors.assert_called()

    def test_insert_batch_with_retry_exponential_backoff(
        self, mock_source_store, mock_target_store
    ):
        """Test that exponential backoff is applied correctly."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=3,
            retry_delay=0.1,
            retry_backoff=2.0,
        )
        
        call_times = []
        
        def mock_insert_with_timing(*args, **kwargs):
            call_times.append(time())
            if len(call_times) < 3:
                raise Exception("Temporary failure")
        
        migrator._insert_batch = Mock(side_effect=mock_insert_with_timing)
        
        batch_data = [{"id": "1", "vector": [0.1], "text": "test"}]
        migrator._insert_batch_with_retry("test_col", batch_data)
        
        # Check that delays increased (with some tolerance)
        assert len(call_times) == 3
        
        # First retry after ~0.1s
        delay1 = call_times[1] - call_times[0]
        assert 0.08 < delay1 < 0.15
        
        # Second retry after ~0.2s (0.1 * 2.0 backoff)
        delay2 = call_times[2] - call_times[1]
        assert 0.15 < delay2 < 0.3

    def test_retry_logging(self, mock_source_store, mock_target_store, caplog):
        """Test that retry attempts are logged."""
        import logging
        
        caplog.set_level(logging.WARNING)
        
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=2,
            retry_delay=0.01,
        )
        
        # Mock to fail once then succeed
        call_count = [0]
        
        def mock_fetch_fail_once(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First attempt failed")
            return [{"id": "1"}]
        
        migrator._fetch_batch = Mock(side_effect=mock_fetch_fail_once)
        
        migrator._fetch_batch_with_retry("test_col", 0, 10)
        
        # Check that warning was logged
        assert any("Fetch failed" in record.message for record in caplog.records)
        assert any("Retrying" in record.message for record in caplog.records)


class TestPhase2Integration:
    """Integration tests combining Phase 2 features."""

    def test_filtered_migration_with_dry_run(
        self, mock_source_store, mock_target_store
    ):
        """Test combining filtered migration with dry-run."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            validate=False,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            filter={"category": "important"},
            dry_run=True,
            batch_size=50,
        )
        
        assert result.success is True
        assert result.metadata.get("dry_run") is True
        assert result.metadata.get("filter") == {"category": "important"}
        mock_target_store.add_vectors.assert_not_called()

    def test_filtered_migration_with_retry(
        self, mock_source_store, mock_target_store
    ):
        """Test filtered migration with retry on failure."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=2,
            retry_delay=0.01,
            validate=False,
        )
        
        # Mock to fail once
        call_count = [0]
        original_fetch = migrator._fetch_batch
        
        def mock_fetch_with_one_failure(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Simulated network error")
            return original_fetch(*args, **kwargs)
        
        migrator._fetch_batch = Mock(side_effect=mock_fetch_with_one_failure)
        
        result = migrator.migrate(
            source_collection="test_col",
            filter={"category": "test"},
            batch_size=50,
        )
        
        # Should succeed after retry
        assert result.success is True
        assert call_count[0] >= 2  # At least one retry occurred

    def test_all_features_combined(self, mock_source_store, mock_target_store):
        """Test all Phase 2 features working together."""
        migrator = VectorStoreMigrator(
            source=mock_source_store,
            target=mock_target_store,
            max_retries=3,
            retry_delay=0.01,
            retry_backoff=2.0,
            validate=True,
        )
        
        result = migrator.migrate(
            source_collection="test_col",
            target_collection="test_col_filtered",
            filter={"status": "active", "priority": "high"},
            dry_run=False,
            batch_size=100,
        )
        
        assert result.success is True
        assert result.metadata.get("filter") is not None
        assert result.metadata.get("dry_run") is False
        assert result.source_collection == "test_col"
        assert result.target_collection == "test_col_filtered"
