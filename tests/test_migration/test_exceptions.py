"""Tests for migration exceptions."""

import pytest

from quaerium.migration.exceptions import (
    MigrationError,
    ValidationError,
    SchemaCompatibilityError,
    CollectionNotFoundError,
    MigrationInterruptedError,
)


class TestMigrationExceptions:
    """Test suite for migration exception classes."""

    def test_migration_error_base(self):
        """Test base MigrationError exception."""
        error = MigrationError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError inherits from MigrationError."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, MigrationError)
        assert isinstance(error, Exception)

    def test_schema_compatibility_error(self):
        """Test SchemaCompatibilityError."""
        error = SchemaCompatibilityError("Schema mismatch")
        assert str(error) == "Schema mismatch"
        assert isinstance(error, MigrationError)

    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError."""
        error = CollectionNotFoundError("Collection missing")
        assert str(error) == "Collection missing"
        assert isinstance(error, MigrationError)

    def test_migration_interrupted_error(self):
        """Test MigrationInterruptedError."""
        error = MigrationInterruptedError("Migration cancelled")
        assert str(error) == "Migration cancelled"
        assert isinstance(error, MigrationError)

    def test_exception_with_cause(self):
        """Test exception with cause chain."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise MigrationError("Wrapped error") from e
        except MigrationError as me:
            assert str(me) == "Wrapped error"
            assert isinstance(me.__cause__, ValueError)
            assert str(me.__cause__) == "Original error"

    def test_raising_and_catching_specific_error(self):
        """Test raising and catching specific error types."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Count mismatch")
        
        assert "Count mismatch" in str(exc_info.value)

    def test_catching_base_error(self):
        """Test catching derived errors with base class."""
        with pytest.raises(MigrationError):
            raise ValidationError("Specific error")

    def test_multiple_error_types(self):
        """Test that different error types are distinct."""
        validation_err = ValidationError("validation")
        schema_err = SchemaCompatibilityError("schema")
        collection_err = CollectionNotFoundError("not found")
        
        assert type(validation_err) != type(schema_err)
        assert type(schema_err) != type(collection_err)
        
        # But all inherit from MigrationError
        assert isinstance(validation_err, MigrationError)
        assert isinstance(schema_err, MigrationError)
        assert isinstance(collection_err, MigrationError)
