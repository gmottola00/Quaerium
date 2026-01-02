"""Custom exceptions for migration operations."""


class MigrationError(Exception):
    """Base exception for migration-related errors."""

    pass


class ValidationError(MigrationError):
    """Raised when migration validation fails."""

    pass


class SchemaCompatibilityError(MigrationError):
    """Raised when source and target schemas are incompatible."""

    pass


class CollectionNotFoundError(MigrationError):
    """Raised when a collection doesn't exist in the source store."""

    pass


class MigrationInterruptedError(MigrationError):
    """Raised when migration is interrupted or cancelled."""

    pass
