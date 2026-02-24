"""Infrastructure - Document parsing implementations.

This module contains concrete implementations of parsers for various document formats.
Use the factory to create fully configured services.
"""

from .factory import create_ingestion_service, create_lightweight_ingestion_service

__all__ = [
    "create_ingestion_service",
    "create_lightweight_ingestion_service",
]
