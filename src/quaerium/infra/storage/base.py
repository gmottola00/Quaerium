"""Storage client abstractions."""

from typing import Protocol, Optional


class StorageClient(Protocol):
    """Abstract storage client interface."""
    
    def ensure_bucket(self) -> None:
        """Ensure the storage bucket exists."""
        ...
    
    def build_path(self, tender_id: str, lot_id: Optional[str], filename: str) -> str:
        """Build a storage path for organizing files."""
        ...
    
    def upload_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> None:
        """Upload raw bytes to storage."""
        ...
    
    def download_bytes(self, path: str) -> bytes:
        """Download raw bytes from storage."""
        ...
    
    def delete(self, path: str) -> None:
        """Delete a file from storage."""
        ...
