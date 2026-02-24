"""Supabase storage implementation."""

from __future__ import annotations

import threading
from typing import Optional

from supabase import Client, create_client

from configs.config import settings


class SupabaseStorageClient:
    """Supabase storage adapter for file management."""

    def __init__(self, bucket_name: str, url: str, api_key: str, public: bool = False) -> None:
        """
        Initialize Supabase storage client.
        
        Args:
            bucket_name: Name of the storage bucket
            url: Supabase project URL
            api_key: Supabase API key (service role or anon key)
            public: Whether the bucket should be public
        """
        if not bucket_name:
            raise ValueError("bucket_name is required")
        self.bucket_name = bucket_name
        self.public = public
        self._client: Client = create_client(url, api_key)
        self._bucket_checked = False
        self._lock = threading.Lock()

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist."""
        if self._bucket_checked:
            return
        with self._lock:
            if self._bucket_checked:
                return
            existing = {b.name for b in (self._client.storage.list_buckets() or [])}
            if self.bucket_name not in existing:
                self._client.storage.create_bucket(self.bucket_name, {"public": self.public})
            self._bucket_checked = True

    def build_path(self, tender_id: str, lot_id: Optional[str], filename: str) -> str:
        """
        Build a hierarchical storage path.
        
        Structure: tenders/{tender_id}/lots/{lot_id}/documents/{filename}
        
        Args:
            tender_id: Tender identifier
            lot_id: Optional lot identifier
            filename: Name of the file
            
        Returns:
            Storage path string
        """
        parts = ["tenders", str(tender_id)]
        if lot_id:
            parts.append(f"lots/{lot_id}")
        parts.append("documents")
        parts.append(filename)
        return "/".join(parts)

    def upload_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> None:
        """
        Upload raw bytes to the configured bucket.
        
        Args:
            path: Storage path
            data: File data as bytes
            content_type: Optional MIME type (e.g., 'application/pdf')
            
        Raises:
            StorageException: If upload fails
        """
        self.ensure_bucket()
        options = {}
        if content_type:
            options["contentType"] = content_type
        self._client.storage.from_(self.bucket_name).upload(path, data, options)

    def download_bytes(self, path: str) -> bytes:
        """
        Download raw bytes from the configured bucket.
        
        Args:
            path: Storage path
            
        Returns:
            File data as bytes
            
        Raises:
            StorageException: If download fails
        """
        self.ensure_bucket()
        return self._client.storage.from_(self.bucket_name).download(path)

    def delete(self, path: str) -> None:
        """
        Delete a file from storage.
        
        Args:
            path: Storage path
            
        Raises:
            StorageException: If deletion fails
        """
        self.ensure_bucket()
        self._client.storage.from_(self.bucket_name).remove([path])


def get_storage_client() -> SupabaseStorageClient:
    """
    Factory function for creating a configured storage client.
    
    Returns:
        Configured SupabaseStorageClient instance
        
    Raises:
        RuntimeError: If STORAGE_BUCKET is not configured
        
    Example:
        >>> storage = get_storage_client()
        >>> storage.upload_bytes("path/to/file.pdf", pdf_bytes, "application/pdf")
    """
    bucket = settings.STORAGE_BUCKET
    if not bucket:
        raise RuntimeError("STORAGE_BUCKET env variable is required")
    
    # Prefer service role key for admin operations, fallback to anon key
    api_key = (
        settings.SUPABASE_SERVICE_ROLE_KEY.get_secret_value()
        if settings.SUPABASE_SERVICE_ROLE_KEY
        else settings.SUPABASE_KEY.get_secret_value()
    )
    
    return SupabaseStorageClient(
        bucket_name=bucket,
        url=settings.SUPABASE_URL,
        api_key=api_key,
        public=False,
    )
