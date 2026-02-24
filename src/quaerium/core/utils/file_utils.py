"""File utility helpers for ingestion workflows."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

FilePath = Union[str, Path]


def write_bytes_to_file(data: bytes, destination: FilePath) -> Path:
    """Write bytes to a file ensuring parent directories exist."""
    if data is None:
        raise ValueError("data cannot be None")

    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    return dest_path


def copy_to_temporary_file(
    source_path: FilePath,
    suffix: str = "",
    base_dir: Optional[FilePath] = None,
) -> Path:
    """Copy a file into a temporary location."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found at {src}")

    fd, tmp_name = tempfile.mkstemp(
        dir=Path(base_dir).as_posix() if base_dir else None, suffix=suffix
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    shutil.copy(src, tmp_path)
    return tmp_path


@contextmanager
def temporary_directory(
    base_dir: Optional[FilePath] = None, prefix: str = "ingestion_"
) -> Generator[Path, None, None]:
    """Context manager creating and cleaning a temporary directory."""
    temp_dir = Path(
        tempfile.mkdtemp(
            dir=Path(base_dir).as_posix() if base_dir else None, prefix=prefix
        )
    )
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["write_bytes_to_file", "copy_to_temporary_file", "temporary_directory"]
