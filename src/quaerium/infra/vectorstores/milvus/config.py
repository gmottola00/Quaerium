"""Configuration models for Milvus vector store access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus connection configuration."""

    uri: str
    alias: str = "default"
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: str = "default"
    secure: bool = False
    timeout: Optional[float] = None

