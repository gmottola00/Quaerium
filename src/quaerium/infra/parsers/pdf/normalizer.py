"""Text normalization utilities for ingestion."""

from __future__ import annotations

from typing import Optional, Union

import ftfy
from charset_normalizer import from_bytes


def normalize_bytes(raw_bytes: bytes, fallback_encoding: str = "utf-8") -> str:
    """Normalize bytes into a cleaned UTF-8 string.

    The function attempts charset detection via ``charset-normalizer`` and
    falls back to the provided encoding. Final text anomalies are corrected
    with ``ftfy``.

    Args:
        raw_bytes: Input byte content to normalize.
        fallback_encoding: Encoding used if detection fails. Defaults to "utf-8".

    Returns:
        Normalized text as ``str``.

    Raises:
        ValueError: If ``raw_bytes`` is ``None``.
    """
    if raw_bytes is None:
        raise ValueError("raw_bytes cannot be None")

    result = from_bytes(raw_bytes).best()
    if result is None:
        decoded = raw_bytes.decode(fallback_encoding, errors="replace")
    else:
        decoded = str(result)

    return ftfy.fix_text(decoded)


def normalize_text(text: Union[str, bytes], fallback_encoding: str = "utf-8") -> str:
    """Normalize arbitrary text or bytes to a cleaned ``str``.

    Args:
        text: Text as ``str`` or ``bytes``.
        fallback_encoding: Encoding used when ``text`` is ``bytes`` and detection
            fails. Defaults to "utf-8".

    Returns:
        Normalized text as ``str``.

    Raises:
        ValueError: If ``text`` is ``None``.
    """
    if text is None:
        raise ValueError("text cannot be None")

    if isinstance(text, bytes):
        return normalize_bytes(text, fallback_encoding=fallback_encoding)

    return ftfy.fix_text(text)


__all__ = ["normalize_bytes", "normalize_text"]
