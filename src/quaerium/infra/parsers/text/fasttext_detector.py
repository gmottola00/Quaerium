"""Language detection utilities using fastText or lingua-py."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    import fasttext
except ImportError:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

try:
    from lingua import Language, LanguageDetector, LanguageDetectorBuilder
except ImportError:  # pragma: no cover - optional dependency
    Language = None  # type: ignore
    LanguageDetector = None  # type: ignore
    LanguageDetectorBuilder = None  # type: ignore

DEFAULT_FASTTEXT_MODEL = "models/lid.176.bin"


@lru_cache(maxsize=1)
def _lingua_detector() -> Optional[LanguageDetector]:
    """Instantiate and cache a lingua detector if available."""
    if LanguageDetectorBuilder is None:
        return None

    languages = list(Language) if Language else []
    if not languages:
        return None

    return LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()


@lru_cache(maxsize=2)
def _fasttext_model(model_path: str) -> Optional["fasttext.FastText._FastText"]:
    """Load and cache a fastText language model."""
    if fasttext is None:
        return None

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"fastText model not found at {path}")

    return fasttext.load_model(path.as_posix())


def _detect_with_fasttext(text: str, model_path: str) -> Optional[str]:
    """Run detection via fastText returning ISO code or ``None`` on errors."""
    model = _fasttext_model(model_path)
    if model is None:
        return None

    labels, probabilities = model.predict(text)
    if not labels:
        return None

    label = labels[0]
    return label.replace("__label__", "")


def _detect_with_lingua(text: str) -> Optional[str]:
    """Run detection via lingua returning ISO code or ``None`` on errors."""
    detector = _lingua_detector()
    if detector is None:
        return None

    language = detector.detect_language_of(text)
    if language is None:
        return None

    try:
        return language.iso_code_639_1.name.lower()
    except AttributeError:
        return None


def detect_language(text: str, max_chars: int = 5000, model_path: Optional[str] = None) -> str:
    """Detect the primary language of the given text.

    Args:
        text: Input text.
        max_chars: Maximum characters to consider for detection. Defaults to 5000.
        model_path: Optional path to a fastText model. Defaults to ``models/lid.176.bin``.

    Returns:
        A lowercase ISO-639-1 code (e.g., ``"it"``, ``"en"``) or ``"unknown"``.

    Raises:
        RuntimeError: If fastText is requested but the model file is missing and
            no alternative detector is available.
    """
    snippet = (text or "").strip()
    if not snippet:
        return "unknown"

    normalized_snippet = snippet[:max_chars].replace("\n", " ")
    target_model = model_path or DEFAULT_FASTTEXT_MODEL

    if fasttext is not None:
        try:
            lang = _detect_with_fasttext(normalized_snippet, target_model)
            if lang:
                return lang
        except FileNotFoundError as exc:
            if _lingua_detector() is None:
                raise RuntimeError(
                    f"fastText model not found at {target_model} and no lingua detector available"
                ) from exc
        except Exception:
            # Fallback to lingua on any fastText error.
            pass

    lingua_lang = _detect_with_lingua(normalized_snippet)
    if lingua_lang:
        return lingua_lang

    return "unknown"


__all__ = ["detect_language"]
