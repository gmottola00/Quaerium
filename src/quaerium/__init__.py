"""
rag-toolkit: Advanced RAG library with multi-vectorstore support.

A production-ready toolkit for building Retrieval-Augmented Generation (RAG) pipelines
with support for multiple vector stores, LLM providers, and document formats.

Key Features:
    - Protocol-based architecture for easy extensibility
    - Multi-provider support (Ollama, OpenAI, custom)
    - Vector store abstractions (Milvus, Pinecone, Qdrant)
    - Advanced RAG pipeline (query rewriting, reranking, context assembly)
    - Document parsing (PDF, DOCX, text)
    - Smart chunking strategies (dynamic, token-based)
    - LLM-based metadata extraction for structured information
    - Metadata enrichment for improved retrieval quality

Example:
    >>> from quaerium import RagPipeline, OllamaEmbedding, OllamaLLMClient
    >>> from quaerium.infra.vectorstore.milvus import MilvusVectorStore
    >>>
    >>> # Initialize components
    >>> embedding = OllamaEmbedding(model="nomic-embed-text")
    >>> llm = OllamaLLMClient(model="llama3.2")
    >>> vectorstore = MilvusVectorStore(host="localhost")
    >>>
    >>> # Build and run pipeline
    >>> pipeline = RagPipeline.from_components(
    ...     vectorstore=vectorstore,
    ...     embedding=embedding,
    ...     llm=llm
    ... )
    >>> response = pipeline.run("What is RAG?")
    >>> print(response.answer)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Gianmarco Mottola"
__license__ = "MIT"

# ============================================================================
# Core Protocols (Type Definitions)
# ============================================================================

from quaerium.core.chunking import Chunk, TokenChunk, MetadataEnricher
from quaerium.core.embedding import EmbeddingClient
from quaerium.core.llm import LLMClient
from quaerium.core.metadata import LLMMetadataExtractor
from quaerium.core.vectorstore import VectorStoreClient

# ============================================================================
# Infrastructure Implementations
# ============================================================================

# Lazy imports for optional dependencies
def _import_ollama_embedding() -> type:
    """Lazy import for Ollama embedding (requires ollama package)."""
    try:
        from quaerium.infra.embedding.ollama import OllamaEmbeddingClient
        return OllamaEmbeddingClient
    except ImportError as e:
        raise ImportError(
            "Ollama support requires the 'ollama' package. "
            "Install it with: pip install rag-toolkit[ollama]"
        ) from e


def _import_openai_embedding() -> type:
    """Lazy import for OpenAI embedding (requires openai package)."""
    try:
        from quaerium.infra.embedding.openai import OpenAIEmbeddingClient
        return OpenAIEmbeddingClient
    except ImportError as e:
        raise ImportError(
            "OpenAI support requires the 'openai' package. "
            "Install it with: pip install rag-toolkit[openai]"
        ) from e


def _import_ollama_llm() -> type:
    """Lazy import for Ollama LLM (requires ollama package)."""
    try:
        from quaerium.infra.llm.ollama import OllamaLLMClient
        return OllamaLLMClient
    except ImportError as e:
        raise ImportError(
            "Ollama support requires the 'ollama' package. "
            "Install it with: pip install rag-toolkit[ollama]"
        ) from e


def _import_openai_llm() -> type:
    """Lazy import for OpenAI LLM (requires openai package)."""
    try:
        from quaerium.infra.llm.openai import OpenAILLMClient
        return OpenAILLMClient
    except ImportError as e:
        raise ImportError(
            "OpenAI support requires the 'openai' package. "
            "Install it with: pip install rag-toolkit[openai]"
        ) from e


# Public lazy loaders
def get_ollama_embedding() -> type:
    """Get OllamaEmbedding class (lazy loaded)."""
    return _import_ollama_embedding()


def get_openai_embedding() -> type:
    """Get OpenAIEmbedding class (lazy loaded)."""
    return _import_openai_embedding()


def get_ollama_llm() -> type:
    """Get OllamaLLMClient class (lazy loaded)."""
    return _import_ollama_llm()


def get_openai_llm() -> type:
    """Get OpenAILLMClient class (lazy loaded)."""
    return _import_openai_llm()


# ============================================================================
# RAG Components
# ============================================================================

from quaerium.rag.models import RagResponse, RetrievedChunk
from quaerium.rag.pipeline import RagPipeline

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Core Protocols
    "Chunk",
    "TokenChunk",
    "EmbeddingClient",
    "LLMClient",
    "VectorStoreClient",
    # Metadata & Enrichment
    "LLMMetadataExtractor",
    "MetadataEnricher",
    # Lazy loaders for implementations
    "get_ollama_embedding",
    "get_openai_embedding",
    "get_ollama_llm",
    "get_openai_llm",
    # RAG Components
    "RagPipeline",
    "RagResponse",
    "RetrievedChunk",
]


# ============================================================================
# Convenience imports (for backward compatibility)
# ============================================================================

def __getattr__(name: str) -> type:
    """
    Lazy attribute access for backward compatibility.
    
    This allows users to do:
        from quaerium import OllamaEmbedding
    
    Instead of:
        OllamaEmbedding = rag_toolkit.get_ollama_embedding()
    """
    lazy_imports = {
        "OllamaEmbedding": _import_ollama_embedding,
        "OpenAIEmbedding": _import_openai_embedding,
        "OllamaLLMClient": _import_ollama_llm,
        "OpenAILLMClient": _import_openai_llm,
    }
    
    if name in lazy_imports:
        return lazy_imports[name]()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
