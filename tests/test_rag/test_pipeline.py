"""Test RAG pipeline integration."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from quaerium.rag.models import RagResponse, RetrievedChunk
from quaerium.rag.pipeline import RagPipeline


class MockSearchStrategy:
    """Mock search strategy for testing."""

    def __init__(self):
        self.search_calls: List[str] = []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return mock search results."""
        self.search_calls.append(query)
        return [
            {
                "id": f"doc_{i}",
                "text": f"Mock result {i} for query: {query}",
                "section_path": f"Section {i}",
                "metadata": {"source": f"source_{i}"},
                "page_numbers": [i],
                "source_chunk_id": f"chunk_{i}",
                "score": 1.0 - (i * 0.1),
            }
            for i in range(min(top_k, 3))
        ]


class MockQueryRewriter:
    """Mock query rewriter for testing."""

    def __init__(self):
        self.rewrite_calls: List[str] = []

    def rewrite(
        self, query: str, *, metadata_hint: Dict[str, str] | None = None
    ) -> str:
        """Return rewritten query."""
        self.rewrite_calls.append(query)
        return f"rewritten: {query}"


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self):
        self.rerank_calls: List[tuple[str, List[Dict[str, Any]]]] = []

    def rerank(
        self, query: str, hits: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return reranked results."""
        self.rerank_calls.append((query, hits))
        # Just return the same hits (mock behavior)
        return hits[:top_k]


class MockContextAssembler:
    """Mock context assembler for testing."""

    def __init__(self):
        self.assemble_calls: List[List[RetrievedChunk]] = []

    def assemble(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Return assembled context."""
        self.assemble_calls.append(chunks)
        # Just return the same chunks (mock behavior)
        return chunks


@pytest.fixture
def mock_search_strategy():
    """Provide mock search strategy."""
    return MockSearchStrategy()


@pytest.fixture
def mock_query_rewriter():
    """Provide mock query rewriter."""
    return MockQueryRewriter()


@pytest.fixture
def mock_reranker():
    """Provide mock reranker."""
    return MockReranker()


@pytest.fixture
def mock_context_assembler():
    """Provide mock context assembler."""
    return MockContextAssembler()


@pytest.fixture
def rag_pipeline(
    mock_search_strategy,
    mock_query_rewriter,
    mock_reranker,
    mock_context_assembler,
    mock_llm,
):
    """Create RAG pipeline with mock components."""
    return RagPipeline(
        vector_searcher=mock_search_strategy,
        rewriter=mock_query_rewriter,
        reranker=mock_reranker,
        assembler=mock_context_assembler,
        generator_llm=mock_llm,
    )


def test_rag_pipeline_initialization(rag_pipeline):
    """Test RAG pipeline initialization."""
    assert rag_pipeline.vector_searcher is not None
    assert rag_pipeline.rewriter is not None
    assert rag_pipeline.reranker is not None
    assert rag_pipeline.assembler is not None
    assert rag_pipeline.generator_llm is not None


def test_rag_pipeline_run_basic(rag_pipeline):
    """Test basic RAG pipeline execution."""
    question = "What is RAG?"
    
    response = rag_pipeline.run(question, top_k=3)
    
    # Check response type and structure
    assert isinstance(response, RagResponse)
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert isinstance(response.citations, list)
    assert len(response.citations) > 0
    
    # Check that citations are RetrievedChunk instances
    for citation in response.citations:
        assert isinstance(citation, RetrievedChunk)
        assert citation.id
        assert citation.text
        assert isinstance(citation.metadata, dict)
        assert isinstance(citation.page_numbers, list)


def test_rag_pipeline_component_interaction(
    rag_pipeline,
    mock_query_rewriter,
    mock_search_strategy,
    mock_reranker,
    mock_context_assembler,
):
    """Test that all pipeline components are called correctly."""
    question = "Test question"
    
    response = rag_pipeline.run(question, top_k=2)
    
    # Verify rewriter was called
    assert len(mock_query_rewriter.rewrite_calls) == 1
    assert mock_query_rewriter.rewrite_calls[0] == question
    
    # Verify search was called with rewritten query
    assert len(mock_search_strategy.search_calls) == 1
    assert "rewritten:" in mock_search_strategy.search_calls[0]
    
    # Verify reranker was called
    assert len(mock_reranker.rerank_calls) == 1
    
    # Verify assembler was called
    assert len(mock_context_assembler.assemble_calls) == 1
    
    # Verify response is valid
    assert isinstance(response, RagResponse)


def test_rag_pipeline_with_metadata_hint(rag_pipeline):
    """Test RAG pipeline with metadata hints."""
    question = "What is the tender about?"
    metadata_hint = {"tender_code": "ABC123", "lot_id": "LOT-001"}
    
    response = rag_pipeline.run(question, metadata_hint=metadata_hint, top_k=3)
    
    assert isinstance(response, RagResponse)
    assert response.answer
    assert response.citations


def test_rag_pipeline_variable_top_k(rag_pipeline):
    """Test RAG pipeline with different top_k values."""
    question = "Test question"
    
    # Small top_k
    response1 = rag_pipeline.run(question, top_k=1)
    assert len(response1.citations) <= 1
    
    # Larger top_k
    response2 = rag_pipeline.run(question, top_k=5)
    assert len(response2.citations) <= 5


def test_retrieved_chunk_structure():
    """Test RetrievedChunk dataclass structure."""
    chunk = RetrievedChunk(
        id="test-id",
        text="Test text content",
        section_path="Section 1 > Subsection A",
        metadata={"source": "doc1", "type": "tender"},
        page_numbers=[1, 2],
        source_chunk_id="chunk-123",
        score=0.95,
    )
    
    assert chunk.id == "test-id"
    assert chunk.text == "Test text content"
    assert chunk.section_path == "Section 1 > Subsection A"
    assert chunk.metadata["source"] == "doc1"
    assert chunk.page_numbers == [1, 2]
    assert chunk.source_chunk_id == "chunk-123"
    assert chunk.score == 0.95
    
    # Test optional fields
    chunk_minimal = RetrievedChunk(
        id="min-id",
        text="Minimal text",
        section_path=None,
        metadata={},
        page_numbers=[],
        source_chunk_id=None,
    )
    
    assert chunk_minimal.section_path is None
    assert chunk_minimal.source_chunk_id is None
    assert chunk_minimal.score is None


def test_rag_response_structure():
    """Test RagResponse dataclass structure."""
    chunks = [
        RetrievedChunk(
            id="1",
            text="Text 1",
            section_path="Section 1",
            metadata={},
            page_numbers=[1],
            source_chunk_id="chunk-1",
        ),
        RetrievedChunk(
            id="2",
            text="Text 2",
            section_path="Section 2",
            metadata={},
            page_numbers=[2],
            source_chunk_id="chunk-2",
        ),
    ]
    
    response = RagResponse(
        answer="This is the generated answer.",
        citations=chunks,
    )
    
    assert response.answer == "This is the generated answer."
    assert len(response.citations) == 2
    assert all(isinstance(c, RetrievedChunk) for c in response.citations)


def test_rag_pipeline_empty_results(mock_llm):
    """Test RAG pipeline behavior with empty search results."""
    # Create pipeline with mock that returns empty results
    empty_search = MockSearchStrategy()
    empty_search.search = lambda q, top_k: []  # Return empty list
    
    pipeline = RagPipeline(
        vector_searcher=empty_search,
        rewriter=MockQueryRewriter(),
        reranker=MockReranker(),
        assembler=MockContextAssembler(),
        generator_llm=mock_llm,
    )
    
    response = pipeline.run("Test question")
    
    # Should still return valid response even with no results
    assert isinstance(response, RagResponse)
    assert isinstance(response.answer, str)
    assert isinstance(response.citations, list)


def test_rag_pipeline_llm_integration(rag_pipeline, mock_llm):
    """Test that RAG pipeline uses LLM correctly."""
    # Set custom LLM response
    mock_llm.set_responses([
        "This is a detailed answer based on the provided context."
    ])
    
    response = rag_pipeline.run("What is RAG?", top_k=2)
    
    # Check that custom LLM response is used
    assert "detailed answer" in response.answer.lower()
