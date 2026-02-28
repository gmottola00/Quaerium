"""
Dataset protocols and models for evaluation.

This module provides protocols and data structures for managing
evaluation datasets with ground truth labels for testing RAG systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvaluationExample:
    """
    Single evaluation example with ground truth.

    Represents one query-answer pair with ground truth labels
    for evaluating RAG system quality.

    Attributes:
        query: User query/question
        relevant_doc_ids: List of relevant document IDs (for retrieval eval)
        reference_answer: Optional ground truth answer (for generation eval)
        metadata: Additional metadata (e.g., difficulty, category)

    Example:
        >>> example = EvaluationExample(
        ...     query="What is RAG?",
        ...     relevant_doc_ids=["doc1", "doc3", "doc7"],
        ...     reference_answer="RAG combines retrieval with generation.",
        ...     metadata={"category": "definition", "difficulty": "easy"}
        ... )
        >>> print(f"Query: {example.query}")
        >>> print(f"Relevant docs: {len(example.relevant_doc_ids)}")
    """

    query: str
    relevant_doc_ids: list[str]
    reference_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate evaluation example."""
        if not isinstance(self.query, str) or not self.query:
            raise ValueError("Query must be a non-empty string")
        if not isinstance(self.relevant_doc_ids, list):
            raise ValueError("relevant_doc_ids must be a list")
        if not self.relevant_doc_ids:
            raise ValueError("At least one relevant document ID required")


@dataclass
class DatasetMetadata:
    """
    Metadata about an evaluation dataset.

    Attributes:
        name: Dataset name
        description: Dataset description
        num_examples: Number of examples in dataset
        created_at: Creation timestamp (ISO format)
        version: Dataset version
        tags: Optional tags for categorization

    Example:
        >>> metadata = DatasetMetadata(
        ...     name="rag_eval_v1",
        ...     description="RAG evaluation dataset with 100 examples",
        ...     num_examples=100,
        ...     created_at="2024-01-15T10:30:00Z",
        ...     version="1.0",
        ...     tags=["rag", "qa", "technical"]
        ... )
        >>> print(f"{metadata.name}: {metadata.num_examples} examples")
    """

    name: str
    description: str
    num_examples: int
    created_at: str
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)


@runtime_checkable
class EvaluationDataset(Protocol):
    """
    Protocol for evaluation datasets.

    Defines the interface for loading and iterating over
    evaluation examples from various sources (JSONL, CSV, etc.).

    Implementations:
        - JSONLDataset: Load from JSONL files

    Example:
        >>> dataset: EvaluationDataset = JSONLDataset("eval_data.jsonl")
        >>> print(f"Dataset has {len(dataset)} examples")
        >>> for example in dataset:
        ...     print(f"Query: {example.query}")
    """

    def __len__(self) -> int:
        """
        Get number of examples in dataset.

        Returns:
            Number of evaluation examples
        """
        ...

    def __iter__(self) -> Iterator[EvaluationExample]:
        """
        Iterate over evaluation examples.

        Yields:
            EvaluationExample instances
        """
        ...

    def __getitem__(self, index: int) -> EvaluationExample:
        """
        Get example by index.

        Args:
            index: Example index (0-based)

        Returns:
            EvaluationExample at the given index

        Raises:
            IndexError: If index out of range
        """
        ...

    @property
    def metadata(self) -> DatasetMetadata:
        """
        Get dataset metadata.

        Returns:
            DatasetMetadata with information about the dataset
        """
        ...


__all__ = [
    "EvaluationExample",
    "DatasetMetadata",
    "EvaluationDataset",
]
