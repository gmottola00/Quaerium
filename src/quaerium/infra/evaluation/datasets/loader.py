"""
Dataset loaders for evaluation.

Provides loaders for various dataset formats (JSONL, CSV, etc.).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

from quaerium.core.evaluation.dataset import (
    DatasetMetadata,
    EvaluationExample,
)

logger = logging.getLogger(__name__)


class JSONLDataset:
    """
    Load evaluation dataset from JSONL format.

    JSONL Format (one JSON object per line):
    ```json
    {"query": "What is Python?", "relevant_doc_ids": ["doc1", "doc3"], "reference_answer": "Python is a programming language."}
    {"query": "How do I install Python?", "relevant_doc_ids": ["doc5", "doc7"]}
    ```

    Example:
        >>> dataset = JSONLDataset("eval_data.jsonl")
        >>> print(f"Dataset has {len(dataset)} examples")
        >>> for example in dataset:
        ...     print(f"Query: {example.query}")
        ...     print(f"Relevant docs: {example.relevant_doc_ids}")
    """

    def __init__(self, file_path: str | Path):
        """
        Initialize JSONL dataset loader.

        Args:
            file_path: Path to JSONL file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or has invalid format
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load examples
        self._examples = self._load_examples()

        if not self._examples:
            raise ValueError(f"Dataset file is empty: {file_path}")

        # Generate metadata
        self._metadata = self._generate_metadata()

        logger.info(
            f"Loaded JSONL dataset from {file_path}: "
            f"{len(self._examples)} examples"
        )

    def _load_examples(self) -> list[EvaluationExample]:
        """Load examples from JSONL file."""
        examples = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)

                    # Required fields
                    query = data.get("query")
                    relevant_doc_ids = data.get("relevant_doc_ids", [])

                    if not query:
                        logger.warning(
                            f"Line {line_num}: Missing 'query' field, skipping"
                        )
                        continue

                    if not relevant_doc_ids:
                        logger.warning(
                            f"Line {line_num}: Empty 'relevant_doc_ids', skipping"
                        )
                        continue

                    # Optional fields
                    reference_answer = data.get("reference_answer")
                    metadata = data.get("metadata", {})

                    example = EvaluationExample(
                        query=query,
                        relevant_doc_ids=relevant_doc_ids,
                        reference_answer=reference_answer,
                        metadata=metadata,
                    )

                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: Error creating example - {e}")
                    continue

        return examples

    def _generate_metadata(self) -> DatasetMetadata:
        """Generate metadata for the dataset."""
        return DatasetMetadata(
            name=self.file_path.stem,
            description=f"Evaluation dataset loaded from {self.file_path.name}",
            num_examples=len(self._examples),
            created_at=datetime.now().isoformat(),
            version="1.0",
            tags=["evaluation", "jsonl"],
        )

    def __len__(self) -> int:
        """Get number of examples in dataset."""
        return len(self._examples)

    def __iter__(self) -> Iterator[EvaluationExample]:
        """Iterate over evaluation examples."""
        return iter(self._examples)

    def __getitem__(self, index: int) -> EvaluationExample:
        """Get example by index."""
        return self._examples[index]

    @property
    def metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        return self._metadata


__all__ = ["JSONLDataset"]
