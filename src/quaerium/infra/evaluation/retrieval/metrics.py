"""
Individual retrieval metric calculators.

This module implements specific retrieval metrics:
- Precision@K: What fraction of top-K results are relevant
- Recall@K: What fraction of relevant docs are in top-K
- MRR: Mean Reciprocal Rank of first relevant result
- NDCG: Normalized Discounted Cumulative Gain (ranking quality)
- Hit Rate: Whether any relevant doc was retrieved
"""

from __future__ import annotations

import math
from typing import Any

from quaerium.core.evaluation.metrics import MetricScore


class PrecisionAtK:
    """
    Precision@K metric calculator.

    Measures what fraction of the top-K retrieved documents are relevant.

    Formula:
        Precision@K = (# relevant docs in top-K) / K

    Example:
        >>> calc = PrecisionAtK(k=5)
        >>> score = calc.calculate(
        ...     predictions=["doc1", "doc2", "doc3", "doc4", "doc5"],
        ...     ground_truth=["doc1", "doc3", "doc7"]
        ... )
        >>> print(f"{score.name}: {score.value}")  # Precision@5: 0.4
    """

    def __init__(self, k: int):
        """
        Initialize Precision@K calculator.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"Precision@{self._k}"

    def calculate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate Precision@K.

        Args:
            predictions: List of retrieved document IDs (in ranked order)
            ground_truth: List of relevant document IDs
            **kwargs: Unused

        Returns:
            MetricScore with precision value (0-1)

        Example:
            >>> score = PrecisionAtK(k=5).calculate(
            ...     predictions=["d1", "d2", "d3", "d4", "d5"],
            ...     ground_truth=["d1", "d3"]
            ... )
            >>> assert score.value == 0.4  # 2/5
        """
        if not predictions:
            return MetricScore(name=self.name, value=0.0)

        # Take only top-K predictions
        top_k = predictions[: self._k]

        # Convert to sets for efficient lookup
        relevant_set = set(ground_truth)

        # Count how many top-K docs are relevant
        relevant_found = sum(1 for doc_id in top_k if doc_id in relevant_set)

        precision = relevant_found / len(top_k)

        return MetricScore(
            name=self.name,
            value=precision,
            metadata={"k": self._k, "relevant_found": relevant_found},
        )


class RecallAtK:
    """
    Recall@K metric calculator.

    Measures what fraction of relevant documents are in the top-K results.

    Formula:
        Recall@K = (# relevant docs in top-K) / (total # relevant docs)

    Example:
        >>> calc = RecallAtK(k=10)
        >>> score = calc.calculate(
        ...     predictions=["d1", "d2", "d3", "d4", "d5"],
        ...     ground_truth=["d1", "d3", "d7", "d9", "d10"]
        ... )
        >>> print(f"{score.name}: {score.value}")  # Recall@10: 0.4 (2/5 found)
    """

    def __init__(self, k: int):
        """
        Initialize Recall@K calculator.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"Recall@{self._k}"

    def calculate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate Recall@K.

        Args:
            predictions: List of retrieved document IDs (in ranked order)
            ground_truth: List of relevant document IDs
            **kwargs: Unused

        Returns:
            MetricScore with recall value (0-1)

        Example:
            >>> score = RecallAtK(k=10).calculate(
            ...     predictions=["d1", "d2", "d3"],
            ...     ground_truth=["d1", "d3", "d7", "d9"]
            ... )
            >>> assert score.value == 0.5  # 2/4 found
        """
        if not ground_truth:
            return MetricScore(name=self.name, value=0.0)

        if not predictions:
            return MetricScore(name=self.name, value=0.0)

        # Take only top-K predictions
        top_k = predictions[: self._k]

        # Convert to sets
        relevant_set = set(ground_truth)
        retrieved_set = set(top_k)

        # Count how many relevant docs were found
        relevant_found = len(relevant_set & retrieved_set)

        recall = relevant_found / len(relevant_set)

        return MetricScore(
            name=self.name,
            value=recall,
            metadata={
                "k": self._k,
                "relevant_found": relevant_found,
                "total_relevant": len(relevant_set),
            },
        )


class MRRCalculator:
    """
    Mean Reciprocal Rank (MRR) calculator.

    Measures the rank of the first relevant document.
    Higher rank (earlier position) gives higher score.

    Formula:
        MRR = 1 / (rank of first relevant doc)
        If no relevant doc found, MRR = 0

    Example:
        >>> calc = MRRCalculator()
        >>> score = calc.calculate(
        ...     predictions=["doc1", "doc2", "doc3", "doc4"],
        ...     ground_truth=["doc3", "doc7"]
        ... )
        >>> print(f"{score.name}: {score.value}")  # MRR: 0.333 (1/3)
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "MRR"

    def calculate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            predictions: List of retrieved document IDs (in ranked order)
            ground_truth: List of relevant document IDs
            **kwargs: Unused

        Returns:
            MetricScore with MRR value (0-1)

        Example:
            >>> score = MRRCalculator().calculate(
            ...     predictions=["d1", "d2", "d3"],
            ...     ground_truth=["d2", "d5"]
            ... )
            >>> assert score.value == 0.5  # Found at rank 2, so 1/2
        """
        if not predictions or not ground_truth:
            return MetricScore(name=self.name, value=0.0)

        relevant_set = set(ground_truth)

        # Find rank of first relevant document (1-indexed)
        for rank, doc_id in enumerate(predictions, start=1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                return MetricScore(
                    name=self.name,
                    value=mrr,
                    metadata={"first_relevant_rank": rank},
                )

        # No relevant document found
        return MetricScore(name=self.name, value=0.0, metadata={"first_relevant_rank": None})


class NDCGCalculator:
    """
    Normalized Discounted Cumulative Gain (NDCG) calculator.

    Measures ranking quality with position-based discounting.
    Relevant docs at higher ranks contribute more to the score.

    Formula:
        DCG@K = Σ(relevance / log2(rank + 1)) for rank in 1..K
        NDCG@K = DCG@K / IDCG@K

    where IDCG is the "ideal" DCG (perfect ranking).

    Example:
        >>> calc = NDCGCalculator(k=10)
        >>> score = calc.calculate(
        ...     predictions=["doc1", "doc2", "doc3"],
        ...     ground_truth=["doc1", "doc3", "doc7"]
        ... )
        >>> print(f"{score.name}: {score.value:.3f}")
    """

    def __init__(self, k: int):
        """
        Initialize NDCG calculator.

        Args:
            k: Number of top results to consider

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"NDCG@{self._k}"

    def calculate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate NDCG@K.

        Args:
            predictions: List of retrieved document IDs (in ranked order)
            ground_truth: List of relevant document IDs
            **kwargs: Unused

        Returns:
            MetricScore with NDCG value (0-1)

        Note:
            Binary relevance (1 if relevant, 0 if not) is used.
        """
        if not predictions or not ground_truth:
            return MetricScore(name=self.name, value=0.0)

        # Take only top-K
        top_k = predictions[: self._k]
        relevant_set = set(ground_truth)

        # Calculate DCG
        dcg = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            relevance = 1.0 if doc_id in relevant_set else 0.0
            dcg += relevance / math.log2(rank + 1)

        # Calculate IDCG (ideal DCG with perfect ranking)
        # Ideal ranking has all relevant docs first
        num_relevant_in_top_k = min(len(relevant_set), self._k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant_in_top_k + 1))

        # Avoid division by zero
        if idcg == 0.0:
            return MetricScore(name=self.name, value=0.0)

        ndcg = dcg / idcg

        return MetricScore(
            name=self.name,
            value=ndcg,
            metadata={"k": self._k, "dcg": dcg, "idcg": idcg},
        )


class HitRateCalculator:
    """
    Hit Rate calculator.

    Binary metric: 1 if at least one relevant doc was retrieved, 0 otherwise.

    Example:
        >>> calc = HitRateCalculator(k=10)
        >>> score = calc.calculate(
        ...     predictions=["doc1", "doc2"],
        ...     ground_truth=["doc3", "doc4"]
        ... )
        >>> print(f"{score.name}: {score.value}")  # Hit@10: 0.0
    """

    def __init__(self, k: int | None = None):
        """
        Initialize Hit Rate calculator.

        Args:
            k: Optional number of top results to consider (None = all)

        Raises:
            ValueError: If k <= 0
        """
        if k is not None and k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        """Return metric name."""
        if self._k is None:
            return "Hit Rate"
        return f"Hit@{self._k}"

    def calculate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate Hit Rate.

        Args:
            predictions: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            **kwargs: Unused

        Returns:
            MetricScore with value 0.0 or 1.0
        """
        if not predictions or not ground_truth:
            return MetricScore(name=self.name, value=0.0)

        # Take only top-K if specified
        docs_to_check = predictions[: self._k] if self._k else predictions
        relevant_set = set(ground_truth)

        # Check if any relevant doc was found
        hit = 1.0 if any(doc_id in relevant_set for doc_id in docs_to_check) else 0.0

        return MetricScore(name=self.name, value=hit, metadata={"k": self._k})


__all__ = [
    "PrecisionAtK",
    "RecallAtK",
    "MRRCalculator",
    "NDCGCalculator",
    "HitRateCalculator",
]
