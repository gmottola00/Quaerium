"""
Composite retrieval evaluators.

This module provides composite evaluators that combine multiple
retrieval metrics into a single evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from quaerium.core.evaluation.metrics import RetrievalMetrics
from quaerium.infra.evaluation.retrieval.metrics import (
    HitRateCalculator,
    MRRCalculator,
    NDCGCalculator,
    PrecisionAtK,
    RecallAtK,
)

logger = logging.getLogger(__name__)


class StandardRetrievalEvaluator:
    """
    Standard retrieval evaluator with all common metrics.

    Computes Precision@K, Recall@K, MRR, NDCG, and Hit Rate
    for multiple K values in a single evaluation.

    Example:
        >>> evaluator = StandardRetrievalEvaluator(k_values=[5, 10])
        >>> metrics = evaluator.evaluate_retrieval(
        ...     query="What is RAG?",
        ...     retrieved_docs=[
        ...         {"id": "doc1", "score": 0.95},
        ...         {"id": "doc2", "score": 0.87},
        ...         {"id": "doc3", "score": 0.75}
        ...     ],
        ...     relevant_doc_ids=["doc1", "doc3", "doc7"]
        ... )
        >>> print(f"Precision@5: {metrics.precision_at_k[5]:.2f}")
        >>> print(f"MRR: {metrics.mrr:.2f}")
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize standard retrieval evaluator.

        Args:
            k_values: List of K values to evaluate (default: [5, 10])

        Raises:
            ValueError: If k_values contains non-positive values
        """
        self.k_values = k_values or [5, 10]

        # Validate K values
        for k in self.k_values:
            if k <= 0:
                raise ValueError(f"All K values must be positive, got {k}")

        logger.debug(f"Initialized retrieval evaluator with K values: {self.k_values}")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        relevant_doc_ids: list[str],
        **kwargs: Any,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality for a single query.

        Args:
            query: Original search query
            retrieved_docs: List of retrieved documents with 'id' and 'score' fields
            relevant_doc_ids: List of IDs for relevant documents (ground truth)
            **kwargs: Unused

        Returns:
            RetrievalMetrics with all computed metrics

        Raises:
            ValueError: If retrieved_docs missing required 'id' field

        Example:
            >>> evaluator = StandardRetrievalEvaluator(k_values=[3, 5])
            >>> metrics = evaluator.evaluate_retrieval(
            ...     query="installation guide",
            ...     retrieved_docs=[
            ...         {"id": "manual_5", "score": 0.92},
            ...         {"id": "api_3", "score": 0.88},
            ...         {"id": "manual_12", "score": 0.75}
            ...     ],
            ...     relevant_doc_ids=["manual_5", "manual_12"]
            ... )
            >>> assert 0.0 <= metrics.precision_at_k[3] <= 1.0
        """
        # Validate retrieved_docs format
        if not all("id" in doc for doc in retrieved_docs):
            raise ValueError("All retrieved_docs must have 'id' field")

        # Extract document IDs in ranked order
        retrieved_ids = [doc["id"] for doc in retrieved_docs]

        logger.debug(
            f"Evaluating retrieval for query: {query[:50]}... "
            f"({len(retrieved_ids)} retrieved, {len(relevant_doc_ids)} relevant)"
        )

        # Calculate Precision@K for each K
        precision_at_k = {}
        for k in self.k_values:
            calc = PrecisionAtK(k=k)
            score = calc.calculate(
                predictions=retrieved_ids,
                ground_truth=relevant_doc_ids,
            )
            precision_at_k[k] = score.value

        # Calculate Recall@K for each K
        recall_at_k = {}
        for k in self.k_values:
            calc = RecallAtK(k=k)
            score = calc.calculate(
                predictions=retrieved_ids,
                ground_truth=relevant_doc_ids,
            )
            recall_at_k[k] = score.value

        # Calculate MRR
        mrr_calc = MRRCalculator()
        mrr_score = mrr_calc.calculate(
            predictions=retrieved_ids,
            ground_truth=relevant_doc_ids,
        )
        mrr = mrr_score.value

        # Calculate NDCG (use largest K value)
        max_k = max(self.k_values)
        ndcg_calc = NDCGCalculator(k=max_k)
        ndcg_score = ndcg_calc.calculate(
            predictions=retrieved_ids,
            ground_truth=relevant_doc_ids,
        )
        ndcg = ndcg_score.value

        # Calculate Hit Rate (use largest K value)
        hit_calc = HitRateCalculator(k=max_k)
        hit_score = hit_calc.calculate(
            predictions=retrieved_ids,
            ground_truth=relevant_doc_ids,
        )
        hit_rate = hit_score.value

        # Create metrics object
        metrics = RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            metadata={
                "query": query,
                "num_retrieved": len(retrieved_ids),
                "num_relevant": len(relevant_doc_ids),
                "k_values": self.k_values,
            },
        )

        logger.debug(
            f"Retrieval metrics: P@{max_k}={precision_at_k[max_k]:.3f}, "
            f"R@{max_k}={recall_at_k[max_k]:.3f}, MRR={mrr:.3f}, NDCG={ndcg:.3f}"
        )

        return metrics


__all__ = ["StandardRetrievalEvaluator"]
