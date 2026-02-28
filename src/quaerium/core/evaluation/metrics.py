"""
Data models for evaluation metrics and results.

This module provides dataclasses for representing evaluation metrics
at different levels: individual scores, retrieval quality, generation
quality, end-to-end performance, and complete evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricScore:
    """
    Individual metric score.

    Represents the result of a single metric calculation (e.g., Precision@5).

    Attributes:
        name: Metric name (e.g., "Precision@5", "MRR", "NDCG")
        value: Metric value (typically 0-1 range)
        metadata: Optional additional information

    Example:
        >>> score = MetricScore(
        ...     name="Precision@5",
        ...     value=0.8,
        ...     metadata={"k": 5, "relevant_found": 4}
        ... )
        >>> print(f"{score.name}: {score.value:.2%}")  # Precision@5: 80.00%
    """

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metric score."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Metric name must be a non-empty string")
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(self.value)}")


@dataclass(frozen=True)
class RetrievalMetrics:
    """
    Retrieval quality metrics.

    Comprehensive metrics for evaluating information retrieval quality,
    including precision, recall, ranking quality, and hit rate.

    Attributes:
        precision_at_k: Precision at different K values {5: 0.8, 10: 0.7}
        recall_at_k: Recall at different K values {5: 0.4, 10: 0.6}
        mrr: Mean Reciprocal Rank (0-1)
        ndcg: Normalized Discounted Cumulative Gain (0-1)
        hit_rate: Whether any relevant doc was found (0 or 1)
        metadata: Additional metadata (e.g., query, num_relevant)

    Example:
        >>> metrics = RetrievalMetrics(
        ...     precision_at_k={5: 0.8, 10: 0.7},
        ...     recall_at_k={5: 0.4, 10: 0.6},
        ...     mrr=0.85,
        ...     ndcg=0.78,
        ...     hit_rate=1.0,
        ...     metadata={"query": "What is RAG?", "num_relevant": 5}
        ... )
        >>> print(f"P@5: {metrics.precision_at_k[5]:.2%}")  # P@5: 80.00%
        >>> print(f"R@10: {metrics.recall_at_k[10]:.2%}")  # R@10: 60.00%
    """

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg: float
    hit_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate retrieval metrics."""
        # Validate precision_at_k
        if not self.precision_at_k:
            raise ValueError("precision_at_k cannot be empty")
        for k, p in self.precision_at_k.items():
            if k <= 0:
                raise ValueError(f"K must be positive, got {k}")
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Precision must be in [0, 1], got {p} for k={k}")

        # Validate recall_at_k
        if not self.recall_at_k:
            raise ValueError("recall_at_k cannot be empty")
        for k, r in self.recall_at_k.items():
            if k <= 0:
                raise ValueError(f"K must be positive, got {k}")
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"Recall must be in [0, 1], got {r} for k={k}")

        # Validate MRR
        if not 0.0 <= self.mrr <= 1.0:
            raise ValueError(f"MRR must be in [0, 1], got {self.mrr}")

        # Validate NDCG
        if not 0.0 <= self.ndcg <= 1.0:
            raise ValueError(f"NDCG must be in [0, 1], got {self.ndcg}")

        # Validate hit_rate
        if self.hit_rate not in (0.0, 1.0):
            raise ValueError(f"Hit rate must be 0 or 1, got {self.hit_rate}")


@dataclass(frozen=True)
class GenerationMetrics:
    """
    Generation quality metrics.

    LLM-based metrics for evaluating generated answer quality,
    including relevance, faithfulness, and hallucination detection.

    Attributes:
        relevance_score: How relevant the answer is to the question (0-1)
        faithfulness_score: How faithful the answer is to the context (0-1)
        hallucination_score: How much the answer hallucinates (0-1, 0=no hallucination)
        answer_similarity: Optional similarity to reference answer (0-1)
        metadata: Additional metadata (e.g., judge_model, reasoning)

    Example:
        >>> metrics = GenerationMetrics(
        ...     relevance_score=0.9,
        ...     faithfulness_score=0.85,
        ...     hallucination_score=0.1,
        ...     answer_similarity=0.75,
        ...     metadata={"judge_model": "gpt-4", "reasoning": "..."}
        ... )
        >>> print(f"Relevance: {metrics.relevance_score:.2%}")  # Relevance: 90.00%
        >>> print(f"Faithfulness: {metrics.faithfulness_score:.2%}")  # Faithfulness: 85.00%
    """

    relevance_score: float
    faithfulness_score: float
    hallucination_score: float
    answer_similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate generation metrics."""
        # Validate relevance
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                f"Relevance score must be in [0, 1], got {self.relevance_score}"
            )

        # Validate faithfulness
        if not 0.0 <= self.faithfulness_score <= 1.0:
            raise ValueError(
                f"Faithfulness score must be in [0, 1], got {self.faithfulness_score}"
            )

        # Validate hallucination
        if not 0.0 <= self.hallucination_score <= 1.0:
            raise ValueError(
                f"Hallucination score must be in [0, 1], got {self.hallucination_score}"
            )

        # Validate answer_similarity if provided
        if self.answer_similarity is not None:
            if not 0.0 <= self.answer_similarity <= 1.0:
                raise ValueError(
                    f"Answer similarity must be in [0, 1], got {self.answer_similarity}"
                )


@dataclass
class EndToEndMetrics:
    """
    End-to-end RAG pipeline metrics.

    Performance metrics for the entire RAG pipeline, including latency,
    token usage, cost estimation, and context utilization.

    Attributes:
        total_latency_ms: Total pipeline latency in milliseconds
        retrieval_latency_ms: Retrieval stage latency in milliseconds
        generation_latency_ms: Generation stage latency in milliseconds
        total_tokens: Total tokens used (input + output)
        cost_usd: Estimated cost in USD
        context_utilization: Context window utilization (0-1)
        metadata: Additional metadata (e.g., model, prompt_tokens, completion_tokens)

    Example:
        >>> metrics = EndToEndMetrics(
        ...     total_latency_ms=1250.5,
        ...     retrieval_latency_ms=450.2,
        ...     generation_latency_ms=800.3,
        ...     total_tokens=1500,
        ...     cost_usd=0.0045,
        ...     context_utilization=0.75,
        ...     metadata={
        ...         "model": "gpt-4",
        ...         "prompt_tokens": 1200,
        ...         "completion_tokens": 300
        ...     }
        ... )
        >>> print(f"Total: {metrics.total_latency_ms:.0f}ms")  # Total: 1251ms
        >>> print(f"Cost: ${metrics.cost_usd:.4f}")  # Cost: $0.0045
    """

    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens: int
    cost_usd: float
    context_utilization: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate end-to-end metrics."""
        # Validate latencies
        if self.total_latency_ms < 0:
            raise ValueError(f"Total latency must be >= 0, got {self.total_latency_ms}")
        if self.retrieval_latency_ms < 0:
            raise ValueError(
                f"Retrieval latency must be >= 0, got {self.retrieval_latency_ms}"
            )
        if self.generation_latency_ms < 0:
            raise ValueError(
                f"Generation latency must be >= 0, got {self.generation_latency_ms}"
            )

        # Validate tokens
        if self.total_tokens < 0:
            raise ValueError(f"Total tokens must be >= 0, got {self.total_tokens}")

        # Validate cost
        if self.cost_usd < 0:
            raise ValueError(f"Cost must be >= 0, got {self.cost_usd}")

        # Validate context utilization
        if not 0.0 <= self.context_utilization <= 1.0:
            raise ValueError(
                f"Context utilization must be in [0, 1], got {self.context_utilization}"
            )


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a RAG query.

    Aggregates all evaluation metrics (retrieval, generation, end-to-end)
    for a single query-answer pair.

    Attributes:
        query: Original query
        retrieval_metrics: Optional retrieval quality metrics
        generation_metrics: Optional generation quality metrics
        end_to_end_metrics: Optional end-to-end performance metrics
        success: Whether the evaluation completed successfully
        error: Optional error message if evaluation failed

    Example:
        >>> result = EvaluationResult(
        ...     query="What is RAG?",
        ...     retrieval_metrics=RetrievalMetrics(...),
        ...     generation_metrics=GenerationMetrics(...),
        ...     end_to_end_metrics=EndToEndMetrics(...),
        ...     success=True
        ... )
        >>> if result.success:
        ...     print(f"Query: {result.query}")
        ...     if result.retrieval_metrics:
        ...         print(f"MRR: {result.retrieval_metrics.mrr:.2f}")
        ...     if result.generation_metrics:
        ...         print(f"Relevance: {result.generation_metrics.relevance_score:.2f}")
    """

    query: str
    retrieval_metrics: RetrievalMetrics | None = None
    generation_metrics: GenerationMetrics | None = None
    end_to_end_metrics: EndToEndMetrics | None = None
    success: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate evaluation result."""
        if not isinstance(self.query, str) or not self.query:
            raise ValueError("Query must be a non-empty string")

        # If success=False, error should be provided
        if not self.success and not self.error:
            raise ValueError("Error message required when success=False")

        # If success=True, at least one metric should be present
        if self.success:
            has_metrics = any(
                [
                    self.retrieval_metrics is not None,
                    self.generation_metrics is not None,
                    self.end_to_end_metrics is not None,
                ]
            )
            if not has_metrics:
                raise ValueError(
                    "At least one metric type required when success=True"
                )


__all__ = [
    "MetricScore",
    "RetrievalMetrics",
    "GenerationMetrics",
    "EndToEndMetrics",
    "EvaluationResult",
]
