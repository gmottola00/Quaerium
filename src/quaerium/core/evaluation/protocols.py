"""
Core protocols for evaluation and metrics.

This module defines the protocol interfaces that all evaluation implementations
must satisfy, enabling seamless switching between different evaluation strategies
and metrics calculators without changing application code.

Design Philosophy:
    - Protocol-based: No inheritance required, duck typing with type safety
    - Minimal interface: Only essential operations
    - Composable: Individual calculators combine into evaluators
    - Observable: Non-invasive pipeline integration
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quaerium.core.evaluation.metrics import (
    EndToEndMetrics,
    EvaluationResult,
    GenerationMetrics,
    MetricScore,
    RetrievalMetrics,
)


@runtime_checkable
class MetricCalculator(Protocol):
    """
    Protocol for individual metric calculation.

    A metric calculator computes a single metric (e.g., Precision@K, MRR)
    from predictions and ground truth data.

    Implementations:
        - PrecisionAtK: Precision at top K results
        - RecallAtK: Recall at top K results
        - MRRCalculator: Mean Reciprocal Rank
        - NDCGCalculator: Normalized Discounted Cumulative Gain

    Example:
        >>> calculator: MetricCalculator = PrecisionAtK(k=5)
        >>> score = calculator.calculate(
        ...     predictions=["doc1", "doc2", "doc3", "doc4", "doc5"],
        ...     ground_truth=["doc1", "doc3", "doc7"]
        ... )
        >>> print(f"{score.name}: {score.value:.2f}")  # Precision@5: 0.40
    """

    @property
    def name(self) -> str:
        """
        Human-readable name of this metric.

        Returns:
            Metric name (e.g., "Precision@5", "MRR", "NDCG")
        """
        ...

    def calculate(
        self,
        predictions: Any,
        ground_truth: Any,
        **kwargs: Any,
    ) -> MetricScore:
        """
        Calculate metric score from predictions and ground truth.

        Args:
            predictions: Model predictions (format varies by metric)
            ground_truth: Ground truth labels/answers
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricScore with name, value, and optional metadata

        Raises:
            ValueError: If predictions/ground_truth format is invalid

        Example:
            >>> score = calculator.calculate(
            ...     predictions=["doc1", "doc2"],
            ...     ground_truth=["doc1", "doc3"]
            ... )
            >>> assert 0.0 <= score.value <= 1.0
        """
        ...


@runtime_checkable
class RetrievalEvaluator(Protocol):
    """
    Protocol for evaluating retrieval quality.

    Evaluates how well a retrieval system finds relevant documents,
    measuring precision, recall, ranking quality, and hit rate.

    Implementations:
        - StandardRetrievalEvaluator: Computes all standard retrieval metrics

    Example:
        >>> evaluator: RetrievalEvaluator = StandardRetrievalEvaluator(k_values=[5, 10])
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
            **kwargs: Additional evaluator-specific parameters

        Returns:
            RetrievalMetrics with precision, recall, MRR, NDCG, and hit rate

        Raises:
            ValueError: If retrieved_docs missing required fields

        Example:
            >>> metrics = evaluator.evaluate_retrieval(
            ...     query="installation guide",
            ...     retrieved_docs=[
            ...         {"id": "doc_manual_5", "score": 0.92},
            ...         {"id": "doc_api_3", "score": 0.88}
            ...     ],
            ...     relevant_doc_ids=["doc_manual_5", "doc_manual_12"]
            ... )
            >>> assert 0.0 <= metrics.precision_at_k[5] <= 1.0
        """
        ...


@runtime_checkable
class GenerationEvaluator(Protocol):
    """
    Protocol for evaluating generation quality.

    Evaluates the quality of generated answers using LLM-based judges
    to measure relevance, faithfulness, and hallucination detection.

    Implementations:
        - StandardGenerationEvaluator: Computes all standard generation metrics

    Example:
        >>> judge = OpenAIJudge(llm=my_llm)
        >>> evaluator: GenerationEvaluator = StandardGenerationEvaluator(judge)
        >>> metrics = evaluator.evaluate_answer(
        ...     question="What is RAG?",
        ...     generated_answer="RAG combines retrieval with generation.",
        ...     context=["RAG is retrieval-augmented generation..."],
        ...     reference_answer="RAG is a technique that combines retrieval..."
        ... )
        >>> print(f"Relevance: {metrics.relevance_score:.2f}")
        >>> print(f"Faithfulness: {metrics.faithfulness_score:.2f}")
    """

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        context: list[str],
        reference_answer: str | None = None,
        **kwargs: Any,
    ) -> GenerationMetrics:
        """
        Evaluate the quality of a generated answer.

        Args:
            question: Original question
            generated_answer: Answer generated by the RAG system
            context: Context chunks used for generation
            reference_answer: Optional ground truth answer
            **kwargs: Additional evaluator-specific parameters

        Returns:
            GenerationMetrics with relevance, faithfulness, and hallucination scores

        Example:
            >>> metrics = evaluator.evaluate_answer(
            ...     question="How do I install Python?",
            ...     generated_answer="Download from python.org and run installer.",
            ...     context=["Python installation: Visit python.org..."],
            ...     reference_answer="Install from python.org"
            ... )
            >>> assert 0.0 <= metrics.relevance_score <= 1.0
            >>> assert 0.0 <= metrics.faithfulness_score <= 1.0
        """
        ...


@runtime_checkable
class LLMJudge(Protocol):
    """
    Protocol for LLM-as-judge evaluation.

    Uses an LLM to evaluate subjective qualities like relevance,
    faithfulness, and answer quality that are difficult to measure
    with traditional metrics.

    Implementations:
        - OpenAIJudge: Uses OpenAI models for evaluation

    Example:
        >>> judge: LLMJudge = OpenAIJudge(llm=my_llm)
        >>> score = judge.judge(
        ...     prompt="Rate the relevance of this answer on a scale of 0-10: ..."
        ... )
        >>> print(f"Score: {score:.2f}")  # Normalized to 0-1
        >>>
        >>> score, reasoning = judge.judge_with_reasoning(
        ...     prompt="Evaluate this answer and explain your reasoning..."
        ... )
        >>> print(f"Score: {score:.2f}, Reasoning: {reasoning}")
    """

    def judge(self, prompt: str, **kwargs: Any) -> float:
        """
        Get a numerical judgment from the LLM.

        Args:
            prompt: Evaluation prompt for the LLM
            **kwargs: Additional parameters (temperature, model, etc.)

        Returns:
            Score normalized to 0-1 range (0=worst, 1=best)

        Raises:
            ValueError: If LLM response cannot be parsed as a score

        Example:
            >>> prompt = '''
            ... Rate the relevance of this answer on a scale of 0-10:
            ... Question: What is Python?
            ... Answer: Python is a programming language.
            ... '''
            >>> score = judge.judge(prompt)
            >>> assert 0.0 <= score <= 1.0
        """
        ...

    def judge_with_reasoning(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[float, str]:
        """
        Get a judgment with explanation from the LLM.

        Args:
            prompt: Evaluation prompt for the LLM
            **kwargs: Additional parameters (temperature, model, etc.)

        Returns:
            Tuple of (score, reasoning):
                - score: Numerical score normalized to 0-1
                - reasoning: LLM's explanation for the score

        Example:
            >>> prompt = "Evaluate this answer: ..."
            >>> score, reasoning = judge.judge_with_reasoning(prompt)
            >>> print(f"Score: {score:.2f}")
            >>> print(f"Reasoning: {reasoning}")
        """
        ...


@runtime_checkable
class PipelineObserver(Protocol):
    """
    Protocol for observing RAG pipeline execution.

    Observers implement the observer pattern to monitor pipeline stages
    without modifying pipeline code. They track timing, token usage,
    and collect data for evaluation.

    Implementations:
        - MetricsObserver: Collects end-to-end metrics (latency, cost, tokens)

    Example:
        >>> observer: PipelineObserver = MetricsObserver()
        >>> pipeline = RagPipeline(..., observers=[observer])
        >>> response = pipeline.run("What is RAG?")
        >>> result = observer.get_results()
        >>> print(f"Latency: {result.end_to_end_metrics.total_latency_ms}ms")
        >>> print(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")
    """

    def on_query_rewrite(
        self,
        original: str,
        rewritten: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after query rewriting.

        Args:
            original: Original query
            rewritten: Rewritten query
            metadata: Additional metadata (e.g., rewrite method)
        """
        ...

    def on_retrieval(
        self,
        query: str,
        results: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after retrieval.

        Args:
            query: Search query (possibly rewritten)
            results: Retrieved documents with scores
            metadata: Additional metadata (e.g., search method, filters)
        """
        ...

    def on_reranking(
        self,
        query: str,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after reranking.

        Args:
            query: Search query
            before: Documents before reranking
            after: Documents after reranking
            metadata: Additional metadata (e.g., reranker model)
        """
        ...

    def on_context_assembly(
        self,
        chunks: list[dict[str, Any]],
        token_count: int,
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after context assembly.

        Args:
            chunks: Selected context chunks
            token_count: Total tokens in assembled context
            metadata: Additional metadata (e.g., max_tokens, assembly method)
        """
        ...

    def on_generation(
        self,
        question: str,
        answer: str,
        context: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after answer generation.

        Args:
            question: Original question
            answer: Generated answer
            context: Context used for generation
            metadata: Additional metadata (e.g., model, tokens, cost)
        """
        ...

    def get_results(self) -> EvaluationResult:
        """
        Get collected evaluation results.

        Returns:
            EvaluationResult with all collected metrics

        Example:
            >>> result = observer.get_results()
            >>> if result.end_to_end_metrics:
            ...     print(f"Total latency: {result.end_to_end_metrics.total_latency_ms}ms")
            ...     print(f"Total tokens: {result.end_to_end_metrics.total_tokens}")
        """
        ...


__all__ = [
    "MetricCalculator",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "LLMJudge",
    "PipelineObserver",
]
