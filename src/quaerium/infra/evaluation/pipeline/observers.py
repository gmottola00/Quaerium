"""
Pipeline observers for metrics collection.

Implements the observer pattern for non-invasive pipeline monitoring.
"""

from __future__ import annotations

import logging
from typing import Any

from quaerium.core.evaluation.metrics import (
    EndToEndMetrics,
    EvaluationResult,
)
from quaerium.infra.evaluation.pipeline.trackers import (
    LatencyTracker,
    TokenTracker,
)

logger = logging.getLogger(__name__)


class MetricsObserver:
    """
    Observer for collecting end-to-end pipeline metrics.

    Tracks latency, token usage, cost, and context utilization
    throughout RAG pipeline execution.

    Example:
        >>> observer = MetricsObserver()
        >>> pipeline = RagPipeline(..., observers=[observer])
        >>> response = pipeline.run("What is RAG?")
        >>> result = observer.get_results()
        >>> print(f"Latency: {result.end_to_end_metrics.total_latency_ms}ms")
        >>> print(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")
    """

    def __init__(self):
        """Initialize metrics observer."""
        self.latency_tracker = LatencyTracker()
        self.token_tracker = TokenTracker()
        self.query = ""
        self.context_token_count = 0
        self.max_context_tokens = 8192  # Default, can be updated
        self._retrieval_count = 0
        self._generation_metadata: dict[str, Any] = {}

        logger.debug("Initialized metrics observer")

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
            metadata: Additional metadata
        """
        self.query = original
        self.latency_tracker.start("total")
        logger.debug(f"Query rewrite: '{original}' -> '{rewritten}'")

    def on_retrieval(
        self,
        query: str,
        results: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """
        Called after retrieval.

        Args:
            query: Search query
            results: Retrieved documents
            metadata: Additional metadata
        """
        if self._retrieval_count == 0:
            self.latency_tracker.start("retrieval")

        self._retrieval_count += 1

        logger.debug(f"Retrieval: {len(results)} results for query: {query[:50]}...")

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
            metadata: Additional metadata
        """
        if self._retrieval_count == 1:
            self.latency_tracker.end("retrieval")

        logger.debug(
            f"Reranking: {len(before)} docs -> {len(after)} docs for query: {query[:50]}..."
        )

        # Track LLM usage if available in metadata
        if "usage" in metadata:
            usage = metadata["usage"]
            model = metadata.get("model")
            self.token_tracker.add_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                model=model,
            )

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
            metadata: Additional metadata
        """
        self.context_token_count = token_count
        if "max_tokens" in metadata:
            self.max_context_tokens = metadata["max_tokens"]

        logger.debug(
            f"Context assembly: {len(chunks)} chunks, {token_count} tokens "
            f"({token_count}/{self.max_context_tokens} utilization)"
        )

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
            metadata: Additional metadata (should include 'usage' and 'model')
        """
        self.latency_tracker.start("generation")

        # Track LLM usage
        if "usage" in metadata:
            usage = metadata["usage"]
            model = metadata.get("model")
            self.token_tracker.add_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                model=model,
            )
            self._generation_metadata = metadata

        self.latency_tracker.end("generation")
        self.latency_tracker.end("total")

        logger.debug(
            f"Generation: {len(answer)} chars for question: {question[:50]}..."
        )

    def get_results(self) -> EvaluationResult:
        """
        Get collected evaluation results.

        Returns:
            EvaluationResult with end-to-end metrics

        Example:
            >>> result = observer.get_results()
            >>> print(f"Latency: {result.end_to_end_metrics.total_latency_ms:.0f}ms")
            >>> print(f"Tokens: {result.end_to_end_metrics.total_tokens}")
        """
        try:
            # If no query was set, we haven't observed a pipeline run
            if not self.query:
                return EvaluationResult(
                    query="<not set>",
                    success=False,
                    error="No pipeline execution observed",
                )

            latencies = self.latency_tracker.get_all_latencies()

            # Calculate context utilization
            context_utilization = 0.0
            if self.max_context_tokens > 0:
                context_utilization = min(
                    self.context_token_count / self.max_context_tokens, 1.0
                )

            # Build end-to-end metrics
            end_to_end_metrics = EndToEndMetrics(
                total_latency_ms=latencies.get("total", 0.0),
                retrieval_latency_ms=latencies.get("retrieval", 0.0),
                generation_latency_ms=latencies.get("generation", 0.0),
                total_tokens=self.token_tracker.total_tokens,
                cost_usd=self.token_tracker.estimated_cost_usd,
                context_utilization=context_utilization,
                metadata={
                    "prompt_tokens": self.token_tracker.prompt_tokens,
                    "completion_tokens": self.token_tracker.completion_tokens,
                    "context_tokens": self.context_token_count,
                    "max_context_tokens": self.max_context_tokens,
                    **self._generation_metadata,
                },
            )

            result = EvaluationResult(
                query=self.query,
                end_to_end_metrics=end_to_end_metrics,
                success=True,
            )

            logger.debug(
                f"Metrics collected: {latencies.get('total', 0):.0f}ms total, "
                f"{self.token_tracker.total_tokens} tokens, "
                f"${self.token_tracker.estimated_cost_usd:.4f} cost"
            )

            return result

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}", exc_info=True)
            return EvaluationResult(
                query=self.query,
                success=False,
                error=str(e),
            )


__all__ = ["MetricsObserver"]
