"""
Tracking utilities for pipeline metrics.

Provides helpers for tracking latency, tokens, and cost.
"""

from __future__ import annotations

import time
from typing import Any


class LatencyTracker:
    """
    Track latency for pipeline stages.

    Example:
        >>> tracker = LatencyTracker()
        >>> tracker.start("retrieval")
        >>> # ... do retrieval ...
        >>> tracker.end("retrieval")
        >>> print(f"Retrieval took {tracker.get_latency('retrieval')}ms")
    """

    def __init__(self):
        """Initialize latency tracker."""
        self._start_times: dict[str, float] = {}
        self._latencies: dict[str, float] = {}

    def start(self, stage: str) -> None:
        """
        Start timing a stage.

        Args:
            stage: Stage name (e.g., "retrieval", "generation")
        """
        self._start_times[stage] = time.time()

    def end(self, stage: str) -> None:
        """
        End timing a stage and record latency.

        Args:
            stage: Stage name

        Raises:
            KeyError: If stage was not started
        """
        if stage not in self._start_times:
            raise KeyError(f"Stage '{stage}' was not started")

        elapsed = time.time() - self._start_times[stage]
        self._latencies[stage] = elapsed * 1000  # Convert to milliseconds
        del self._start_times[stage]

    def get_latency(self, stage: str) -> float:
        """
        Get latency for a stage in milliseconds.

        Args:
            stage: Stage name

        Returns:
            Latency in milliseconds

        Raises:
            KeyError: If stage has no recorded latency
        """
        if stage not in self._latencies:
            raise KeyError(f"No latency recorded for stage '{stage}'")
        return self._latencies[stage]

    def get_all_latencies(self) -> dict[str, float]:
        """Get all recorded latencies."""
        return self._latencies.copy()


class TokenTracker:
    """
    Track token usage and cost estimation.

    Example:
        >>> tracker = TokenTracker()
        >>> tracker.add_usage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        >>> print(f"Total tokens: {tracker.total_tokens}")
        >>> print(f"Cost: ${tracker.estimated_cost_usd:.4f}")
    """

    # Pricing per 1M tokens (as of 2024, approximate)
    PRICING = {
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4o": {"prompt": 5.0, "completion": 15.0},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        # Fallback for unknown models
        "default": {"prompt": 10.0, "completion": 30.0},
    }

    def __init__(self):
        """Initialize token tracker."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost_usd = 0.0

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
    ) -> None:
        """
        Add token usage and update cost estimate.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name for cost estimation
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        if model:
            self.total_cost_usd += self._estimate_cost(
                prompt_tokens, completion_tokens, model
            )

    def _estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """Estimate cost for token usage."""
        # Find pricing for model (or use default)
        pricing = self.PRICING.get(model, self.PRICING["default"])

        # Calculate cost (pricing is per 1M tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost

    @property
    def total_tokens(self) -> int:
        """Get total tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Get estimated total cost in USD."""
        return self.total_cost_usd


__all__ = ["LatencyTracker", "TokenTracker"]
