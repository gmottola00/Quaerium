"""
Pipeline evaluation and observation.

This module provides observers for monitoring RAG pipeline execution
and collecting end-to-end metrics.
"""

from quaerium.infra.evaluation.pipeline.observers import MetricsObserver

__all__ = ["MetricsObserver"]
