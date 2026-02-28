"""
Evaluation and metrics protocols for Quaerium.

This module defines the protocol interfaces for evaluating RAG system quality,
including retrieval precision, generation quality, and end-to-end performance.

Design Philosophy:
    - Protocol-based: No inheritance required, duck typing with type safety
    - Composable: Individual metrics can be combined into evaluators
    - Non-invasive: Observer pattern for pipeline integration
    - Production-ready: Comprehensive metrics for monitoring and optimization

Key Protocols:
    - MetricCalculator: Individual metric computation (e.g., Precision@K)
    - RetrievalEvaluator: Retrieval quality evaluation
    - GenerationEvaluator: Generation quality evaluation (LLM-as-judge)
    - LLMJudge: LLM-based subjective evaluation
    - PipelineObserver: Non-invasive pipeline monitoring

Data Models:
    - MetricScore: Individual metric result
    - RetrievalMetrics: Retrieval quality metrics
    - GenerationMetrics: Generation quality metrics
    - EndToEndMetrics: Latency, cost, and performance metrics
    - EvaluationResult: Complete evaluation result
"""

from quaerium.core.evaluation.dataset import (
    DatasetMetadata,
    EvaluationDataset,
    EvaluationExample,
)
from quaerium.core.evaluation.metrics import (
    EndToEndMetrics,
    EvaluationResult,
    GenerationMetrics,
    MetricScore,
    RetrievalMetrics,
)
from quaerium.core.evaluation.protocols import (
    GenerationEvaluator,
    LLMJudge,
    MetricCalculator,
    PipelineObserver,
    RetrievalEvaluator,
)

__all__ = [
    # Protocols
    "MetricCalculator",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "LLMJudge",
    "PipelineObserver",
    # Data Models
    "MetricScore",
    "RetrievalMetrics",
    "GenerationMetrics",
    "EndToEndMetrics",
    "EvaluationResult",
    # Dataset Models
    "EvaluationExample",
    "DatasetMetadata",
    "EvaluationDataset",
]
