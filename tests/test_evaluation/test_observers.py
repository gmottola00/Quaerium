"""
Tests for pipeline observers and metrics tracking.

Tests the observer pattern integration with RAG pipeline
and end-to-end metrics collection.
"""

import pytest

from quaerium.core.evaluation.protocols import PipelineObserver
from quaerium.infra.evaluation.pipeline import MetricsObserver
from quaerium.infra.evaluation.pipeline.trackers import (
    LatencyTracker,
    TokenTracker,
)


class TestLatencyTracker:
    """Test latency tracking."""

    def test_track_single_stage(self):
        """Test tracking single stage latency."""
        import time

        tracker = LatencyTracker()
        tracker.start("test_stage")
        time.sleep(0.01)  # Sleep 10ms
        tracker.end("test_stage")

        latency = tracker.get_latency("test_stage")
        assert latency >= 10  # At least 10ms
        assert latency < 50  # Should be less than 50ms

    def test_track_multiple_stages(self):
        """Test tracking multiple stages."""
        tracker = LatencyTracker()

        tracker.start("stage1")
        tracker.end("stage1")

        tracker.start("stage2")
        tracker.end("stage2")

        latencies = tracker.get_all_latencies()
        assert "stage1" in latencies
        assert "stage2" in latencies
        assert latencies["stage1"] >= 0
        assert latencies["stage2"] >= 0

    def test_end_without_start_raises_error(self):
        """Test that ending unstarted stage raises error."""
        tracker = LatencyTracker()

        with pytest.raises(KeyError, match="was not started"):
            tracker.end("nonexistent_stage")

    def test_get_latency_without_end_raises_error(self):
        """Test that getting latency for incomplete stage raises error."""
        tracker = LatencyTracker()
        tracker.start("test_stage")

        with pytest.raises(KeyError, match="No latency recorded"):
            tracker.get_latency("test_stage")


class TestTokenTracker:
    """Test token usage and cost tracking."""

    def test_track_tokens(self):
        """Test basic token tracking."""
        tracker = TokenTracker()

        tracker.add_usage(prompt_tokens=100, completion_tokens=50)

        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150

    def test_track_multiple_usages(self):
        """Test accumulating multiple usages."""
        tracker = TokenTracker()

        tracker.add_usage(prompt_tokens=100, completion_tokens=50)
        tracker.add_usage(prompt_tokens=200, completion_tokens=75)

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 125
        assert tracker.total_tokens == 425

    def test_cost_estimation_gpt4(self):
        """Test cost estimation for GPT-4."""
        tracker = TokenTracker()

        # 1000 prompt tokens + 500 completion tokens
        tracker.add_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4",
        )

        # Expected: (1000/1M * $30) + (500/1M * $60) = $0.03 + $0.03 = $0.06
        expected_cost = 0.06
        assert abs(tracker.estimated_cost_usd - expected_cost) < 0.001

    def test_cost_estimation_gpt35(self):
        """Test cost estimation for GPT-3.5."""
        tracker = TokenTracker()

        tracker.add_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo",
        )

        # Expected: (1000/1M * $0.5) + (500/1M * $1.5) = $0.0005 + $0.00075 = $0.00125
        expected_cost = 0.00125
        assert abs(tracker.estimated_cost_usd - expected_cost) < 0.0001

    def test_cost_estimation_unknown_model(self):
        """Test cost estimation falls back to default for unknown model."""
        tracker = TokenTracker()

        tracker.add_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="unknown-model",
        )

        # Should use default pricing
        assert tracker.estimated_cost_usd > 0

    def test_no_cost_without_model(self):
        """Test that no cost is estimated without model name."""
        tracker = TokenTracker()

        tracker.add_usage(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        assert tracker.estimated_cost_usd == 0.0


class TestMetricsObserver:
    """Test metrics observer."""

    def test_observer_initialization(self):
        """Test observer initializes correctly."""
        observer = MetricsObserver()

        assert observer.query == ""
        assert observer.context_token_count == 0
        assert observer.token_tracker.total_tokens == 0

    def test_query_rewrite_callback(self):
        """Test query rewrite callback."""
        observer = MetricsObserver()

        observer.on_query_rewrite(
            original="test query",
            rewritten="test query expanded",
            metadata={},
        )

        assert observer.query == "test query"

    def test_retrieval_callback(self):
        """Test retrieval callback."""
        observer = MetricsObserver()

        observer.on_query_rewrite("test", "test", {})
        observer.on_retrieval(
            query="test",
            results=[{"id": "doc1"}, {"id": "doc2"}],
            metadata={},
        )

        # Should have started tracking
        assert observer._retrieval_count == 1

    def test_reranking_callback_with_usage(self):
        """Test reranking callback tracks LLM usage."""
        observer = MetricsObserver()

        observer.on_query_rewrite("test", "test", {})
        observer.on_retrieval("test", [], {})

        observer.on_reranking(
            query="test",
            before=[{"id": "doc1"}, {"id": "doc2"}],
            after=[{"id": "doc1"}],
            metadata={
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "model": "gpt-4",
            },
        )

        assert observer.token_tracker.total_tokens == 150

    def test_context_assembly_callback(self):
        """Test context assembly callback."""
        observer = MetricsObserver()

        observer.on_context_assembly(
            chunks=[{"text": "chunk1"}, {"text": "chunk2"}],
            token_count=500,
            metadata={"max_tokens": 8192},
        )

        assert observer.context_token_count == 500
        assert observer.max_context_tokens == 8192

    def test_generation_callback_with_usage(self):
        """Test generation callback tracks LLM usage."""
        observer = MetricsObserver()

        observer.on_query_rewrite("test", "test", {})
        observer.on_generation(
            question="test question",
            answer="test answer",
            context=["context1", "context2"],
            metadata={
                "usage": {"prompt_tokens": 200, "completion_tokens": 100},
                "model": "gpt-4",
            },
        )

        assert observer.token_tracker.total_tokens == 300

    def test_complete_pipeline_flow(self):
        """Test complete pipeline observation flow."""
        observer = MetricsObserver()

        # Simulate complete pipeline
        observer.on_query_rewrite("What is Python?", "What is Python programming?", {})
        observer.on_retrieval("What is Python programming?", [{"id": "doc1"}], {})
        observer.on_reranking(
            "What is Python?",
            [{"id": "doc1"}],
            [{"id": "doc1"}],
            {"usage": {"prompt_tokens": 50, "completion_tokens": 10}, "model": "gpt-4"},
        )
        observer.on_context_assembly([{"text": "Python is a language"}], 300, {"max_tokens": 8192})
        observer.on_generation(
            "What is Python?",
            "Python is a programming language.",
            ["Python is a language"],
            {"usage": {"prompt_tokens": 400, "completion_tokens": 50}, "model": "gpt-4"},
        )

        # Get results
        result = observer.get_results()

        assert result.success is True
        assert result.query == "What is Python?"
        assert result.end_to_end_metrics is not None

        metrics = result.end_to_end_metrics
        assert metrics.total_latency_ms > 0
        assert metrics.retrieval_latency_ms >= 0
        assert metrics.generation_latency_ms > 0
        assert metrics.total_tokens == 510  # 50+10+400+50
        assert metrics.cost_usd > 0
        assert 0 <= metrics.context_utilization <= 1

    def test_context_utilization_calculation(self):
        """Test context utilization is calculated correctly."""
        observer = MetricsObserver()

        observer.on_query_rewrite("test", "test", {})
        observer.on_context_assembly(
            chunks=[],
            token_count=4096,  # Half of default 8192
            metadata={},
        )
        observer.on_generation("test", "answer", [], {})

        result = observer.get_results()
        assert result.end_to_end_metrics.context_utilization == 0.5

    def test_protocol_compliance(self):
        """Test that MetricsObserver satisfies protocol."""
        observer = MetricsObserver()
        assert isinstance(observer, PipelineObserver)

    def test_error_handling_in_get_results(self):
        """Test that get_results handles errors gracefully."""
        observer = MetricsObserver()

        # Don't call any callbacks, just try to get results
        # Should detect that no pipeline run was observed
        result = observer.get_results()

        # Should return failure since no execution was observed
        assert result.success is False
        assert result.error == "No pipeline execution observed"
        assert result.query == "<not set>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
