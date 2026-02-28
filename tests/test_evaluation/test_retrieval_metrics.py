"""
Tests for retrieval evaluation metrics.

Tests individual metric calculators (Precision@K, Recall@K, MRR, NDCG)
and the composite StandardRetrievalEvaluator.
"""

import pytest

from quaerium.core.evaluation.protocols import RetrievalEvaluator
from quaerium.infra.evaluation.retrieval import (
    HitRateCalculator,
    MRRCalculator,
    NDCGCalculator,
    PrecisionAtK,
    RecallAtK,
    StandardRetrievalEvaluator,
)


class TestPrecisionAtK:
    """Test Precision@K calculator."""

    def test_perfect_precision(self):
        """Test with all retrieved docs relevant."""
        calc = PrecisionAtK(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4", "d5"],
            ground_truth=["d1", "d2", "d3", "d4", "d5"],
        )
        assert score.name == "Precision@5"
        assert score.value == 1.0
        assert score.metadata["relevant_found"] == 5

    def test_half_precision(self):
        """Test with half of retrieved docs relevant."""
        calc = PrecisionAtK(k=4)
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4"],
            ground_truth=["d1", "d3", "d7", "d9"],
        )
        assert score.value == 0.5  # 2 out of 4
        assert score.metadata["relevant_found"] == 2

    def test_zero_precision(self):
        """Test with no relevant docs retrieved."""
        calc = PrecisionAtK(k=3)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d7", "d8", "d9"],
        )
        assert score.value == 0.0

    def test_empty_predictions(self):
        """Test with empty predictions list."""
        calc = PrecisionAtK(k=5)
        score = calc.calculate(
            predictions=[],
            ground_truth=["d1", "d2"],
        )
        assert score.value == 0.0

    def test_fewer_predictions_than_k(self):
        """Test when fewer predictions than K."""
        calc = PrecisionAtK(k=10)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d1", "d2"],
        )
        # Only 3 predictions, so precision is 2/3
        assert score.value == pytest.approx(2 / 3)

    def test_invalid_k(self):
        """Test that invalid K raises error."""
        with pytest.raises(ValueError, match="K must be positive"):
            PrecisionAtK(k=0)
        with pytest.raises(ValueError, match="K must be positive"):
            PrecisionAtK(k=-5)


class TestRecallAtK:
    """Test Recall@K calculator."""

    def test_perfect_recall(self):
        """Test with all relevant docs retrieved."""
        calc = RecallAtK(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4", "d5"],
            ground_truth=["d1", "d2", "d3"],
        )
        assert score.name == "Recall@5"
        assert score.value == 1.0
        assert score.metadata["relevant_found"] == 3
        assert score.metadata["total_relevant"] == 3

    def test_partial_recall(self):
        """Test with only some relevant docs retrieved."""
        calc = RecallAtK(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4", "d5"],
            ground_truth=["d1", "d3", "d7", "d9", "d10"],
        )
        assert score.value == 0.4  # 2 out of 5
        assert score.metadata["relevant_found"] == 2
        assert score.metadata["total_relevant"] == 5

    def test_zero_recall(self):
        """Test with no relevant docs retrieved."""
        calc = RecallAtK(k=3)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d7", "d8", "d9"],
        )
        assert score.value == 0.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        calc = RecallAtK(k=5)
        score = calc.calculate(
            predictions=[],
            ground_truth=["d1", "d2"],
        )
        assert score.value == 0.0

    def test_empty_ground_truth(self):
        """Test with empty ground truth."""
        calc = RecallAtK(k=5)
        score = calc.calculate(
            predictions=["d1", "d2"],
            ground_truth=[],
        )
        assert score.value == 0.0


class TestMRRCalculator:
    """Test MRR (Mean Reciprocal Rank) calculator."""

    def test_first_position(self):
        """Test with relevant doc in first position."""
        calc = MRRCalculator()
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d1", "d7"],
        )
        assert score.name == "MRR"
        assert score.value == 1.0  # 1/1
        assert score.metadata["first_relevant_rank"] == 1

    def test_second_position(self):
        """Test with relevant doc in second position."""
        calc = MRRCalculator()
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d2", "d7"],
        )
        assert score.value == 0.5  # 1/2
        assert score.metadata["first_relevant_rank"] == 2

    def test_third_position(self):
        """Test with relevant doc in third position."""
        calc = MRRCalculator()
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4"],
            ground_truth=["d3", "d7"],
        )
        assert score.value == pytest.approx(1 / 3)
        assert score.metadata["first_relevant_rank"] == 3

    def test_no_relevant_found(self):
        """Test with no relevant docs in predictions."""
        calc = MRRCalculator()
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d7", "d8"],
        )
        assert score.value == 0.0
        assert score.metadata["first_relevant_rank"] is None

    def test_empty_predictions(self):
        """Test with empty predictions."""
        calc = MRRCalculator()
        score = calc.calculate(
            predictions=[],
            ground_truth=["d1"],
        )
        assert score.value == 0.0


class TestNDCGCalculator:
    """Test NDCG calculator."""

    def test_perfect_ranking(self):
        """Test with perfect ranking (all relevant docs first)."""
        calc = NDCGCalculator(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4", "d5"],
            ground_truth=["d1", "d2", "d3"],
        )
        assert score.name == "NDCG@5"
        assert score.value == 1.0

    def test_worst_ranking(self):
        """Test with worst ranking (relevant docs last)."""
        calc = NDCGCalculator(k=5)
        score = calc.calculate(
            predictions=["d4", "d5", "d6", "d1", "d2"],
            ground_truth=["d1", "d2", "d3"],
        )
        # NDCG should be less than 1 but greater than 0
        assert 0.0 < score.value < 1.0

    def test_no_relevant_docs(self):
        """Test with no relevant docs retrieved."""
        calc = NDCGCalculator(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d7", "d8"],
        )
        assert score.value == 0.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        calc = NDCGCalculator(k=5)
        score = calc.calculate(
            predictions=[],
            ground_truth=["d1"],
        )
        assert score.value == 0.0

    def test_partial_overlap(self):
        """Test with partial overlap between predictions and ground truth."""
        calc = NDCGCalculator(k=3)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d1", "d5"],
        )
        # Should have some positive NDCG
        assert 0.0 < score.value < 1.0
        assert "dcg" in score.metadata
        assert "idcg" in score.metadata


class TestHitRateCalculator:
    """Test Hit Rate calculator."""

    def test_hit_found(self):
        """Test when at least one relevant doc is found."""
        calc = HitRateCalculator(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d2", "d7"],
        )
        assert score.name == "Hit@5"
        assert score.value == 1.0

    def test_hit_not_found(self):
        """Test when no relevant docs are found."""
        calc = HitRateCalculator(k=5)
        score = calc.calculate(
            predictions=["d1", "d2", "d3"],
            ground_truth=["d7", "d8"],
        )
        assert score.value == 0.0

    def test_hit_without_k(self):
        """Test hit rate without K limit."""
        calc = HitRateCalculator()
        score = calc.calculate(
            predictions=["d1", "d2", "d3", "d4", "d5"],
            ground_truth=["d5"],
        )
        assert score.name == "Hit Rate"
        assert score.value == 1.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        calc = HitRateCalculator(k=5)
        score = calc.calculate(
            predictions=[],
            ground_truth=["d1"],
        )
        assert score.value == 0.0


class TestStandardRetrievalEvaluator:
    """Test StandardRetrievalEvaluator."""

    def test_basic_evaluation(self):
        """Test basic retrieval evaluation."""
        evaluator = StandardRetrievalEvaluator(k_values=[3, 5])

        retrieved_docs = [
            {"id": "d1", "score": 0.95},
            {"id": "d2", "score": 0.87},
            {"id": "d3", "score": 0.75},
            {"id": "d4", "score": 0.60},
            {"id": "d5", "score": 0.55},
        ]
        relevant_ids = ["d1", "d3", "d7"]

        metrics = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=relevant_ids,
        )

        # Check all metrics are present
        assert 3 in metrics.precision_at_k
        assert 5 in metrics.precision_at_k
        assert 3 in metrics.recall_at_k
        assert 5 in metrics.recall_at_k
        assert isinstance(metrics.mrr, float)
        assert isinstance(metrics.ndcg, float)
        assert metrics.hit_rate in (0.0, 1.0)

        # Check specific values
        assert metrics.precision_at_k[3] == pytest.approx(2 / 3)  # 2 out of 3
        assert metrics.precision_at_k[5] == pytest.approx(2 / 5)  # 2 out of 5
        assert metrics.recall_at_k[3] == pytest.approx(2 / 3)  # 2 out of 3 relevant
        assert metrics.recall_at_k[5] == pytest.approx(2 / 3)  # 2 out of 3 relevant
        assert metrics.mrr == 1.0  # First doc is relevant
        assert metrics.hit_rate == 1.0  # At least one relevant found

        # Check metadata
        assert metrics.metadata["query"] == "test query"
        assert metrics.metadata["num_retrieved"] == 5
        assert metrics.metadata["num_relevant"] == 3

    def test_no_relevant_docs_retrieved(self):
        """Test when no relevant docs are retrieved."""
        evaluator = StandardRetrievalEvaluator(k_values=[5])

        retrieved_docs = [
            {"id": "d1", "score": 0.95},
            {"id": "d2", "score": 0.87},
        ]
        relevant_ids = ["d7", "d8", "d9"]

        metrics = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=relevant_ids,
        )

        # All metrics should be 0
        assert metrics.precision_at_k[5] == 0.0
        assert metrics.recall_at_k[5] == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0
        assert metrics.hit_rate == 0.0

    def test_perfect_retrieval(self):
        """Test with perfect retrieval (all relevant docs first)."""
        evaluator = StandardRetrievalEvaluator(k_values=[3])

        retrieved_docs = [
            {"id": "d1", "score": 0.95},
            {"id": "d2", "score": 0.87},
            {"id": "d3", "score": 0.75},
        ]
        relevant_ids = ["d1", "d2", "d3"]

        metrics = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=relevant_ids,
        )

        # All metrics should be perfect
        assert metrics.precision_at_k[3] == 1.0
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg == 1.0
        assert metrics.hit_rate == 1.0

    def test_missing_id_field(self):
        """Test that missing 'id' field raises error."""
        evaluator = StandardRetrievalEvaluator()

        retrieved_docs = [
            {"score": 0.95},  # Missing 'id'
            {"id": "d2", "score": 0.87},
        ]

        with pytest.raises(ValueError, match="must have 'id' field"):
            evaluator.evaluate_retrieval(
                query="test",
                retrieved_docs=retrieved_docs,
                relevant_doc_ids=["d1"],
            )

    def test_default_k_values(self):
        """Test default K values."""
        evaluator = StandardRetrievalEvaluator()
        assert evaluator.k_values == [5, 10]

    def test_custom_k_values(self):
        """Test custom K values."""
        evaluator = StandardRetrievalEvaluator(k_values=[1, 3, 5, 10])
        assert evaluator.k_values == [1, 3, 5, 10]

    def test_invalid_k_values(self):
        """Test that invalid K values raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            StandardRetrievalEvaluator(k_values=[5, 0, 10])

    def test_protocol_compliance(self):
        """Test that StandardRetrievalEvaluator satisfies protocol."""
        evaluator = StandardRetrievalEvaluator()
        assert isinstance(evaluator, RetrievalEvaluator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
