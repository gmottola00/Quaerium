"""
Tests for generation evaluation metrics.

Tests LLM-based evaluation including relevance, faithfulness,
and hallucination detection using mock LLM clients.
"""

import pytest

from quaerium.core.evaluation.protocols import GenerationEvaluator
from quaerium.infra.evaluation.generation import (
    AnswerRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationDetector,
    OpenAIJudge,
    StandardGenerationEvaluator,
)
from tests.conftest import MockLLMClient


class TestOpenAIJudge:
    """Test OpenAI judge implementation."""

    def test_parse_score_from_fraction(self):
        """Test parsing score from X/10 format."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["8/10"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.8

    def test_parse_score_from_out_of(self):
        """Test parsing score from 'X out of 10' format."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["7 out of 10"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.7

    def test_parse_score_from_labeled(self):
        """Test parsing score from 'Score: X' format."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["Score: 9"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.9

    def test_parse_score_from_labeled_rating(self):
        """Test parsing score from 'Rating: X' format."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["Rating: 6"]
)
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.6

    def test_parse_score_standalone_number(self):
        """Test parsing standalone number."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["The score is 8.5"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.85

    def test_parse_score_decimal(self):
        """Test parsing decimal score (0-1 scale)."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["0.75"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.75

    def test_parse_score_clamping_high(self):
        """Test that scores > 10 are clamped to 1.0."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["15/10"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 1.0

    def test_parse_score_clamping_low(self):
        """Test that negative scores are clamped to 0.0."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["-2/10"])
        judge = OpenAIJudge(llm=mock_llm)

        score = judge.judge("Test prompt")
        assert score == 0.0

    def test_parse_score_failure(self):
        """Test that unparseable response raises error."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["This answer is good."])
        judge = OpenAIJudge(llm=mock_llm)

        with pytest.raises(ValueError, match="Could not parse score"):
            judge.judge("Test prompt")

    def test_judge_with_reasoning(self):
        """Test judge_with_reasoning returns both score and text."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["Score: 8. The answer is mostly relevant."])
        judge = OpenAIJudge(llm=mock_llm)

        score, reasoning = judge.judge_with_reasoning("Test prompt")
        assert score == 0.8
        assert "mostly relevant" in reasoning.lower()


class TestAnswerRelevanceEvaluator:
    """Test answer relevance evaluator."""

    def test_high_relevance(self):
        """Test with highly relevant answer."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["9/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = AnswerRelevanceEvaluator(judge)

        score = evaluator.evaluate(
            question="What is Python?",
            answer="Python is a programming language.",
        )
        assert score == 0.9

    def test_low_relevance(self):
        """Test with low relevance answer."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["2/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = AnswerRelevanceEvaluator(judge)

        score = evaluator.evaluate(
            question="What is Python?",
            answer="I like cats.",
        )
        assert score == 0.2

    def test_prompt_contains_question_and_answer(self):
        """Test that evaluation prompt contains question and answer."""
        mock_llm = MockLLMClient()
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = AnswerRelevanceEvaluator(judge)

        question = "What is RAG?"
        answer = "RAG is retrieval-augmented generation."

        prompt = evaluator._build_relevance_prompt(question, answer)
        assert question in prompt
        assert answer in prompt
        assert "relevance" in prompt.lower()


class TestFaithfulnessEvaluator:
    """Test faithfulness evaluator."""

    def test_high_faithfulness(self):
        """Test with fully faithful answer."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["10/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = FaithfulnessEvaluator(judge)

        score = evaluator.evaluate(
            answer="Python was created in 1991.",
            context=["Python was created by Guido van Rossum in 1991."],
        )
        assert score == 1.0

    def test_low_faithfulness(self):
        """Test with unfaithful answer."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["1/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = FaithfulnessEvaluator(judge)

        score = evaluator.evaluate(
            answer="Python was created in 2020.",
            context=["Python was created in 1991."],
        )
        assert score == 0.1

    def test_prompt_contains_answer_and_context(self):
        """Test that evaluation prompt contains answer and context."""
        mock_llm = MockLLMClient()
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = FaithfulnessEvaluator(judge)

        answer = "RAG is a technique."
        context = ["RAG combines retrieval with generation."]

        prompt = evaluator._build_faithfulness_prompt(answer, context)
        assert answer in prompt
        assert context[0] in prompt
        assert "faithful" in prompt.lower()

    def test_multiple_context_chunks(self):
        """Test with multiple context chunks."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["9/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = FaithfulnessEvaluator(judge)

        context = [
            "Python is a programming language.",
            "It was created in 1991.",
            "It is widely used for data science.",
        ]

        score = evaluator.evaluate(
            answer="Python is a programming language from 1991.",
            context=context,
        )
        assert score == 0.9


class TestHallucinationDetector:
    """Test hallucination detector."""

    def test_no_hallucination(self):
        """Test with no hallucinations."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["0/10"])
        judge = OpenAIJudge(llm=mock_llm)
        detector = HallucinationDetector(judge)

        score = detector.detect(
            answer="Python was created in 1991.",
            context=["Python was created by Guido van Rossum in 1991."],
        )
        assert score == 0.0

    def test_high_hallucination(self):
        """Test with high hallucination."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["9/10"])
        judge = OpenAIJudge(llm=mock_llm)
        detector = HallucinationDetector(judge)

        score = detector.detect(
            answer="Python was created in 2020 by aliens.",
            context=["Python was created in 1991."],
        )
        assert score == 0.9

    def test_prompt_contains_answer_and_context(self):
        """Test that detection prompt contains answer and context."""
        mock_llm = MockLLMClient()
        judge = OpenAIJudge(llm=mock_llm)
        detector = HallucinationDetector(judge)

        answer = "RAG uses quantum computing."
        context = ["RAG combines retrieval with generation."]

        prompt = detector._build_hallucination_prompt(answer, context)
        assert answer in prompt
        assert context[0] in prompt
        assert "hallucination" in prompt.lower()


class TestStandardGenerationEvaluator:
    """Test standard generation evaluator."""

    def test_complete_evaluation(self):
        """Test complete evaluation with all metrics."""
        mock_llm = MockLLMClient()
        # Set responses for: relevance, faithfulness, hallucination
        mock_llm.set_responses(["8/10", "9/10", "1/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = StandardGenerationEvaluator(judge)

        metrics = evaluator.evaluate_answer(
            question="What is Python?",
            generated_answer="Python is a programming language created in 1991.",
            context=["Python is a high-level programming language.", "It was created in 1991."],
        )

        # Check all metrics are present
        assert metrics.relevance_score == 0.8
        assert metrics.faithfulness_score == 0.9
        assert metrics.hallucination_score == 0.1
        assert metrics.answer_similarity is None  # Not implemented yet

        # Check metadata
        assert metrics.metadata["question"] == "What is Python?"
        assert metrics.metadata["num_context_chunks"] == 2
        assert "judge_model" in metrics.metadata

    def test_with_reference_answer(self):
        """Test evaluation with reference answer."""
        mock_llm = MockLLMClient()
        mock_llm.set_responses(["9/10", "10/10", "0/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = StandardGenerationEvaluator(judge)

        metrics = evaluator.evaluate_answer(
            question="What is Python?",
            generated_answer="Python is a programming language.",
            context=["Python is a high-level programming language."],
            reference_answer="Python is a programming language.",
        )

        # Reference answer provided but similarity not yet implemented
        assert metrics.answer_similarity is None
        assert metrics.relevance_score == 0.9
        assert metrics.faithfulness_score == 1.0
        assert metrics.hallucination_score == 0.0

    def test_poor_quality_answer(self):
        """Test with poor quality answer."""
        mock_llm = MockLLMClient()
        # Low relevance, low faithfulness, high hallucination
        mock_llm.set_responses(["2/10", "3/10", "8/10"])
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = StandardGenerationEvaluator(judge)

        metrics = evaluator.evaluate_answer(
            question="What is Python?",
            generated_answer="I don't know about programming.",
            context=["Python is a programming language."],
        )

        assert metrics.relevance_score == 0.2
        assert metrics.faithfulness_score == 0.3
        assert metrics.hallucination_score == 0.8

    def test_protocol_compliance(self):
        """Test that StandardGenerationEvaluator satisfies protocol."""
        mock_llm = MockLLMClient()
        judge = OpenAIJudge(llm=mock_llm)
        evaluator = StandardGenerationEvaluator(judge)

        assert isinstance(evaluator, GenerationEvaluator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
