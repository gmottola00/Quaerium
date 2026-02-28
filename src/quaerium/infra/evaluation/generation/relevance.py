"""
Answer relevance evaluation.

Evaluates how well a generated answer addresses the question.
"""

from __future__ import annotations

import logging
from typing import Any

from quaerium.infra.evaluation.generation.llm_judge import OpenAIJudge

logger = logging.getLogger(__name__)


class AnswerRelevanceEvaluator:
    """
    Evaluates how relevant an answer is to the question.

    Uses an LLM judge to score answer relevance on a 0-1 scale.

    Example:
        >>> judge = OpenAIJudge(llm=my_llm)
        >>> evaluator = AnswerRelevanceEvaluator(judge)
        >>> score = evaluator.evaluate(
        ...     question="What is Python?",
        ...     answer="Python is a programming language."
        ... )
        >>> print(f"Relevance: {score:.2f}")
    """

    def __init__(self, judge: OpenAIJudge):
        """
        Initialize relevance evaluator.

        Args:
            judge: LLM judge for evaluation
        """
        self.judge = judge
        logger.debug("Initialized answer relevance evaluator")

    def evaluate(
        self,
        question: str,
        answer: str,
        **kwargs: Any,
    ) -> float:
        """
        Evaluate answer relevance.

        Args:
            question: Original question
            answer: Generated answer
            **kwargs: Additional parameters for LLM judge

        Returns:
            Relevance score (0-1, where 1 is most relevant)

        Example:
            >>> score = evaluator.evaluate(
            ...     question="How do I install Python?",
            ...     answer="Download from python.org and run the installer."
            ... )
            >>> assert 0.0 <= score <= 1.0
        """
        prompt = self._build_relevance_prompt(question, answer)
        score = self.judge.judge(prompt, **kwargs)

        logger.debug(f"Answer relevance score: {score:.2f}")
        return score

    def _build_relevance_prompt(self, question: str, answer: str) -> str:
        """Build evaluation prompt for relevance."""
        return f"""Evaluate how relevant this answer is to the question.

Question: {question}

Answer: {answer}

Rate the relevance on a scale of 0-10, where:
- 10: Perfectly answers the question
- 7-9: Mostly relevant with minor issues
- 4-6: Partially relevant
- 1-3: Barely relevant
- 0: Completely irrelevant

Provide only a numerical score (e.g., "8/10" or "Score: 7")."""


__all__ = ["AnswerRelevanceEvaluator"]
