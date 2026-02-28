"""
Faithfulness and hallucination evaluation.

Evaluates whether a generated answer is faithful to the provided context
and detects hallucinations (claims not supported by context).
"""

from __future__ import annotations

import logging
from typing import Any

from quaerium.infra.evaluation.generation.llm_judge import OpenAIJudge

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    """
    Evaluates how faithful an answer is to the provided context.

    Uses an LLM judge to determine if the answer's claims are
    supported by the context.

    Example:
        >>> judge = OpenAIJudge(llm=my_llm)
        >>> evaluator = FaithfulnessEvaluator(judge)
        >>> score = evaluator.evaluate(
        ...     answer="Python was created in 1991.",
        ...     context=["Python was created by Guido van Rossum in 1991."]
        ... )
        >>> print(f"Faithfulness: {score:.2f}")
    """

    def __init__(self, judge: OpenAIJudge):
        """
        Initialize faithfulness evaluator.

        Args:
            judge: LLM judge for evaluation
        """
        self.judge = judge
        logger.debug("Initialized faithfulness evaluator")

    def evaluate(
        self,
        answer: str,
        context: list[str],
        **kwargs: Any,
    ) -> float:
        """
        Evaluate answer faithfulness to context.

        Args:
            answer: Generated answer
            context: List of context chunks used for generation
            **kwargs: Additional parameters for LLM judge

        Returns:
            Faithfulness score (0-1, where 1 is fully faithful)

        Example:
            >>> score = evaluator.evaluate(
            ...     answer="The capital of France is Paris.",
            ...     context=["Paris is the capital and largest city of France."]
            ... )
            >>> assert 0.0 <= score <= 1.0
        """
        prompt = self._build_faithfulness_prompt(answer, context)
        score = self.judge.judge(prompt, **kwargs)

        logger.debug(f"Faithfulness score: {score:.2f}")
        return score

    def _build_faithfulness_prompt(self, answer: str, context: list[str]) -> str:
        """Build evaluation prompt for faithfulness."""
        context_text = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))

        return f"""Evaluate whether the answer is faithful to the provided context.

Context:
{context_text}

Answer: {answer}

Rate the faithfulness on a scale of 0-10, where:
- 10: All claims in the answer are fully supported by the context
- 7-9: Most claims supported, minor unsupported details
- 4-6: Some claims supported, some unsupported
- 1-3: Few claims supported, mostly unsupported
- 0: No claims are supported by the context

Provide only a numerical score (e.g., "9/10" or "Score: 8")."""


class HallucinationDetector:
    """
    Detects hallucinations in generated answers.

    Hallucination is the inverse of faithfulness: claims in the answer
    that are NOT supported by the context.

    Example:
        >>> judge = OpenAIJudge(llm=my_llm)
        >>> detector = HallucinationDetector(judge)
        >>> score = detector.detect(
        ...     answer="Python supports quantum computing.",
        ...     context=["Python is a general-purpose programming language."]
        ... )
        >>> print(f"Hallucination: {score:.2f}")  # Higher = more hallucination
    """

    def __init__(self, judge: OpenAIJudge):
        """
        Initialize hallucination detector.

        Args:
            judge: LLM judge for evaluation
        """
        self.judge = judge
        logger.debug("Initialized hallucination detector")

    def detect(
        self,
        answer: str,
        context: list[str],
        **kwargs: Any,
    ) -> float:
        """
        Detect hallucinations in answer.

        Args:
            answer: Generated answer
            context: List of context chunks used for generation
            **kwargs: Additional parameters for LLM judge

        Returns:
            Hallucination score (0-1, where 0 is no hallucination, 1 is complete hallucination)

        Note:
            This is the inverse of faithfulness: hallucination = 1 - faithfulness

        Example:
            >>> score = detector.detect(
            ...     answer="Python was invented in 2020.",
            ...     context=["Python was created in 1991."]
            ... )
            >>> assert score > 0.5  # High hallucination since contradicts context
        """
        prompt = self._build_hallucination_prompt(answer, context)
        score = self.judge.judge(prompt, **kwargs)

        logger.debug(f"Hallucination score: {score:.2f}")
        return score

    def _build_hallucination_prompt(self, answer: str, context: list[str]) -> str:
        """Build evaluation prompt for hallucination detection."""
        context_text = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))

        return f"""Detect hallucinations in the answer.

Context:
{context_text}

Answer: {answer}

Rate the hallucination level on a scale of 0-10, where:
- 10: All claims are hallucinated (not supported by context)
- 7-9: Mostly hallucinated with some supported claims
- 4-6: Mix of hallucinated and supported claims
- 1-3: Mostly supported with minor hallucinations
- 0: No hallucinations (all claims supported by context)

Provide only a numerical score (e.g., "2/10" or "Score: 1")."""


__all__ = [
    "FaithfulnessEvaluator",
    "HallucinationDetector",
]
