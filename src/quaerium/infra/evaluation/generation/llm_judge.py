"""
LLM-as-judge implementation for subjective evaluation.

Uses an LLM to evaluate qualities that are difficult to measure
with traditional metrics, such as relevance and faithfulness.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from quaerium.core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class OpenAIJudge:
    """
    LLM-as-judge using OpenAI-compatible LLM clients.

    Uses an LLM to provide numerical judgments on text quality,
    parsing scores from LLM responses.

    Example:
        >>> from quaerium.infra.llm import OpenAIClient
        >>> llm = OpenAIClient(model="gpt-4")
        >>> judge = OpenAIJudge(llm=llm)
        >>> score = judge.judge("Rate this answer on a scale of 0-10: ...")
        >>> print(f"Score: {score:.2f}")  # Normalized to 0-1
    """

    def __init__(self, llm: LLMClient):
        """
        Initialize LLM judge.

        Args:
            llm: LLM client for generating judgments
        """
        self.llm = llm
        logger.debug(f"Initialized LLM judge with model: {llm.model_name}")

    def judge(self, prompt: str, **kwargs: Any) -> float:
        """
        Get a numerical judgment from the LLM.

        Args:
            prompt: Evaluation prompt for the LLM
            **kwargs: Additional parameters for LLM generation

        Returns:
            Score normalized to 0-1 range (0=worst, 1=best)

        Raises:
            ValueError: If LLM response cannot be parsed as a score

        Example:
            >>> prompt = "Rate the relevance of this answer on a scale of 0-10:\\n..."
            >>> score = judge.judge(prompt)
            >>> assert 0.0 <= score <= 1.0
        """
        response = self.llm.generate(prompt, **kwargs)
        score = self._parse_score(response)

        logger.debug(f"LLM judge score: {score:.2f}")
        return score

    def judge_with_reasoning(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[float, str]:
        """
        Get a judgment with explanation from the LLM.

        Args:
            prompt: Evaluation prompt for the LLM
            **kwargs: Additional parameters for LLM generation

        Returns:
            Tuple of (score, reasoning):
                - score: Numerical score normalized to 0-1
                - reasoning: LLM's explanation for the score

        Example:
            >>> prompt = "Evaluate this answer and explain your reasoning:\\n..."
            >>> score, reasoning = judge.judge_with_reasoning(prompt)
            >>> print(f"Score: {score:.2f}, Reasoning: {reasoning}")
        """
        response = self.llm.generate(prompt, **kwargs)
        score = self._parse_score(response)

        logger.debug(f"LLM judge score: {score:.2f}, reasoning: {response[:100]}...")
        return score, response

    def _parse_score(self, response: str) -> float:
        """
        Parse numerical score from LLM response.

        Supports multiple formats:
        - "8/10" or "8 out of 10" -> 0.8
        - "Score: 7" -> 0.7 (assumes 0-10 scale)
        - "0.75" -> 0.75 (assumes 0-1 scale)

        Args:
            response: LLM response text

        Returns:
            Score normalized to 0-1

        Raises:
            ValueError: If no score can be parsed
        """
        # Try pattern: "X/10" or "X out of 10" (supports negative numbers)
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:/|out of)\s*10", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score / 10.0, 0.0), 1.0)

        # Try pattern: "Score: X" (assume 0-10 scale, supports negative)
        match = re.search(r"(?:score|rating):\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # If score > 1, assume 0-10 scale
            if score > 1:
                return min(max(score / 10.0, 0.0), 1.0)
            return min(max(score, 0.0), 1.0)

        # Try pattern: standalone number (first number in response, supports negative)
        match = re.search(r"-?\d+(?:\.\d+)?", response)
        if match:
            score = float(match.group(0))
            # If score > 1, assume 0-10 scale
            if score > 1:
                return min(max(score / 10.0, 0.0), 1.0)
            return min(max(score, 0.0), 1.0)

        raise ValueError(f"Could not parse score from LLM response: {response[:100]}...")


__all__ = ["OpenAIJudge"]
