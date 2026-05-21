"""Data models for the Quaerium Agents system.

All models are plain dataclasses — no Pydantic dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a single tool invocation by the agent."""

    tool_name: str
    arguments: dict[str, Any]
    raw_llm_output: str = ""


@dataclass
class AgentStep:
    """One reasoning step in the ReAct loop."""

    step_number: int
    thought: str
    tool_call: ToolCall | None
    observation: str | None
    timestamp: float
    latency_ms: float = 0.0


@dataclass
class AgentTrace:
    """Full execution trace for an agent run."""

    task: str
    steps: list[AgentStep]
    final_answer: str
    success: bool
    total_latency_ms: float
    error: str | None = None


@dataclass
class AgentResponse:
    """High-level response returned to the caller."""

    answer: str
    trace: AgentTrace
    sources: list[Any] = field(default_factory=list)
    """RetrievedChunk instances when coming from RAGAgent, otherwise empty."""


__all__ = ["ToolCall", "AgentStep", "AgentTrace", "AgentResponse"]
