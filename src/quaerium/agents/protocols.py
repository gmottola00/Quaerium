"""Core Protocols for the Quaerium Agents system.

Defines the structural contracts (Protocol classes) that all agent components
must satisfy. Uses runtime_checkable to enable isinstance() checks.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quaerium.agents.models import AgentStep, AgentTrace


@runtime_checkable
class Tool(Protocol):
    """Protocol for agent tools.

    Any object with these attributes and methods can be used as a tool
    in the agent system (duck typing).
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def run(self, **kwargs: Any) -> str:
        """Execute the tool synchronously."""
        ...

    async def arun(self, **kwargs: Any) -> str:
        """Execute the tool asynchronously."""
        ...


@runtime_checkable
class AgentMemory(Protocol):
    """Protocol for agent conversation memory."""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory."""
        ...

    def get_history(self, max_turns: int | None = None) -> list[dict[str, str]]:
        """Retrieve conversation history."""
        ...

    def clear(self) -> None:
        """Clear all stored messages."""
        ...


@runtime_checkable
class AgentObserver(Protocol):
    """Protocol for observing agent execution events.

    Observers implement the observer pattern to monitor agent steps
    without modifying agent code.
    """

    def on_step_start(self, step: AgentStep) -> None:
        """Called when an agent step begins."""
        ...

    def on_step_end(self, step: AgentStep) -> None:
        """Called when an agent step completes."""
        ...

    def on_tool_call(self, tool_name: str, kwargs: dict[str, Any], result: str) -> None:
        """Called after a tool is invoked."""
        ...

    def on_agent_finish(self, trace: AgentTrace) -> None:
        """Called when the agent produces its final answer."""
        ...


@runtime_checkable
class ExecutionStrategy(Protocol):
    """Protocol for agent execution strategies.

    Swappable execution loop (ReAct, Plan-then-Execute, etc.).
    """

    def execute(
        self,
        *,
        task: str,
        tools: dict[str, Any],
        memory: Any,
        llm: Any,
        observers: list[Any],
        max_steps: int,
    ) -> AgentTrace:
        """Execute an agent task and return the full trace."""
        ...


__all__ = ["Tool", "AgentMemory", "AgentObserver", "ExecutionStrategy"]
