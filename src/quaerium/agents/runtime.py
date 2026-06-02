"""AgentRuntime — central orchestrator for Quaerium Agents.

AgentRuntime wires together:
- LLMClient (reasoning engine)
- ToolRegistry (available tools)
- AgentMemory (conversation context)
- ExecutionStrategy (ReAct by default)
- AgentObservers (monitoring hooks)
"""

from __future__ import annotations

import logging
from typing import Any

from quaerium.agents.executor import ReActExecutor
from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.models import AgentResponse, AgentTrace
from quaerium.agents.tool import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Central orchestrator that drives an agent conversation.

    Args:
        llm: Any object satisfying the LLMClient Protocol
        tools: List of Tool-compatible objects to register
        memory: AgentMemory instance (default: InMemoryConversationMemory)
        execution_strategy: ExecutionStrategy instance (default: ReActExecutor)
        observers: List of AgentObserver instances
        max_steps: Maximum ReAct iterations per run() call
    """

    def __init__(
        self,
        *,
        llm: Any,
        tools: list[Any] | None = None,
        memory: Any | None = None,
        execution_strategy: Any | None = None,
        observers: list[Any] | None = None,
        max_steps: int = 10,
    ) -> None:
        self.llm = llm
        self.memory = memory if memory is not None else InMemoryConversationMemory()
        self.strategy = execution_strategy if execution_strategy is not None else ReActExecutor()
        self.observers: list[Any] = observers or []
        self.max_steps = max_steps
        self.registry = ToolRegistry()
        for t in (tools or []):
            self.registry.register(t)

    def register_tool(self, tool_obj: Any) -> None:
        """Dynamically add a tool to the runtime."""
        self.registry.register(tool_obj)

    def add_observer(self, observer: Any) -> None:
        """Add an observer to the runtime."""
        self.observers.append(observer)

    def run(self, task: str) -> AgentResponse:
        """Execute a task and return an AgentResponse.

        The task is also stored in memory as a user message.
        The final answer is stored as an assistant message.

        Args:
            task: Natural-language task or question for the agent

        Returns:
            AgentResponse with answer, full trace, and optional sources
        """
        self.memory.add_message("user", task)

        trace: AgentTrace = self.strategy.execute(
            task=task,
            tools=self.registry.all(),
            memory=self.memory,
            llm=self.llm,
            observers=self.observers,
            max_steps=self.max_steps,
        )

        self.memory.add_message("assistant", trace.final_answer)

        return AgentResponse(
            answer=trace.final_answer,
            trace=trace,
            sources=[],
        )

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()


__all__ = ["AgentRuntime"]
