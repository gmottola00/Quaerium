"""ResearchAgent — multi-hop research agent using sub-agents as tools.

Composes multiple agents (each as a @tool) to perform multi-hop reasoning.
"""

from __future__ import annotations

from typing import Any

from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.models import AgentResponse
from quaerium.agents.runtime import AgentRuntime
from quaerium.agents.tool import FunctionTool


class ResearchAgent:
    """Agent that orchestrates other agents for multi-hop research.

    Each sub-agent is registered as a tool. The ResearchAgent can call
    them in sequence to synthesize a comprehensive answer.

    Args:
        llm: LLMClient for orchestration reasoning
        sub_agents: Dict mapping tool_name -> agent with a .run(task) method
        extra_tools: Additional Tool-compatible objects
        memory: AgentMemory instance
        observers: List of AgentObserver instances
        max_steps: Maximum ReAct iterations
    """

    def __init__(
        self,
        *,
        llm: Any,
        sub_agents: dict[str, Any] | None = None,
        extra_tools: list[Any] | None = None,
        memory: Any | None = None,
        observers: list[Any] | None = None,
        max_steps: int = 12,
    ) -> None:
        tools = self._wrap_sub_agents(sub_agents or {}) + (extra_tools or [])
        self._runtime = AgentRuntime(
            llm=llm,
            tools=tools,
            memory=memory or InMemoryConversationMemory(),
            observers=observers,
            max_steps=max_steps,
        )

    def _wrap_sub_agents(self, sub_agents: dict[str, Any]) -> list[FunctionTool]:
        """Wrap each sub-agent's .run() method as a FunctionTool."""
        tools: list[FunctionTool] = []
        for tool_name, agent in sub_agents.items():
            # Capture agent reference in closure
            def _make_runner(a: Any, tname: str) -> FunctionTool:
                def run_sub_agent(query: str) -> str:
                    f"""Delegate a research query to the {tname} sub-agent."""
                    try:
                        response = a.run(query)
                        return response.answer
                    except Exception as exc:
                        return f"Sub-agent '{tname}' error: {exc}"

                return FunctionTool(
                    fn=run_sub_agent,
                    name=tname,
                    description=f"Delegate a query to the {tname} specialized agent.",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )

            tools.append(_make_runner(agent, tool_name))
        return tools

    def run(self, task: str) -> AgentResponse:
        """Run the research agent on a multi-hop task."""
        return self._runtime.run(task)

    def reset_memory(self) -> None:
        self._runtime.reset_memory()


__all__ = ["ResearchAgent"]
