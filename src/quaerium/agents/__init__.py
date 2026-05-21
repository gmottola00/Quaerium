"""Quaerium Agents — multi-step reasoning and tool-use system.

Provides a Protocol-based agent framework with ReAct execution,
pluggable memory, and built-in agents for RAG, ingestion, evaluation,
and multi-hop research.

Example::

    from quaerium.agents import RAGAgent, tool
    from quaerium.agents.memory import InMemoryConversationMemory

    @tool(description="Returns today's ISO date")
    def get_date() -> str:
        from datetime import date
        return date.today().isoformat()

    agent = RAGAgent(
        llm=llm,
        rag_pipeline=pipeline,
        extra_tools=[get_date],
        memory=InMemoryConversationMemory(max_turns=20),
        max_steps=8,
    )

    response = agent.run("What are the deadlines in section 4?")
    print(response.answer)
    for step in response.trace.steps:
        print(f"  [{step.step_number}] {step.thought[:80]}")
"""

from __future__ import annotations

# Core protocols
from quaerium.agents.protocols import (
    AgentMemory,
    AgentObserver,
    ExecutionStrategy,
    Tool,
)

# Data models
from quaerium.agents.models import (
    AgentResponse,
    AgentStep,
    AgentTrace,
    ToolCall,
)

# Tool system
from quaerium.agents.tool import (
    FunctionTool,
    ToolDefinition,
    ToolRegistry,
    tool,
)

# Memory
from quaerium.agents.memory import (
    InMemoryConversationMemory,
    VectorLongTermMemory,
)

# Execution
from quaerium.agents.executor import ReActExecutor
from quaerium.agents.runtime import AgentRuntime

# Built-in agents
from quaerium.agents.builtin import (
    EvaluationAgent,
    IngestionAgent,
    RAGAgent,
    ResearchAgent,
)

__all__ = [
    # Protocols
    "Tool",
    "AgentMemory",
    "AgentObserver",
    "ExecutionStrategy",
    # Models
    "ToolCall",
    "AgentStep",
    "AgentTrace",
    "AgentResponse",
    # Tool system
    "tool",
    "FunctionTool",
    "ToolDefinition",
    "ToolRegistry",
    # Memory
    "InMemoryConversationMemory",
    "VectorLongTermMemory",
    # Execution
    "ReActExecutor",
    "AgentRuntime",
    # Built-in agents
    "RAGAgent",
    "IngestionAgent",
    "EvaluationAgent",
    "ResearchAgent",
]
