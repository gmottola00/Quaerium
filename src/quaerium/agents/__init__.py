"""The agents kernel: a small LangGraph loop around the Anthropic SDK.

    router -> model -> gate -> act -> model -> ... -> END

What it gives an app:

- **Tools** written as typed async functions (:func:`tool`), each with a
  :class:`RiskLevel`, collected in a :class:`ToolRegistry`.
- **A policy gate** (:class:`PolicyGate`) that lets reads and writes through and
  suspends destructive calls until a human answers — durably, via LangGraph's
  ``interrupt`` and a Postgres checkpointer.
- **A runner** (:class:`AgentRunner`) that turns a run into a flat stream of
  :data:`RunEvent` for the browser, and resumes suspended runs.
- **Sessions and audit** (Redis / Postgres, both optional) and a ready HTTP layer
  in :mod:`quaerium.agents.server`.
- **RAG as tools** (:func:`rag_tool`, :func:`graph_query_tool`).

Needs the ``agents`` extra: ``pip install quaerium[agents]``.

Example:
    >>> from quaerium.agents import AgentRunner, ToolRegistry, build_graph, tool, RiskLevel
    >>> from quaerium.agents.testing import ScriptedLLM
    >>>
    >>> @tool(namespace="clock", risk=RiskLevel.READ)
    ... async def now() -> str:
    ...     '''Current UTC time.'''
    ...     return "2026-11-01T12:00:00Z"
    >>>
    >>> registry = ToolRegistry()
    >>> registry.register(now)
    >>> llm = ScriptedLLM([ScriptedLLM.call("clock__now", {}), ScriptedLLM.say("It is noon.")])
    >>> runner = AgentRunner(graph=build_graph(llm=llm, registry=registry, system=[]))
"""

from __future__ import annotations

try:
    import anthropic  # noqa: F401
    import langgraph  # noqa: F401
    import structlog  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by the minimal install
    raise ImportError(
        "quaerium.agents needs anthropic, langgraph and structlog: pip install 'quaerium[agents]'"
    ) from exc

from quaerium.agents.audit import AuditLog, build_audit_log
from quaerium.agents.checkpointer import build_serde, checkpointer
from quaerium.agents.exceptions import (
    AgentError,
    ConfigurationError,
    MaxStepsExceededError,
    OrchestrationError,
    PolicyError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    UpstreamError,
)
from quaerium.agents.graph import build_graph, initial_state
from quaerium.agents.llm import LLM, AnthropicLLM, ModelTurn
from quaerium.agents.logging import bind_run, clear_run, configure_logging, current_run, get_logger
from quaerium.agents.memory import MemorySessionStore, RedisSessionStore, build_session_store
from quaerium.agents.nodes.model import ContextFn
from quaerium.agents.nodes.router import Classifier
from quaerium.agents.policy import PolicyGate, Ruling
from quaerium.agents.runner import AgentRunner
from quaerium.agents.state import AgentState
from quaerium.agents.tools import ClientProvider, ToolRegistry, ToolSpec, graph_query_tool, rag_tool, tool
from quaerium.agents.types import (
    Approval,
    ApprovalRequiredEvent,
    Decision,
    DoneEvent,
    ErrorEvent,
    ModelUnavailableEvent,
    RiskLevel,
    RunEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    Turn,
    Usage,
    new_id,
)

__all__ = [
    # Types
    "Approval",
    "ApprovalRequiredEvent",
    "Decision",
    "DoneEvent",
    "ErrorEvent",
    "ModelUnavailableEvent",
    "RiskLevel",
    "RunEvent",
    "ThinkingEvent",
    "TokenEvent",
    "ToolCall",
    "ToolCallEvent",
    "ToolResult",
    "ToolResultEvent",
    "Turn",
    "Usage",
    "new_id",
    # Errors
    "AgentError",
    "ConfigurationError",
    "MaxStepsExceededError",
    "OrchestrationError",
    "PolicyError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "UpstreamError",
    # Tools
    "ClientProvider",
    "ToolRegistry",
    "ToolSpec",
    "graph_query_tool",
    "rag_tool",
    "tool",
    # Model
    "LLM",
    "AnthropicLLM",
    "ModelTurn",
    # Graph and runner
    "AgentRunner",
    "AgentState",
    "Classifier",
    "ContextFn",
    "PolicyGate",
    "Ruling",
    "build_graph",
    "initial_state",
    # Persistence
    "AuditLog",
    "MemorySessionStore",
    "RedisSessionStore",
    "build_audit_log",
    "build_serde",
    "build_session_store",
    "checkpointer",
    # Logging
    "bind_run",
    "clear_run",
    "configure_logging",
    "current_run",
    "get_logger",
]
