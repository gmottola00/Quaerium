"""Model node.

Builds the request, streams the turn out to the UI, and puts the assistant's
content back into state. It appends ``response.content`` verbatim — thinking
blocks included — because those blocks have to go back over the wire unchanged
on the next request.

The system prompt comes in two parts: ``system``, the static blocks the app
builds once (and marks for prompt caching), and ``context``, an optional block
rebuilt every turn (today's date, the user's categories) and appended after the
cached prefix so it never invalidates it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping

from quaerium.agents.events import emit_thinking, emit_token
from quaerium.agents.llm import LLM
from quaerium.agents.logging import get_logger
from quaerium.agents.state import AgentState, Node
from quaerium.agents.tools.registry import ToolRegistry

logger = get_logger(__name__)

ContextFn = Callable[[], Awaitable["dict[str, Any] | None"]]
"""Builds the volatile system block for one turn, or None for no block."""

DEFAULT_EFFORT_FOR_ROUTE: Mapping[str, str] = MappingProxyType({"chitchat": "low"})

DEFAULT_MAX_STEPS_MESSAGE = "Stopped after {steps} steps without finishing. Ask me to continue if that was too soon."


def make_model_node(
    *,
    llm: LLM,
    registry: ToolRegistry,
    system: list[dict[str, Any]],
    context: ContextFn | None = None,
    effort: str = "high",
    effort_for_route: Mapping[str, str] = DEFAULT_EFFORT_FOR_ROUTE,
    max_steps: int = 12,
    max_steps_message: str = DEFAULT_MAX_STEPS_MESSAGE,
) -> Node:
    """Build the model node.

    Args:
        llm: The model boundary.
        registry: Supplies the tool definitions offered to the model.
        system: The static system blocks, in Anthropic wire format.
        context: Optional per-turn system block, appended after ``system``.
        effort: Effort for routes not in ``effort_for_route`` (actions, usually).
        effort_for_route: Cheaper effort for some routes.
        max_steps: Ceiling on model turns in one run.
        max_steps_message: The answer when the ceiling is hit; ``{steps}`` is
            replaced by the count.
    """
    static = list(system)

    async def model(state: AgentState) -> dict[str, Any]:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            logger.warning("model.max_steps", steps=steps)
            return {"answer": max_steps_message.format(steps=steps), "pending": [], "error": "max_steps"}

        route = state.get("route", "question")
        tools = None if route == "chitchat" else registry.to_anthropic()

        volatile = await context() if context is not None else None

        def emit(kind: str, text: str) -> None:
            emit_thinking(text) if kind == "thinking" else emit_token(text)

        turn = await llm.complete(
            system=[*static, volatile] if volatile else static,
            messages=state["messages"],
            tools=tools,
            effort=effort_for_route.get(route, effort),
            emit=emit,
            model=state.get("model") or None,
        )

        logger.info(
            "model.turn",
            model=turn.model,
            stop_reason=turn.stop_reason,
            tool_calls=[c.name for c in turn.tool_calls],
            output_tokens=turn.usage.output_tokens,
            cache_read=turn.usage.cache_read_tokens,
        )

        update: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": turn.content}],
            "pending": turn.tool_calls,
            "steps": steps + 1,
            "usage": turn.usage,
            "served_by": turn.model,
        }

        if turn.stop_reason == "refusal":
            update["answer"] = turn.refusal
            update["error"] = "refusal"
            update["pending"] = []
        elif not turn.tool_calls:
            update["answer"] = turn.text

        return update

    return model


def after_model(state: AgentState) -> str:
    """Route out of the model node: to the gate when tools were requested, else stop."""
    return "gate" if state.get("pending") else "end"


__all__ = ["DEFAULT_EFFORT_FOR_ROUTE", "DEFAULT_MAX_STEPS_MESSAGE", "ContextFn", "after_model", "make_model_node"]
