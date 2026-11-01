"""The graph.

    router -> model -> gate -> act -> model -> ... -> END
                  \\-> END

The loop is deliberately small. What LangGraph is here for is the checkpointer
and ``interrupt``: a run that stops to ask a question is written to Postgres and
can be resumed by a different process, minutes later.

Example:
    >>> graph = build_graph(llm=llm, registry=registry, system=SYSTEM)   # doctest: +SKIP
    >>> async for chunk in graph.astream(                                 # doctest: +SKIP
    ...     initial_state("s1", "what is our refund policy?"),
    ...     config={"configurable": {"thread_id": "s1"}},
    ...     stream_mode="custom",
    ... ):
    ...     print(chunk)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from quaerium.agents.llm import LLM
from quaerium.agents.logging import get_logger
from quaerium.agents.nodes.act import make_act
from quaerium.agents.nodes.gate import make_gate
from quaerium.agents.nodes.model import (
    DEFAULT_EFFORT_FOR_ROUTE,
    DEFAULT_MAX_STEPS_MESSAGE,
    ContextFn,
    after_model,
    make_model_node,
)
from quaerium.agents.nodes.router import Classifier, make_router
from quaerium.agents.policy import PolicyGate
from quaerium.agents.state import AgentState
from quaerium.agents.tools.registry import ToolRegistry
from quaerium.agents.types import Approval, ToolCall, ToolResult, Usage, new_id

logger = get_logger(__name__)

OnResult = Callable[[ToolCall, ToolResult], Awaitable[None]]
OnApproval = Callable[[Approval], Awaitable[None]]


def build_graph(
    *,
    llm: LLM,
    registry: ToolRegistry,
    system: list[dict[str, Any]],
    context: ContextFn | None = None,
    classify: Classifier | None = None,
    router_llm: LLM | None = None,
    router_system: str = "",
    effort: str = "high",
    effort_for_route: Mapping[str, str] = DEFAULT_EFFORT_FOR_ROUTE,
    max_steps: int = 12,
    max_steps_message: str = DEFAULT_MAX_STEPS_MESSAGE,
    policy: PolicyGate | None = None,
    checkpointer: Any = None,
    on_result: OnResult | None = None,
    on_approval: OnApproval | None = None,
) -> Any:
    """Compile the agent graph.

    Args:
        llm: The model boundary.
        registry: Tools available this run.
        system: Static system blocks, built once and cached by the API.
        context: Async callable returning the per-turn system block.
        classify: The app's router heuristic; see :func:`~quaerium.agents.nodes.router.make_router`.
        router_llm: Optional cheap model for ambiguous routing.
        router_system: System prompt for ``router_llm``.
        effort: Effort for action turns.
        effort_for_route: Cheaper effort per route, e.g. ``{"chitchat": "low"}``.
        max_steps: Ceiling on model turns per run.
        max_steps_message: Answer when the ceiling is hit (``{steps}`` is filled in).
        policy: Risk policy. Defaults to read=auto, write=auto, destructive=confirm.
        checkpointer: LangGraph checkpointer. Required for ``interrupt`` to survive
            a restart; an in-memory saver is used when omitted.
        on_result: Async callback per tool result, for the audit log.
        on_approval: Async callback per settled approval, for the audit log.
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        from quaerium.agents.checkpointer import build_serde

        checkpointer = InMemorySaver(serde=build_serde())

    builder = StateGraph(AgentState)
    # The ignores are a limitation of LangGraph's stubs, not of these nodes:
    # `add_node` infers its state type from an inline callable and cannot do so
    # through the `Node` alias. The nodes themselves are fully typed.
    builder.add_node(  # type: ignore[call-overload]
        "router", make_router(classify=classify, router_llm=router_llm, router_system=router_system)
    )
    builder.add_node(  # type: ignore[call-overload]
        "model",
        make_model_node(
            llm=llm,
            registry=registry,
            system=system,
            context=context,
            effort=effort,
            effort_for_route=effort_for_route,
            max_steps=max_steps,
            max_steps_message=max_steps_message,
        ),
    )
    builder.add_node("gate", make_gate(policy=policy or PolicyGate(), on_approval=on_approval))  # type: ignore[call-overload]
    builder.add_node("act", make_act(registry=registry, on_result=on_result))  # type: ignore[call-overload]

    builder.add_edge(START, "router")
    builder.add_edge("router", "model")
    builder.add_conditional_edges("model", after_model, {"gate": "gate", "end": END})
    builder.add_edge("gate", "act")
    builder.add_edge("act", "model")

    return builder.compile(checkpointer=checkpointer)


def initial_state(
    session_id: str,
    message: str,
    *,
    run_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
    model: str = "",
) -> AgentState:
    """Build the state a new run starts from.

    Args:
        session_id: Conversation this run belongs to. Also the checkpointer thread.
        message: What the user just said.
        run_id: Override the generated run id.
        history: Prior turns, already in Anthropic wire format.
        model: Model for this run. Empty means the configured default.
    """
    return AgentState(
        session_id=session_id,
        run_id=run_id or new_id("run"),
        messages=[*(history or []), {"role": "user", "content": message}],
        route="question",
        model=model,
        served_by="",
        pending=[],
        approved=[],
        denied=[],
        steps=0,
        usage=Usage(),
        answer="",
        error="",
    )


__all__ = ["OnApproval", "OnResult", "build_graph", "initial_state"]
