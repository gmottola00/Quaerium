"""Router node.

Decides how much machinery a turn deserves. Small talk gets no tools and low
effort; everything else gets the full tool surface.

The app supplies the heuristic (``classify``), which answers the clear cases for
free in the app's own vocabulary. Only genuinely ambiguous messages cost a model
call, and that call goes to a cheap model when one is given.
"""

from __future__ import annotations

from typing import Any, Callable

from quaerium.agents.llm import LLM
from quaerium.agents.logging import get_logger
from quaerium.agents.state import AgentState, Node

logger = get_logger(__name__)

ROUTES = ("chitchat", "question", "action")

Classifier = Callable[[str], "str | None"]
"""Maps a user message to one of :data:`ROUTES`, or None when it is ambiguous."""

DEFAULT_ROUTER_SYSTEM = (
    "Classify the user's message for an assistant with tools. Answer with exactly one word: "
    "chitchat (greetings, thanks, small talk: no tools needed), "
    "question (asks for information: read-only tools), "
    "or action (asks to change something)."
)


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def make_router(
    *,
    classify: Classifier | None = None,
    router_llm: LLM | None = None,
    router_system: str = "",
) -> Node:
    """Build the router node.

    Args:
        classify: The app's heuristic. None sends every message to the
            ambiguous path.
        router_llm: Optional cheap model for the ambiguous cases. When omitted,
            ambiguity resolves to ``question``: tools on, the question effort.
        router_system: System prompt for ``router_llm``. Defaults to
            :data:`DEFAULT_ROUTER_SYSTEM`.
    """
    system = [{"type": "text", "text": router_system or DEFAULT_ROUTER_SYSTEM}]

    async def router(state: AgentState) -> dict[str, Any]:
        last = _last_user_text(state)

        route = classify(last) if classify is not None else None
        if route is not None and route not in ROUTES:
            logger.warning("router.bad_route", route=route)
            route = None

        if route is None and router_llm is not None:
            turn = await router_llm.complete(
                system=system,
                messages=[{"role": "user", "content": last}],
                tools=None,
                effort="low",
                emit=lambda _kind, _text: None,
            )
            candidate = turn.text.strip().lower().strip(".`")
            route = candidate if candidate in ROUTES else None

        route = route or "question"
        logger.info("router.decided", route=route)
        return {"route": route}

    return router


__all__ = ["DEFAULT_ROUTER_SYSTEM", "ROUTES", "Classifier", "make_router"]
