"""Test doubles for the agents kernel.

Lives in the package rather than in a conftest so apps can use it too: in their
own tests, and to run their backend offline against a scripted model.
"""

from __future__ import annotations

from typing import Any

from quaerium.agents.llm import Emit, ModelTurn
from quaerium.agents.types import RiskLevel, ToolCall, new_id


class ScriptedLLM:
    """A model that replays a fixed list of turns.

    Hand the loop a list of responses and assert on what the loop does with
    them. No API key, no network, fully deterministic. Every request it gets is
    kept in :attr:`requests`, so a test can also assert on what was sent.

    Example:
        >>> llm = ScriptedLLM([
        ...     ScriptedLLM.call("kb__search", {"query": "refund policy"}),
        ...     ScriptedLLM.say("Refunds take 14 days."),
        ... ])
    """

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def say(text: str) -> ModelTurn:
        """A turn that just answers."""
        return ModelTurn(content=[{"type": "text", "text": text}], text=text, stop_reason="end_turn")

    @staticmethod
    def call(name: str, args: dict[str, Any], *, risk: Any = None, text: str = "") -> ModelTurn:
        """A turn that calls one tool."""
        call_id = new_id("toolu")
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
        return ModelTurn(
            content=content,
            text=text,
            tool_calls=[ToolCall(id=call_id, name=name, args=args, risk=risk or RiskLevel.READ)],
            stop_reason="tool_use",
        )

    @staticmethod
    def call_many(calls: list[tuple[str, dict[str, Any]]], *, risks: list[Any] | None = None) -> ModelTurn:
        """A turn that calls several tools at once."""
        content: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        for index, (name, args) in enumerate(calls):
            call_id = new_id("toolu")
            content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
            risk = (risks[index] if risks else None) or RiskLevel.READ
            tool_calls.append(ToolCall(id=call_id, name=name, args=args, risk=risk))
        return ModelTurn(content=content, tool_calls=tool_calls, stop_reason="tool_use")

    async def complete(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        effort: str,
        emit: Emit,
        model: str | None = None,
    ) -> ModelTurn:
        self.requests.append(
            {"system": system, "messages": list(messages), "tools": tools, "effort": effort, "model": model}
        )
        if not self._turns:
            return self.say("(scripted model exhausted)")
        turn = self._turns.pop(0)
        if turn.text:
            emit("token", turn.text)
        return turn


__all__ = ["ScriptedLLM"]
