"""The router node: the app's heuristic first, a cheap model for the ambiguous rest."""

from __future__ import annotations

from typing import Any

import pytest

from quaerium.agents.nodes.router import DEFAULT_ROUTER_SYSTEM, make_router
from quaerium.agents.testing import ScriptedLLM
from tests.agents.conftest import toy_classify


def state(message: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": message}]}


class TestRouter:
    @pytest.mark.parametrize(
        ("message", "route"), [("ciao", "chitchat"), ("deploy it", "action"), ("is it up?", "question")]
    )
    async def test_the_heuristic_decides_the_clear_cases(self, message: str, route: str) -> None:
        router_llm = ScriptedLLM([])
        node = make_router(classify=toy_classify, router_llm=router_llm)

        assert await node(state(message)) == {"route": route}
        assert router_llm.requests == [], "a clear case costs no model call"

    async def test_ambiguity_goes_to_the_router_model(self) -> None:
        router_llm = ScriptedLLM([ScriptedLLM.say("Action.")])
        node = make_router(classify=toy_classify, router_llm=router_llm, router_system="Route it.")

        assert await node(state("tickup")) == {"route": "action"}
        assert router_llm.requests[0]["system"] == [{"type": "text", "text": "Route it."}]
        assert router_llm.requests[0]["tools"] is None

    async def test_default_router_prompt(self) -> None:
        router_llm = ScriptedLLM([ScriptedLLM.say("chitchat")])
        await make_router(router_llm=router_llm)(state("hmm"))
        assert router_llm.requests[0]["system"][0]["text"] == DEFAULT_ROUTER_SYSTEM

    async def test_ambiguity_without_a_model_is_a_question(self) -> None:
        assert await make_router(classify=toy_classify)(state("tickup")) == {"route": "question"}

    async def test_no_heuristic_at_all_is_a_question(self) -> None:
        assert await make_router()(state("ciao")) == {"route": "question"}

    async def test_a_nonsense_answer_from_the_model_is_a_question(self) -> None:
        router_llm = ScriptedLLM([ScriptedLLM.say("I think it's an action, probably")])
        assert await make_router(router_llm=router_llm)(state("tickup")) == {"route": "question"}

    async def test_an_invalid_route_from_the_heuristic_is_ignored(self) -> None:
        assert await make_router(classify=lambda _m: "launch")(state("go")) == {"route": "question"}

    async def test_it_routes_on_the_last_user_text(self) -> None:
        messages = [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]},
            {"role": "user", "content": "deploy it"},
        ]
        assert await make_router(classify=toy_classify)({"messages": messages}) == {"route": "action"}
