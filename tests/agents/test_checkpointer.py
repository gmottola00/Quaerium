"""Checkpointer and serializer, including runs suspended before a type moved module."""

from __future__ import annotations

import sys
import types as pytypes
from typing import Any

import pytest

from quaerium.agents import types as agent_types
from quaerium.agents.checkpointer import ALLOWED_TYPES, CHECKPOINT_TYPES, build_serde, checkpointer
from quaerium.agents.graph import build_graph, initial_state
from quaerium.agents.runner import AgentRunner
from quaerium.agents.testing import ScriptedLLM
from quaerium.agents.types import RiskLevel, ToolCall

LEGACY = "legacy_app_for_tests.core.types"


@pytest.fixture
def legacy_module() -> Any:
    """An app's old types module, holding its own copy of ToolCall."""

    class LegacyToolCall(ToolCall):
        pass

    LegacyToolCall.__module__ = LEGACY
    LegacyToolCall.__qualname__ = LegacyToolCall.__name__ = "ToolCall"
    module = pytypes.ModuleType(LEGACY)
    module.ToolCall = LegacyToolCall  # type: ignore[attr-defined]
    sys.modules[LEGACY] = module
    yield module
    sys.modules.pop(LEGACY, None)


def legacy_turn(module: Any) -> Any:
    """A model turn whose tool call is an instance of the *old* class."""
    turn = ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE)
    call = turn.tool_calls[0]
    turn.tool_calls = [module.ToolCall(id=call.id, name=call.name, args=call.args, risk=call.risk)]
    return turn


class TestSerde:
    def test_the_allowlist_names_the_kernel_types(self) -> None:
        assert ALLOWED_TYPES == [("quaerium.agents.types", name) for name in CHECKPOINT_TYPES]

    def test_kernel_types_round_trip(self) -> None:
        serde = build_serde()
        call = ToolCall(id="t", name="toy__peek", args={"a": 1}, risk=RiskLevel.WRITE)

        assert serde.loads_typed(serde.dumps_typed(call)) == call


class TestLegacyModules:
    async def _suspend_with_legacy_types(self, module: Any, toy_registry: Any, new_serde: Any) -> Any:
        llm = ScriptedLLM([legacy_turn(module), ScriptedLLM.say("gone")])
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver(serde=build_serde(extra_types=[(LEGACY, "ToolCall")]))
        graph = build_graph(llm=llm, registry=toy_registry, system=[], checkpointer=saver)
        runner = AgentRunner(graph=graph)
        events = [e async for e in runner.start("s1", "demolish x", run_id="run_1")]
        assert events[-1]["type"] == "approval_required"

        # The deploy: the old module is gone, and the process restarts with a new serde.
        sys.modules.pop(LEGACY, None)
        saver.serde = new_serde()
        return runner

    async def test_without_legacy_modules_the_old_run_cannot_resume(self, legacy_module, toy_registry) -> None:
        runner = await self._suspend_with_legacy_types(legacy_module, toy_registry, build_serde)

        events = [e async for e in runner.resume("s1", "run_1", approved=True)]

        assert events[-1]["type"] == "error", "the call comes back as a dict and the gate cannot read it"

    async def test_with_legacy_modules_the_old_run_resumes(self, legacy_module, toy_registry) -> None:
        runner = await self._suspend_with_legacy_types(
            legacy_module, toy_registry, lambda: build_serde(legacy_modules=[LEGACY])
        )

        events = [e async for e in runner.resume("s1", "run_1", approved=True)]

        assert [e["type"] for e in events] == ["tool_call", "tool_result", "token", "done"]
        assert events[1]["ok"] is True

    def test_a_missing_module_becomes_an_alias(self) -> None:
        name = "gone_app_for_tests.types"
        try:
            build_serde(legacy_modules=[name])
            assert sys.modules[name].ToolCall is agent_types.ToolCall
        finally:
            sys.modules.pop(name, None)

    def test_a_module_that_still_exists_gets_the_missing_names(self) -> None:
        """An app that kept its domain types in the old module."""
        name = "kept_app_for_tests.types"
        module = pytypes.ModuleType(name)
        module.Transaction = object  # type: ignore[attr-defined]
        sys.modules[name] = module
        try:
            build_serde(legacy_modules=[name])
            assert module.Usage is agent_types.Usage  # type: ignore[attr-defined]
            assert module.Transaction is object  # type: ignore[attr-defined]
        finally:
            sys.modules.pop(name, None)


class TestFactory:
    async def test_without_a_dsn_it_is_in_memory(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        async with checkpointer(None) as saver:
            assert isinstance(saver, InMemorySaver)
