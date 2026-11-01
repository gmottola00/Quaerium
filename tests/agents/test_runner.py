"""The runner: how a run ends, and what it leaves in the session store and audit log."""

from __future__ import annotations

from typing import Any

from quaerium.agents.audit import AuditLog
from quaerium.agents.graph import build_graph
from quaerium.agents.memory import MemorySessionStore
from quaerium.agents.runner import AgentRunner
from quaerium.agents.testing import ScriptedLLM
from quaerium.agents.types import RiskLevel, Usage
from tests.agents.conftest import toy_classify


def with_usage(turn: Any, output_tokens: int) -> Any:
    turn.usage = Usage(input_tokens=10, output_tokens=output_tokens)
    return turn


def make(llm: ScriptedLLM, registry: Any) -> tuple[AgentRunner, MemorySessionStore, AuditLog]:
    sessions, audit = MemorySessionStore(), AuditLog()
    graph = build_graph(
        llm=llm,
        registry=registry,
        system=[],
        classify=toy_classify,
        **audit.hooks(),
    )
    return AgentRunner(graph=graph, sessions=sessions, audit=audit), sessions, audit


async def collect(stream: Any) -> list[dict[str, Any]]:
    return [event async for event in stream]


class TestStart:
    async def test_an_answer_ends_with_done(self, toy_registry) -> None:
        llm = ScriptedLLM([ScriptedLLM.call("toy__peek", {"what": "x"}), ScriptedLLM.say("saw it")])
        runner, sessions, audit = make(llm, toy_registry)

        events = await collect(runner.start("s1", "look?", run_id="run_1"))

        assert [e["type"] for e in events] == ["tool_call", "tool_result", "token", "done"]
        assert events[-1]["run_id"] == "run_1"
        assert [(t.role, t.content) for t in await sessions.history("s1")] == [("user", "look?"), ("assistant", "saw it")]
        run = audit.get("run_1")
        assert run["answer"] == "saw it"
        assert [c["tool"] for c in run["tool_calls"]] == ["toy__peek"]

    async def test_usage_is_this_runs_share_not_the_sessions(self, toy_registry) -> None:
        llm = ScriptedLLM([with_usage(ScriptedLLM.say("one"), 5), with_usage(ScriptedLLM.say("two"), 7)])
        runner, _, _ = make(llm, toy_registry)

        first = await collect(runner.start("s1", "one?"))
        second = await collect(runner.start("s1", "two?"))

        assert first[-1]["usage"]["output_tokens"] == 5
        assert second[-1]["usage"]["output_tokens"] == 7

    async def test_the_requested_model_is_reported(self, toy_registry) -> None:
        llm = ScriptedLLM([ScriptedLLM.say("ok")])
        runner, _, audit = make(llm, toy_registry)

        events = await collect(runner.start("s1", "hi?", run_id="r", model="claude-haiku-4-5", picked="local/x"))

        assert events[-1]["requested_model"] == "claude-haiku-4-5"
        assert audit.get("r")["model"] == "local/x", "the audit keeps what the user picked"

    async def test_a_failure_ends_with_error(self, toy_registry) -> None:
        class Broken(ScriptedLLM):
            async def complete(self, **kwargs: Any) -> Any:
                raise RuntimeError("model down")

        runner, _, _ = make(Broken([]), toy_registry)

        events = await collect(runner.start("s1", "hi?"))

        assert events == [{"type": "error", "message": "RuntimeError: model down"}]


class TestApprovals:
    async def test_a_destructive_call_ends_with_approval_required(self, toy_registry) -> None:
        llm = ScriptedLLM([ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE)])
        runner, _, audit = make(llm, toy_registry)

        events = await collect(runner.start("s1", "demolish x", run_id="run_1"))

        assert events[-1]["type"] == "approval_required"
        assert events[-1]["tool"] == "toy__demolish"
        assert (await runner.pending_approval("s1"))["tool"] == "toy__demolish"
        assert [p["run_id"] for p in audit.pending()] == ["run_1"]

    async def test_resume_approved_runs_the_tool_and_finishes(self, toy_registry) -> None:
        llm = ScriptedLLM(
            [ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE), ScriptedLLM.say("gone")]
        )
        runner, _, audit = make(llm, toy_registry)
        await collect(runner.start("s1", "demolish x", run_id="run_1"))

        events = await collect(runner.resume("s1", "run_1", approved=True))

        assert [e["type"] for e in events] == ["tool_call", "tool_result", "token", "done"]
        assert await runner.pending_approval("s1") is None
        assert audit.pending() == []
        assert audit.decisions()[0]["approved"] is True

    async def test_resume_with_nothing_pending_is_an_error(self, toy_registry) -> None:
        runner, _, _ = make(ScriptedLLM([]), toy_registry)

        events = await collect(runner.resume("s1", "run_x", approved=True))

        assert events[0]["type"] == "error"


class TestHistory:
    async def test_history_is_the_wire_format(self, toy_registry) -> None:
        runner, _, _ = make(ScriptedLLM([ScriptedLLM.say("hello")]), toy_registry)
        await collect(runner.start("s1", "hi?"))

        assert [m["role"] for m in await runner.history("s1")] == ["user", "assistant"]
