"""The runner: how a run ends, and what it leaves in the session store and audit log."""

from __future__ import annotations

import asyncio
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


def make(llm: ScriptedLLM, registry: Any, **runner_kwargs: Any) -> tuple[AgentRunner, MemorySessionStore, AuditLog]:
    sessions, audit = MemorySessionStore(), AuditLog()
    graph = build_graph(
        llm=llm,
        registry=registry,
        system=[],
        classify=toy_classify,
        **audit.hooks(),
    )
    return AgentRunner(graph=graph, sessions=sessions, audit=audit, **runner_kwargs), sessions, audit


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


class TestOnSuspend:
    async def test_a_suspension_calls_on_suspend_once(self, toy_registry) -> None:
        seen: list[Any] = []

        async def on_suspend(event: Any) -> None:
            seen.append(event)

        llm = ScriptedLLM([ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE)])
        runner, _, _ = make(llm, toy_registry, on_suspend=on_suspend)

        await collect(runner.start("s1", "demolish x", run_id="run_1"))
        await runner.aclose()

        assert [(e.run_id, e.tool) for e in seen] == [("run_1", "toy__demolish")]

    async def test_the_stream_does_not_wait_for_it(self, toy_registry) -> None:
        release = asyncio.Event()

        async def slow(_event: Any) -> None:
            await release.wait()

        llm = ScriptedLLM([ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE)])
        runner, _, _ = make(llm, toy_registry, on_suspend=slow)

        events = await asyncio.wait_for(collect(runner.start("s1", "demolish x")), timeout=5)

        assert events[-1]["type"] == "approval_required"
        release.set()
        await runner.aclose()

    async def test_a_failing_callback_does_not_break_the_run(self, toy_registry) -> None:
        async def broken(_event: Any) -> None:
            raise RuntimeError("hub down")

        llm = ScriptedLLM(
            [ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE), ScriptedLLM.say("gone")]
        )
        runner, _, _ = make(llm, toy_registry, on_suspend=broken)

        await collect(runner.start("s1", "demolish x", run_id="run_1"))
        await runner.aclose()
        events = await collect(runner.resume("s1", "run_1", approved=True))

        assert events[-1]["type"] == "done"

    async def test_an_answer_does_not_call_it(self, toy_registry) -> None:
        seen: list[Any] = []

        async def on_suspend(event: Any) -> None:
            seen.append(event)

        runner, _, _ = make(ScriptedLLM([ScriptedLLM.say("hello")]), toy_registry, on_suspend=on_suspend)
        await collect(runner.start("s1", "hi?"))
        await runner.aclose()

        assert seen == []


class TestRestorePending:
    async def test_a_restart_finds_the_waiting_run(self, toy_registry) -> None:
        """A new audit log (empty, as after a restart) over the same checkpoints."""
        llm = ScriptedLLM(
            [
                ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE),
                ScriptedLLM.say("hello"),
            ]
        )
        runner, _, before = make(llm, toy_registry)
        await collect(runner.start("s1", "demolish x", run_id="run_1"))
        await collect(runner.start("s2", "hi?", run_id="run_2"))

        after = AuditLog()
        for run in before.recent():
            await after.start_run(run["run_id"], run["session_id"], run["prompt"])
        await after.finish_run("run_2", "hello", Usage())
        restarted = AgentRunner(graph=runner._graph, audit=after)

        assert await restarted.restore_pending() == 1

        (pending,) = after.pending()
        assert pending["run_id"] == "run_1"
        assert pending["tool"] == "toy__demolish"
        assert pending["since"] == after.get("run_1")["started_at"]  # type: ignore[index]

    async def test_an_older_run_of_a_session_is_not_marked(self, toy_registry) -> None:
        """Only the session's current interrupt counts, and it names its own run."""
        llm = ScriptedLLM([ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE)])
        runner, _, _ = make(llm, toy_registry)
        await collect(runner.start("s1", "demolish x", run_id="run_new"))

        audit = AuditLog()
        await audit.start_run("run_old", "s1", "crashed earlier")
        restarted = AgentRunner(graph=runner._graph, audit=audit)

        assert await restarted.restore_pending() == 0
        assert audit.pending() == []

    async def test_without_an_audit_log_there_is_nothing_to_do(self, toy_registry) -> None:
        runner = AgentRunner(graph=build_graph(llm=ScriptedLLM([]), registry=toy_registry, system=[]))
        assert await runner.restore_pending() == 0


class TestHistory:
    async def test_history_is_the_wire_format(self, toy_registry) -> None:
        runner, _, _ = make(ScriptedLLM([ScriptedLLM.say("hello")]), toy_registry)
        await collect(runner.start("s1", "hi?"))

        assert [m["role"] for m in await runner.history("s1")] == ["user", "assistant"]
