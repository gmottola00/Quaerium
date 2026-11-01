"""The audit log: the in-memory record, and the SQL it sends to Postgres."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import pytest

from quaerium.agents.audit import AuditLog, build_audit_log, schema
from quaerium.agents.logging import bind_run, clear_run
from quaerium.agents.types import Approval, RiskLevel, ToolCall, ToolResult, Usage


class FakePool:
    """Records every statement instead of running it."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    @asynccontextmanager
    async def connection(self) -> AsyncIterator["FakePool"]:
        yield self

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.statements.append((sql, params))


def call(name: str = "toy__peek", risk: RiskLevel = RiskLevel.READ) -> ToolCall:
    return ToolCall(id="t1", name=name, args={"a": 1}, risk=risk)


def result(ok: bool = True) -> ToolResult:
    return ToolResult(id="t1", name="toy__peek", ok=ok, content="line one\nline two", duration_ms=12.34)


class TestRecording:
    async def test_a_run_from_start_to_finish(self) -> None:
        audit = AuditLog()
        await audit.start_run("r1", "s1", "look", model="claude-opus-5")
        await audit.record_tool_call("r1", call(), result())
        await audit.finish_run("r1", "done", Usage(output_tokens=3), served_by="claude-haiku-4-5")

        run = audit.get("r1")
        assert run["answer"] == "done"
        assert run["served_by"] == "claude-haiku-4-5"
        assert run["usage"]["output_tokens"] == 3
        assert run["tool_calls"][0]["summary"] == "line one"

    async def test_summary_reports_the_highest_risk_reached(self) -> None:
        audit = AuditLog()
        await audit.start_run("r1", "s1", "x")
        await audit.record_tool_call("r1", call(risk=RiskLevel.WRITE), result())
        audit.suspend_run("r1", "toy__demolish", {}, "why")

        summary = audit.recent()[0]
        assert summary["calls_by_risk"] == {"read": 0, "write": 1, "destructive": 0}
        assert summary["risk"] == "destructive", "the waiting call counts"
        assert "tool_calls" not in summary

    async def test_pending_and_decisions(self) -> None:
        audit = AuditLog()
        await audit.start_run("r1", "s1", "demolish")
        audit.suspend_run("r1", "toy__demolish", {"target": "x"}, "Hard to reverse.")
        assert audit.pending()[0]["tool"] == "toy__demolish"

        approval = Approval(run_id="r1", tool_call=call("toy__demolish"), why="w", approved=True)
        await audit.record_approval("r1", approval)

        assert audit.pending() == []
        assert audit.decisions()[0]["approved"] is True

    async def test_capacity_evicts_the_oldest(self) -> None:
        audit = AuditLog(capacity=2)
        for i in range(3):
            await audit.start_run(f"r{i}", "s", "x")
        assert audit.get("r0") is None
        assert [r["run_id"] for r in audit.recent()] == ["r2", "r1"]

    async def test_tool_usage(self) -> None:
        audit = AuditLog()
        await audit.start_run("r1", "s", "x")
        await audit.record_tool_call("r1", call(), result())
        await audit.record_tool_call("r1", call(), result(ok=False))

        usage = audit.tool_usage(since=datetime.now(timezone.utc) - timedelta(minutes=1))
        assert usage["toy__peek"]["calls"] == 2
        assert usage["toy__peek"]["errors"] == 1

    async def test_hooks_read_the_run_id_from_the_context(self) -> None:
        audit = AuditLog()
        await audit.start_run("r1", "s", "x")
        hooks = audit.hooks()
        bind_run("r1")
        try:
            await hooks["on_result"](call(), result())
        finally:
            clear_run()
        assert len(audit.get("r1")["tool_calls"]) == 1


class TestPostgres:
    async def test_statements_use_the_apps_table_prefix(self) -> None:
        """An app keeps its history across the migration by passing its old prefix."""
        pool = FakePool()
        audit = AuditLog(table_prefix="jarvis", pool=pool)
        await audit.start_run("r1", "s1", "x")
        await audit.record_tool_call("r1", call(), result())
        await audit.record_approval("r1", Approval(run_id="r1", tool_call=call(), why="w", approved=False))
        await audit.finish_run("r1", "a", Usage())

        tables = [re.search(r"(?:INTO|UPDATE) (\w+)", sql).group(1) for sql, _ in pool.statements]  # type: ignore[union-attr]
        assert tables == ["jarvis_runs", "jarvis_tool_calls", "jarvis_approvals", "jarvis_runs"]

    async def test_a_failing_write_does_not_fail_the_run(self) -> None:
        class Broken(FakePool):
            async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
                raise RuntimeError("db down")

        audit = AuditLog(pool=Broken())
        await audit.start_run("r1", "s1", "x")
        assert audit.get("r1") is not None

    def test_schema_is_per_prefix(self) -> None:
        ddl = schema("alfred")
        assert "CREATE TABLE IF NOT EXISTS alfred_runs" in ddl
        assert "REFERENCES alfred_runs(run_id)" in ddl
        assert "{p}" not in ddl

    @pytest.mark.parametrize("prefix", ["", "Jarvis", "a; DROP TABLE x", "1abc", "a-b"])
    def test_an_unsafe_prefix_is_rejected(self, prefix: str) -> None:
        with pytest.raises(ValueError, match="invalid table prefix"):
            AuditLog(table_prefix=prefix)

    async def test_without_a_dsn_it_is_memory_only(self) -> None:
        audit, pool = await build_audit_log(None, table_prefix="jarvis")
        assert pool is None
        assert isinstance(audit, AuditLog)
