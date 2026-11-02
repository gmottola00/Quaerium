"""A restart against a real Postgres: the run list and the waiting runs survive.

Needs a database, so it only runs with ``QUAERIUM_TEST_PG_DSN`` set, e.g.
``postgresql://postgres@localhost/postgres``. Each test uses its own table
prefix and drops its tables afterwards.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator

import pytest

from quaerium.agents.audit import build_audit_log
from quaerium.agents.checkpointer import checkpointer
from quaerium.agents.graph import build_graph
from quaerium.agents.runner import AgentRunner
from quaerium.agents.testing import ScriptedLLM
from quaerium.agents.types import RiskLevel

DSN = os.environ.get("QUAERIUM_TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="QUAERIUM_TEST_PG_DSN is not set")


@pytest.fixture
async def prefix() -> AsyncIterator[str]:
    name = f"qt_{uuid.uuid4().hex[:8]}"
    yield name
    import psycopg

    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as conn:
        for table in ("approvals", "tool_calls", "runs"):
            await conn.execute(f"DROP TABLE IF EXISTS {name}_{table}")


async def boot(prefix: str, llm: Any, registry: Any, stack: Any) -> tuple[AgentRunner, Any]:
    """What an app's lifespan does: audit (which loads), checkpointer, graph, runner, restore."""
    audit, pool = await build_audit_log(DSN, table_prefix=prefix)
    stack.push_async_callback(pool.close)
    saver = await stack.enter_async_context(checkpointer(DSN))
    graph = build_graph(llm=llm, registry=registry, system=[], checkpointer=saver, **audit.hooks())
    runner = AgentRunner(graph=graph, audit=audit)
    await runner.restore_pending()
    return runner, audit


async def test_a_restart_keeps_runs_and_waiting_approvals(prefix: str, toy_registry: Any) -> None:
    from contextlib import AsyncExitStack

    session = f"ses_{uuid.uuid4().hex[:8]}"
    async with AsyncExitStack() as stack:
        llm = ScriptedLLM(
            [
                ScriptedLLM.call("toy__peek", {"what": "x"}),
                ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE),
            ]
        )
        runner, _ = await boot(prefix, llm, toy_registry, stack)
        events = [e async for e in runner.start(session, "demolish x", run_id=f"run_{prefix}")]
        assert events[-1]["type"] == "approval_required"

    async with AsyncExitStack() as stack:
        runner, audit = await boot(prefix, ScriptedLLM([ScriptedLLM.say("gone")]), toy_registry, stack)

        run = audit.get(f"run_{prefix}")
        assert run is not None and run["prompt"] == "demolish x"
        assert [c["summary"] for c in run["tool_calls"]] == ["saw x"]
        assert [p["tool"] for p in audit.pending()] == ["toy__demolish"]

        events = [e async for e in runner.resume(session, f"run_{prefix}", approved=True)]
        assert events[-1]["type"] == "done"
        assert audit.pending() == []
        assert audit.decisions()[0]["approved"] is True
