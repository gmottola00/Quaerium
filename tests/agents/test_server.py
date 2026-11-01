"""The shared HTTP routers, mounted on a minimal FastAPI app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402

from quaerium.agents.audit import AuditLog  # noqa: E402
from quaerium.agents.graph import build_graph  # noqa: E402
from quaerium.agents.runner import AgentRunner  # noqa: E402
from quaerium.agents.server import make_approvals_router, make_runs_router  # noqa: E402
from quaerium.agents.testing import ScriptedLLM  # noqa: E402
from quaerium.agents.types import RiskLevel  # noqa: E402


@dataclass
class Services:
    runner: AgentRunner
    audit: AuditLog
    durable: bool = False


def app_for(llm: ScriptedLLM, registry: Any) -> tuple[FastAPI, Services]:
    audit = AuditLog()
    graph = build_graph(llm=llm, registry=registry, system=[], **audit.hooks())
    svc = Services(runner=AgentRunner(graph=graph, audit=audit), audit=audit)
    app = FastAPI()
    app.include_router(make_approvals_router(lambda: svc))
    app.include_router(make_runs_router(lambda: svc))
    return app, svc


def client(app: FastAPI) -> Any:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def sse_events(body: str) -> list[str]:
    return [line.split(":", 1)[1].strip() for line in body.splitlines() if line.startswith("event:")]


class TestApprovals:
    async def test_list_then_approve_streams_the_continuation(self, toy_registry) -> None:
        llm = ScriptedLLM(
            [ScriptedLLM.call("toy__demolish", {"target": "x"}, risk=RiskLevel.DESTRUCTIVE), ScriptedLLM.say("gone")]
        )
        app, svc = app_for(llm, toy_registry)
        [e async for e in svc.runner.start("s1", "demolish x", run_id="run_1")]

        async with client(app) as http:
            overview = (await http.get("/api/approvals")).json()
            assert overview["durable"] is False
            assert [p["run_id"] for p in overview["pending"]] == ["run_1"]

            response = await http.post("/api/approvals/run_1", json={"session_id": "s1", "approved": True})

        assert response.headers["content-type"].startswith("text/event-stream")
        assert sse_events(response.text) == ["tool_call", "tool_result", "token", "done"]

    async def test_a_custom_prefix(self, toy_registry) -> None:
        audit = AuditLog()
        svc = Services(runner=AgentRunner(graph=build_graph(llm=ScriptedLLM([]), registry=toy_registry, system=[])), audit=audit)
        app = FastAPI()
        app.include_router(make_approvals_router(lambda: svc, prefix="/v2"))

        async with client(app) as http:
            assert (await http.get("/v2/approvals")).status_code == 200


class TestRuns:
    async def test_list_and_get(self, toy_registry) -> None:
        llm = ScriptedLLM([ScriptedLLM.call("toy__peek", {}), ScriptedLLM.say("ok")])
        app, svc = app_for(llm, toy_registry)
        [e async for e in svc.runner.start("s1", "look?", run_id="run_1")]

        async with client(app) as http:
            runs = (await http.get("/api/runs", params={"session_id": "s1"})).json()
            run = (await http.get("/api/runs/run_1")).json()
            missing = await http.get("/api/runs/nope")

        assert [r["run_id"] for r in runs] == ["run_1"]
        assert [c["tool"] for c in run["tool_calls"]] == ["toy__peek"]
        assert missing.status_code == 404
