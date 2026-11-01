"""HTTP routers shared by every agent app.

Approvals (answer a suspended run, list the waiting ones) and runs (the audit
trail) are the same in every app, so they live here. The chat endpoint does not:
validating the message and choosing the model is the app's business.

Each factory takes the app's FastAPI dependency that returns its services. Any
object with ``runner``, ``audit`` and ``durable`` will do (:class:`AgentServices`).

Example:
    >>> app.include_router(make_approvals_router(services))      # doctest: +SKIP
    >>> app.include_router(make_runs_router(services))           # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from quaerium.agents.server.sse import stream


class AgentServices(Protocol):
    """What the shared routers need from the app's services object."""

    @property
    def runner(self) -> Any:
        """An :class:`~quaerium.agents.runner.AgentRunner`."""
        ...

    @property
    def audit(self) -> Any:
        """An :class:`~quaerium.agents.audit.AuditLog`."""
        ...

    @property
    def durable(self) -> bool:
        """Whether the checkpointer survives a restart (a database is configured)."""
        ...


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


def make_approvals_router(services: Callable[..., Any], *, prefix: str = "/api") -> APIRouter:
    """``GET {prefix}/approvals`` and ``POST {prefix}/approvals/{run_id}``.

    Args:
        services: FastAPI dependency returning an :class:`AgentServices`.
        prefix: Path prefix.
    """
    router = APIRouter(prefix=prefix, tags=["approvals"])

    @router.get("/approvals")
    async def overview(svc: Any = Depends(services)) -> dict[str, Any]:
        """Every run waiting at the gate, and the most recent decisions.

        ``durable`` is false without a database: the checkpointer is then in memory,
        and a restart forgets every waiting run.
        """
        return {
            "durable": bool(svc.durable),
            "pending": svc.audit.pending(),
            "decided": svc.audit.decisions(limit=20),
        }

    @router.post("/approvals/{run_id}")
    async def decide(run_id: str, request: ApprovalRequest, svc: Any = Depends(services)) -> Any:
        """Approve or decline a pending destructive action, and stream the continuation.

        The run resumes from its checkpoint, so this works after a page reload or a
        backend restart — as long as a database is configured.
        """
        return stream(svc.runner.resume(request.session_id, run_id, approved=request.approved))

    return router


def make_runs_router(services: Callable[..., Any], *, prefix: str = "/api") -> APIRouter:
    """``GET {prefix}/runs`` and ``GET {prefix}/runs/{run_id}``.

    Args:
        services: FastAPI dependency returning an :class:`AgentServices`.
        prefix: Path prefix.
    """
    router = APIRouter(prefix=prefix, tags=["runs"])

    @router.get("/runs")
    async def list_runs(limit: int = 50, session_id: str | None = None, svc: Any = Depends(services)) -> list[dict[str, Any]]:
        """List recent runs, newest first."""
        return list(svc.audit.recent(limit=limit, session_id=session_id))

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str, svc: Any = Depends(services)) -> dict[str, Any]:
        """Return one run in full: prompt, tool calls, approvals, answer, usage."""
        run = svc.audit.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return dict(run)

    return router


__all__ = ["AgentServices", "ApprovalRequest", "make_approvals_router", "make_runs_router"]
