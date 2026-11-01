"""HTTP layer for agent apps: SSE streaming and the shared routers.

Needs the ``server`` extra: ``pip install quaerium[server]``.
"""

from __future__ import annotations

try:
    import fastapi  # noqa: F401
    import sse_starlette  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by the minimal install
    raise ImportError(
        "quaerium.agents.server needs FastAPI and sse-starlette: pip install 'quaerium[server]'"
    ) from exc

from quaerium.agents.server.routers import AgentServices, ApprovalRequest, make_approvals_router, make_runs_router
from quaerium.agents.server.sse import stream

__all__ = ["AgentServices", "ApprovalRequest", "make_approvals_router", "make_runs_router", "stream"]
