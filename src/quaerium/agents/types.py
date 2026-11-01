"""Shared value types of the agents kernel.

Pure data, no I/O. Everything that crosses a boundary — the graph to the API,
the API to the browser, a tool call to the audit table — is one of these.

The ``RunEvent`` union is the SSE protocol. Apps that render it in a browser
mirror it by hand in TypeScript: change one, change the other.

These types also end up inside LangGraph checkpoints, serialized with their
module path. Moving them is a breaking change for stored runs; see
:func:`quaerium.agents.checkpointer.build_serde` and its ``legacy_modules``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Return a short prefixed identifier, e.g. ``run_3f9a2c1b``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# Risk
# ============================================================================


class RiskLevel(StrEnum):
    """How much trust a tool needs before it runs.

    The whole safety model of an agent is this enum plus the gate that reads it.
    What each level means is the app's call; the default policy reads them as:

    - ``READ``: no side effects anywhere. Runs immediately.
    - ``WRITE``: a side effect that is easy to see and undo. Runs immediately.
    - ``DESTRUCTIVE``: hard to reverse. Suspends the run and asks.
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class Decision(StrEnum):
    """What the policy gate decided about a pending call."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


# ============================================================================
# Tool calls
# ============================================================================


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    """The Anthropic ``tool_use`` block id. Tool results must echo it back."""

    name: str
    """Fully qualified, ``namespace__verb`` — e.g. ``kb__search``."""

    args: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLevel = RiskLevel.READ


class ToolResult(BaseModel):
    """The outcome of one tool invocation."""

    id: str
    name: str
    ok: bool
    content: str
    """What goes back to the model. Kept as text: tools return prose or JSON, never objects."""

    duration_ms: float = 0.0

    @property
    def summary(self) -> str:
        """A one-line form for the UI, which does not need the whole payload."""
        first = self.content.strip().splitlines()[0] if self.content.strip() else ""
        return first[:200] + ("…" if len(first) > 200 else "")


# ============================================================================
# Approvals
# ============================================================================


class Approval(BaseModel):
    """A pending or settled human decision on a destructive call."""

    run_id: str
    tool_call: ToolCall
    why: str
    """Why the gate stopped — shown verbatim in the dialog."""

    approved: bool | None = None
    decided_at: datetime | None = None

    @property
    def pending(self) -> bool:
        return self.approved is None


# ============================================================================
# Sessions and turns
# ============================================================================


class Turn(BaseModel):
    """One message in a session's recent history."""

    role: Literal["user", "assistant"]
    content: str
    at: datetime = Field(default_factory=_now)


class Usage(BaseModel):
    """Token accounting for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# ============================================================================
# Run events — the SSE protocol
# ============================================================================


class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any]
    risk: RiskLevel


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    id: str
    name: str
    ok: bool
    summary: str
    duration_ms: float


class ApprovalRequiredEvent(BaseModel):
    type: Literal["approval_required"] = "approval_required"
    run_id: str
    tool: str
    args: dict[str, Any]
    why: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    run_id: str
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    """The model that gave the final answer."""
    requested_model: str = ""
    """The model the run asked for; differs from ``model`` if the gateway swapped it."""


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class ModelUnavailableEvent(BaseModel):
    """The requested model is offline. Nothing ran; the client may resend with ``fallback``.

    Optional: only apps with a model catalog (several models behind a gateway) emit it.
    """

    type: Literal["model_unavailable"] = "model_unavailable"
    requested: str
    fallback: str


RunEvent = Annotated[
    ThinkingEvent
    | TokenEvent
    | ToolCallEvent
    | ToolResultEvent
    | ApprovalRequiredEvent
    | DoneEvent
    | ErrorEvent
    | ModelUnavailableEvent,
    Field(discriminator="type"),
]


__all__ = [
    "Approval",
    "ApprovalRequiredEvent",
    "Decision",
    "DoneEvent",
    "ErrorEvent",
    "ModelUnavailableEvent",
    "RiskLevel",
    "RunEvent",
    "ThinkingEvent",
    "TokenEvent",
    "ToolCall",
    "ToolCallEvent",
    "ToolResult",
    "ToolResultEvent",
    "Turn",
    "Usage",
    "new_id",
]
