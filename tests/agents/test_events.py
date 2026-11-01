"""Event emission is defensive: outside a run there is nowhere to write, and that is fine."""

from __future__ import annotations

from quaerium.agents.events import emit_error, emit_token, emit_tool_call
from quaerium.agents.types import RiskLevel, ToolCall


def test_emitting_outside_a_run_does_not_raise() -> None:
    emit_token("x")
    emit_error("y")
    emit_tool_call(ToolCall(id="t", name="toy__peek", risk=RiskLevel.READ))
