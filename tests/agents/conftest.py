"""Fixtures for the agents kernel.

Fakes are written by hand as real classes satisfying the protocols, rather than
``unittest.mock`` objects: the tests then read as specifications of what each
seam is supposed to do.
"""

from __future__ import annotations

from typing import Any

import pytest

from quaerium.agents.policy import PolicyGate
from quaerium.agents.testing import ScriptedLLM
from quaerium.agents.tools.base import tool
from quaerium.agents.tools.registry import ToolRegistry
from quaerium.agents.types import RiskLevel


def toy_classify(message: str) -> str | None:
    """A tiny router heuristic, standing in for an app's."""
    text = message.strip().lower()
    if text in {"ciao", "hi", "thanks"}:
        return "chitchat"
    if text.startswith(("do ", "deploy", "delete", "write", "demolish")):
        return "action"
    if text.endswith("?"):
        return "question"
    return None


@pytest.fixture
def toy_registry() -> ToolRegistry:
    """A registry with one tool per risk level, plus one that always fails."""

    @tool(namespace="toy", risk=RiskLevel.READ)
    async def peek(what: str = "nothing") -> str:
        """Look at something."""
        return f"saw {what}"

    @tool(namespace="toy", risk=RiskLevel.WRITE)
    async def draft(text: str) -> str:
        """Write a draft."""
        return f"drafted {text}"

    @tool(namespace="toy", risk=RiskLevel.DESTRUCTIVE)
    async def demolish(target: str) -> str:
        """Tear something down."""
        return f"demolished {target}"

    @tool(namespace="toy", risk=RiskLevel.READ)
    async def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    built = ToolRegistry()
    built.register_many([peek, draft, demolish, explode])
    return built


@pytest.fixture
def policy() -> PolicyGate:
    """The default policy: read=auto, write=auto, destructive=confirm."""
    return PolicyGate()


@pytest.fixture
def scripted() -> Any:
    """Factory for a scripted model."""
    return ScriptedLLM
