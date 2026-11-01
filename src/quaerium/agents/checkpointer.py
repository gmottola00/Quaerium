"""Checkpointer construction.

A run that stops to ask a question is only durable if its state was written
somewhere. With Postgres configured, a pending approval survives a backend
restart; without it the agent still works and still interrupts, but a restart
forgets what it was waiting on.

The serializer is given an explicit allowlist rather than left permissive.
Checkpoint deserialization can construct arbitrary types, so naming exactly the
types the kernel stores is both quieter and safer.

Moving types between modules
----------------------------

LangGraph stores each object with its module path (``jarvis.core.types.ToolCall``).
An app that moved its types into :mod:`quaerium.agents.types` still has
checkpoints written under the old path — a run suspended at the gate before the
deploy. Without help those come back as plain dicts, and resuming fails. Pass
the old module as ``legacy_modules`` and they deserialize into the new types.
"""

from __future__ import annotations

import importlib
import sys
import types as _pytypes
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from quaerium.agents import types as agent_types
from quaerium.agents.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_TYPES: tuple[str, ...] = ("RiskLevel", "Decision", "ToolCall", "ToolResult", "Approval", "Turn", "Usage")
"""Every kernel type that can end up inside a checkpoint."""

ALLOWED_TYPES: list[tuple[str, str]] = [(agent_types.__name__, name) for name in CHECKPOINT_TYPES]


def _alias_legacy_module(module: str) -> None:
    """Make ``module.<Type>`` resolve to the kernel type, for every checkpoint type.

    A module that no longer exists is replaced by an alias in ``sys.modules``. A
    module that still exists (it kept the app's own domain types) gets the
    missing names attached, unless it already re-exports them.
    """
    try:
        target = importlib.import_module(module)
    except ImportError:
        alias = _pytypes.ModuleType(module, f"Legacy alias of {agent_types.__name__} for old checkpoints.")
        for name in CHECKPOINT_TYPES:
            setattr(alias, name, getattr(agent_types, name))
        sys.modules[module] = alias
        logger.info("checkpointer.legacy_alias", module=module)
        return

    for name in CHECKPOINT_TYPES:
        current = getattr(target, name, None)
        if current is None:
            setattr(target, name, getattr(agent_types, name))
        elif current is not getattr(agent_types, name):
            logger.warning(
                "checkpointer.legacy_conflict",
                module=module,
                type=name,
                hint="the old module defines its own type; old checkpoints will load it, not the kernel's",
            )


def build_serde(
    *,
    extra_types: Iterable[tuple[str, str]] = (),
    legacy_modules: Iterable[str] = (),
) -> Any:
    """Return a serializer that accepts the kernel's types and nothing else exotic.

    Args:
        extra_types: ``(module, name)`` pairs of app types stored in state.
        legacy_modules: Modules the kernel types used to live in; old
            checkpoints that reference them load as the kernel types.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    allowed = [*ALLOWED_TYPES, *extra_types]
    for module in legacy_modules:
        _alias_legacy_module(module)
        allowed.extend((module, name) for name in CHECKPOINT_TYPES)
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


@asynccontextmanager
async def checkpointer(
    dsn: str | None,
    *,
    extra_types: Iterable[tuple[str, str]] = (),
    legacy_modules: Iterable[str] = (),
) -> AsyncIterator[Any]:
    """Yield a checkpointer: Postgres when ``dsn`` is set, in memory otherwise.

    Args:
        dsn: A libpq connection string (``postgresql://...``). Converting an
            SQLAlchemy URL is the app's job.
        extra_types: See :func:`build_serde`.
        legacy_modules: See :func:`build_serde`.

    Example:
        >>> async with checkpointer(settings.pg_dsn) as saver:      # doctest: +SKIP
        ...     graph = build_graph(llm=llm, registry=registry, system=SYSTEM, checkpointer=saver)
    """
    serde = build_serde(extra_types=extra_types, legacy_modules=legacy_modules)

    if not dsn:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning("checkpointer.in_memory", reason="no Postgres DSN configured")
        yield InMemorySaver(serde=serde)
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:  # pragma: no cover - optional dependency
        from langgraph.checkpoint.memory import InMemorySaver

        logger.error("checkpointer.postgres_missing", hint="pip install 'quaerium[agents-postgres]'")
        yield InMemorySaver(serde=serde)
        return

    async with AsyncPostgresSaver.from_conn_string(dsn, serde=serde) as saver:
        await saver.setup()
        logger.info("checkpointer.postgres")
        yield saver


__all__ = ["ALLOWED_TYPES", "CHECKPOINT_TYPES", "build_serde", "checkpointer"]
