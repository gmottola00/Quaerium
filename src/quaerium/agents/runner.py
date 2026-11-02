"""Run driver.

Turns a graph invocation into a flat stream of :class:`~quaerium.agents.types.RunEvent`,
and owns the one thing the graph cannot express by itself: what it means when the
stream stops early.

A run ends in exactly one of three ways, and the runner is what tells them apart:

- the model answered — a ``done`` event closes the stream;
- the gate suspended — an ``approval_required`` event closes it, and the state is
  on disk waiting for :meth:`AgentRunner.resume`;
- something broke — an ``error`` event closes it.

A suspension is also the moment someone outside the browser may want to know
("a run is waiting for you"): ``on_suspend`` is called then, in the background,
so a slow notification never holds up the stream.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable

from langgraph.types import Command

from quaerium.agents.graph import initial_state
from quaerium.agents.types import (
    ApprovalRequiredEvent,
    DoneEvent,
    ErrorEvent,
    Turn,
    Usage,
    new_id,
)
from quaerium.agents.logging import bind_run, clear_run, get_logger

logger = get_logger(__name__)

OnSuspend = Callable[[ApprovalRequiredEvent], Awaitable[None]]
"""Called when a run stops at the gate, with the event the browser receives."""


class AgentRunner:
    """Drives the graph for one session at a time.

    Args:
        graph: The compiled graph.
        sessions: Session store, for the history the UI renders.
        audit: Optional audit log.
        on_suspend: Optional callback for a run that stopped at the gate, e.g. a
            push notification. It runs as a background task: the stream does not
            wait for it, and an exception is logged, never raised into the run.
            Call :meth:`aclose` on shutdown to let pending callbacks finish.
    """

    def __init__(
        self, *, graph: Any, sessions: Any = None, audit: Any = None, on_suspend: OnSuspend | None = None
    ) -> None:
        self._graph = graph
        self._sessions = sessions
        self._audit = audit
        self._on_suspend = on_suspend
        self._tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        """The checkpointer thread is the session: history accumulates by itself."""
        return {"configurable": {"thread_id": session_id}}

    async def _usage(self, session_id: str) -> Usage:
        """Read the session's cumulative token usage from the checkpoint."""
        snapshot = await self._graph.aget_state(self._config(session_id))
        return snapshot.values.get("usage") or Usage() if snapshot.values else Usage()

    @staticmethod
    def _delta(before: Usage, after: Usage) -> Usage:
        """Usage accumulates across a session; the UI wants this run's share."""
        return Usage(
            input_tokens=after.input_tokens - before.input_tokens,
            output_tokens=after.output_tokens - before.output_tokens,
            cache_read_tokens=after.cache_read_tokens - before.cache_read_tokens,
            cache_write_tokens=after.cache_write_tokens - before.cache_write_tokens,
        )

    async def _drain(self, payload: Any, session_id: str, run_id: str, before: Usage) -> AsyncIterator[Any]:
        """Stream one graph invocation, then decide how it ended."""
        config = self._config(session_id)
        try:
            async for chunk in self._graph.astream(payload, config=config, stream_mode="custom"):
                yield chunk
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            logger.error("run.failed", error=str(exc), exc_info=True)
            yield ErrorEvent(message=f"{type(exc).__name__}: {exc}").model_dump(mode="json")
            return

        snapshot = await self._graph.aget_state(config)

        if snapshot.interrupts:
            payload_value = snapshot.interrupts[0].value or {}
            logger.info("run.suspended", tool=payload_value.get("tool"))
            event = ApprovalRequiredEvent(
                run_id=run_id,
                tool=str(payload_value.get("tool", "")),
                args=dict(payload_value.get("args") or {}),
                why=str(payload_value.get("why", "")),
            )
            if self._audit is not None:
                self._audit.suspend_run(run_id, event.tool, event.args, event.why)
            self._notify_suspended(event)
            yield event.model_dump(mode="json")
            return

        answer = (snapshot.values or {}).get("answer", "")
        served_by = (snapshot.values or {}).get("served_by", "")
        requested = (snapshot.values or {}).get("model", "")
        usage = self._delta(before, (snapshot.values or {}).get("usage") or Usage())

        if self._sessions is not None and answer:
            await self._sessions.append(session_id, Turn(role="assistant", content=answer))
        if self._audit is not None:
            await self._audit.finish_run(run_id, answer, usage, served_by=served_by)

        logger.info("run.done", output_tokens=usage.output_tokens)
        yield DoneEvent(run_id=run_id, usage=usage, model=served_by, requested_model=requested).model_dump(mode="json")

    def _notify_suspended(self, event: ApprovalRequiredEvent) -> None:
        """Run ``on_suspend`` in the background, holding a reference until it is done."""
        if self._on_suspend is None:
            return
        task = asyncio.create_task(self._call_on_suspend(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _call_on_suspend(self, event: ApprovalRequiredEvent) -> None:
        assert self._on_suspend is not None
        try:
            await self._on_suspend(event)
        except Exception as exc:  # noqa: BLE001 - a notification must never break a run
            logger.warning("run.on_suspend_failed", run_id=event.run_id, error=str(exc))

    async def restore_pending(self) -> int:
        """After a restart, mark in the audit log every run still waiting at the gate.

        The audit tables know which runs never finished; the checkpointer knows
        which of those stopped at the gate. Each unfinished run's session is asked
        for its interrupt, and the run is marked pending when the interrupt is
        that run's. This also covers runs suspended before the audit was loaded
        from Postgres at all. ``on_suspend`` is not called: nobody new is waiting.

        Returns:
            How many runs were marked.
        """
        if self._audit is None:
            return 0
        restored = 0
        for run in self._audit.unfinished():
            try:
                waiting = await self.pending_approval(run["session_id"])
            except Exception as exc:  # noqa: BLE001 - one unreadable checkpoint must not stop startup
                logger.warning("run.restore_failed", run_id=run["run_id"], error=str(exc))
                continue
            if waiting and waiting.get("run_id") == run["run_id"]:
                self._audit.suspend_run(
                    run["run_id"],
                    str(waiting.get("tool", "")),
                    dict(waiting.get("args") or {}),
                    str(waiting.get("why", "")),
                    since=run.get("started_at"),
                )
                restored += 1
        logger.info("run.restored_pending", runs=restored)
        return restored

    async def aclose(self) -> None:
        """Wait for ``on_suspend`` callbacks still running. Call on shutdown."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def start(
        self, session_id: str, message: str, *, run_id: str | None = None, model: str = "", picked: str = ""
    ) -> AsyncIterator[Any]:
        """Begin a run and stream its events.

        Args:
            session_id: Conversation id; also the checkpointer thread.
            message: The user's message.
            run_id: Override the generated run id.
            model: Model for this run, already resolved. Empty means the default.
            picked: The model the user chose, for the audit log. Differs from
                ``model`` when the user agreed to a fallback. Defaults to ``model``.
        """
        run_id = run_id or new_id("run")
        bind_run(run_id, session=session_id)
        try:
            if self._sessions is not None:
                await self._sessions.append(session_id, Turn(role="user", content=message))
            if self._audit is not None:
                await self._audit.start_run(run_id, session_id, message, model=picked or model)

            before = await self._usage(session_id)
            state = initial_state(session_id, message, run_id=run_id, model=model)
            async for event in self._drain(state, session_id, run_id, before):
                yield event
        finally:
            clear_run()

    async def resume(self, session_id: str, run_id: str, *, approved: bool) -> AsyncIterator[Any]:
        """Answer a pending approval and continue the run.

        Args:
            session_id: The suspended run's session.
            run_id: The run id the approval belongs to.
            approved: The user's decision.
        """
        bind_run(run_id, session=session_id)
        try:
            snapshot = await self._graph.aget_state(self._config(session_id))
            if not snapshot.interrupts:
                yield ErrorEvent(message="Nothing is waiting for approval in this session.").model_dump(mode="json")
                return

            before = self._delta(Usage(), (snapshot.values or {}).get("usage") or Usage())
            logger.info("run.resuming", approved=approved)
            async for event in self._drain(Command(resume={"approved": approved}), session_id, run_id, before):
                yield event
        finally:
            clear_run()

    async def pending_approval(self, session_id: str) -> dict[str, Any] | None:
        """Return the approval this session is waiting on, if any.

        This is what lets the dashboard restore a pending dialog after a reload.
        """
        snapshot = await self._graph.aget_state(self._config(session_id))
        if not snapshot.interrupts:
            return None
        return dict(snapshot.interrupts[0].value or {})

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        """Return the raw message history the model sees for a session."""
        snapshot = await self._graph.aget_state(self._config(session_id))
        return list((snapshot.values or {}).get("messages") or [])


__all__ = ["AgentRunner"]
