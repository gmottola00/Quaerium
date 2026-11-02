"""Audit log.

The record that answers "why did it do that": for each run, the prompt, every
tool call with its arguments and outcome, every approval and how it was decided,
and the final answer.

Always kept in memory so ``/api/runs`` works with no database at all; also
written to Postgres when one is configured, because the interesting question is
usually about a run from last week.

Reads are served from memory. With Postgres, :func:`build_audit_log` fills that
memory from the tables at startup (:meth:`AuditLog.load`), so a restart does not
empty the run list. Which runs are still waiting at the gate is not in the tables:
the checkpointer knows, and :meth:`~quaerium.agents.runner.AgentRunner.restore_pending`
asks it.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime, timezone
from statistics import median
from typing import Any, Deque

from quaerium.agents.logging import current_run, get_logger
from quaerium.agents.types import Approval, ToolCall, ToolResult, Usage

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS {p}_runs (
    run_id       TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    answer       TEXT,
    usage        JSONB,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS {p}_tool_calls (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES {p}_runs(run_id) ON DELETE CASCADE,
    tool         TEXT NOT NULL,
    risk         TEXT NOT NULL,
    args         JSONB NOT NULL,
    ok           BOOLEAN NOT NULL,
    output       TEXT,
    duration_ms  DOUBLE PRECISION,
    at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS {p}_approvals (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES {p}_runs(run_id) ON DELETE CASCADE,
    tool         TEXT NOT NULL,
    args         JSONB NOT NULL,
    why          TEXT NOT NULL,
    approved     BOOLEAN NOT NULL,
    decided_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {p}_runs_session_idx ON {p}_runs(session_id, started_at DESC);
ALTER TABLE {p}_runs ADD COLUMN IF NOT EXISTS model TEXT;
ALTER TABLE {p}_runs ADD COLUMN IF NOT EXISTS served_by TEXT;
ALTER TABLE {p}_tool_calls ADD COLUMN IF NOT EXISTS summary TEXT;
"""

_PREFIX = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def schema(table_prefix: str) -> str:
    """The DDL for one app's audit tables: ``{prefix}_runs``, ``_tool_calls``, ``_approvals``."""
    if not _PREFIX.match(table_prefix):
        raise ValueError(f"invalid table prefix {table_prefix!r}: lowercase letters, digits and _ only")
    return _SCHEMA.format(p=table_prefix)


RISKS = ("read", "write", "destructive")


class AuditLog:
    """Records runs. In-memory always; Postgres additionally, when available.

    Args:
        table_prefix: Postgres tables are ``{prefix}_runs``, ``{prefix}_tool_calls``
            and ``{prefix}_approvals``. An app keeps its history by passing its
            old prefix.
        capacity: How many recent runs to keep in memory.
        pool: An open ``psycopg_pool.AsyncConnectionPool``, or None.
    """

    def __init__(self, *, table_prefix: str = "agent", capacity: int = 200, pool: Any = None) -> None:
        schema(table_prefix)  # validates the prefix before it reaches any SQL
        self._p = table_prefix
        self._runs: Deque[dict[str, Any]] = deque(maxlen=capacity)
        self._index: dict[str, dict[str, Any]] = {}
        self._pool = pool

    # --- recording ----------------------------------------------------------

    async def start_run(self, run_id: str, session_id: str, prompt: str, model: str = "") -> None:
        """Open a run. ``model`` is the one the user picked, before any fallback."""
        record: dict[str, Any] = {
            "run_id": run_id,
            "session_id": session_id,
            "prompt": prompt,
            "model": model,
            "served_by": "",
            "pending": None,
            "answer": "",
            "usage": Usage().model_dump(),
            "tool_calls": [],
            "approvals": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
        if len(self._runs) == self._runs.maxlen and self._runs:
            self._index.pop(self._runs[0]["run_id"], None)
        self._runs.append(record)
        self._index[run_id] = record
        await self._write(
            f"INSERT INTO {self._p}_runs (run_id, session_id, prompt, model) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (run_id, session_id, prompt, model),
        )

    async def record_tool_call(self, run_id: str, call: ToolCall, result: ToolResult) -> None:
        entry = {
            "tool": call.name,
            "risk": call.risk.value,
            "args": call.args,
            "ok": result.ok,
            "summary": result.summary,
            "duration_ms": round(result.duration_ms, 1),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if run_id in self._index:
            self._index[run_id]["tool_calls"].append(entry)
        await self._write(
            f"INSERT INTO {self._p}_tool_calls (run_id, tool, risk, args, ok, output, summary, duration_ms)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                call.name,
                call.risk.value,
                self._json(call.args),
                result.ok,
                result.content[:8000],
                result.summary,
                result.duration_ms,
            ),
        )

    async def record_approval(self, run_id: str, approval: Approval) -> None:
        entry = {
            "tool": approval.tool_call.name,
            "args": approval.tool_call.args,
            "why": approval.why,
            "approved": bool(approval.approved),
            "decided_at": (approval.decided_at or datetime.now(timezone.utc)).isoformat(),
        }
        if run_id in self._index:
            self._index[run_id]["approvals"].append(entry)
            self._index[run_id]["pending"] = None
        logger.info("audit.approval", tool=entry["tool"], approved=entry["approved"])
        await self._write(
            f"INSERT INTO {self._p}_approvals (run_id, tool, args, why, approved) VALUES (%s, %s, %s, %s, %s)",
            (
                run_id,
                approval.tool_call.name,
                self._json(approval.tool_call.args),
                approval.why,
                bool(approval.approved),
            ),
        )

    def suspend_run(self, run_id: str, tool: str, args: dict[str, Any], why: str, *, since: str | None = None) -> None:
        """Mark a run as waiting at the gate. The next :meth:`record_approval` clears it.

        In memory only: the checkpointer is what keeps the run itself, this is
        just what lets ``/api/approvals`` list every waiting run at once.
        """
        if run_id in self._index:
            self._index[run_id]["pending"] = {
                "tool": tool,
                "args": args,
                "why": why,
                "since": since or datetime.now(timezone.utc).isoformat(),
            }

    async def finish_run(self, run_id: str, answer: str, usage: Usage, served_by: str = "") -> None:
        """Close a run. ``served_by`` is the model that actually answered."""
        if run_id in self._index:
            self._index[run_id]["answer"] = answer
            self._index[run_id]["served_by"] = served_by
            self._index[run_id]["usage"] = usage.model_dump()
            self._index[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        await self._write(
            f"UPDATE {self._p}_runs SET answer = %s, usage = %s, served_by = %s, finished_at = now() WHERE run_id = %s",
            (answer, self._json(usage.model_dump()), served_by, run_id),
        )

    def hooks(self) -> dict[str, Any]:
        """The ``on_result`` / ``on_approval`` callbacks for :func:`~quaerium.agents.graph.build_graph`.

        Tool results carry no run id, so it is read from the logging context the
        runner binds for the duration of a run.

        Example:
            >>> graph = build_graph(llm=llm, registry=registry, system=SYSTEM, **audit.hooks())  # doctest: +SKIP
        """

        async def on_result(call: ToolCall, result: ToolResult) -> None:
            await self.record_tool_call(current_run() or "", call, result)

        async def on_approval(approval: Approval) -> None:
            await self.record_approval(approval.run_id, approval)

        return {"on_result": on_result, "on_approval": on_approval}

    # --- reading ------------------------------------------------------------

    def recent(self, *, limit: int = 50, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return recent runs, newest first, with their tool calls counted by risk."""
        runs = [r for r in reversed(self._runs) if session_id is None or r["session_id"] == session_id]
        return [self._summary(r) for r in runs[:limit]]

    @staticmethod
    def _summary(run: dict[str, Any]) -> dict[str, Any]:
        """A run without its tool calls. ``risk`` is the highest one it reached, the waiting call included."""
        by_risk = {risk: sum(1 for c in run["tool_calls"] if c["risk"] == risk) for risk in RISKS}
        reached = [risk for risk in RISKS if by_risk[risk]] + (["destructive"] if run["pending"] else [])
        return {k: v for k, v in run.items() if k != "tool_calls"} | {
            "tool_count": len(run["tool_calls"]),
            "calls_by_risk": by_risk,
            "risk": max(reached, key=RISKS.index) if reached else None,
        }

    def pending(self) -> list[dict[str, Any]]:
        """Every run waiting at the gate, oldest first: the order they should be decided in."""
        return [
            {"run_id": r["run_id"], "session_id": r["session_id"], "prompt": r["prompt"]} | r["pending"]
            for r in self._runs
            if r["pending"]
        ]

    def decisions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Settled approvals across all runs, newest first."""
        settled = [{"run_id": r["run_id"], "prompt": r["prompt"]} | a for r in self._runs for a in r["approvals"]]
        return sorted(settled, key=lambda a: a["decided_at"], reverse=True)[:limit]

    def tool_usage(self, *, since: datetime) -> dict[str, dict[str, Any]]:
        """Per tool, the calls made since ``since``: count, errors, median duration and the last one.

        Counts what is held in memory, so it covers at most the last ``capacity`` runs.
        """
        calls: dict[str, list[dict[str, Any]]] = {}
        for run in self._runs:
            for call in run["tool_calls"]:
                if datetime.fromisoformat(call["at"]) >= since:
                    calls.setdefault(call["tool"], []).append(call)
        return {
            tool: {
                "calls": len(made),
                "errors": sum(1 for c in made if not c["ok"]),
                "p50_ms": median(c["duration_ms"] for c in made),
                "last_at": max(c["at"] for c in made),
            }
            for tool, made in calls.items()
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return one run in full, tool calls and approvals included."""
        return self._index.get(run_id)

    def unfinished(self) -> list[dict[str, Any]]:
        """Runs that never finished and are not known to be waiting, oldest first.

        After a restart these are the candidates for
        :meth:`~quaerium.agents.runner.AgentRunner.restore_pending`: a run that
        stopped at the gate has no answer yet, and neither does one that crashed.
        """
        return [
            {"run_id": r["run_id"], "session_id": r["session_id"], "started_at": r["started_at"]}
            for r in self._runs
            if r["finished_at"] is None and not r["pending"]
        ]

    # --- loading ------------------------------------------------------------

    async def load(self) -> int:
        """Fill memory with the most recent runs from Postgres. Returns how many.

        Best-effort like every other database call here: a failure is logged and
        the log starts empty, it never stops the app. Runs already in memory are
        kept; loaded runs go before them, oldest first.
        """
        if self._pool is None:
            return 0
        try:
            async with self._pool.connection() as conn:
                runs = await self._fetch(
                    conn,
                    f"SELECT run_id, session_id, prompt, model, served_by, answer, usage, started_at, finished_at"
                    f" FROM {self._p}_runs ORDER BY started_at DESC LIMIT %s",
                    (self._runs.maxlen,),
                )
                ids = [row[0] for row in runs]
                calls = await self._fetch(
                    conn,
                    f"SELECT run_id, tool, risk, args, ok, summary, output, duration_ms, at"
                    f" FROM {self._p}_tool_calls WHERE run_id = ANY(%s) ORDER BY id",
                    (ids,),
                )
                approvals = await self._fetch(
                    conn,
                    f"SELECT run_id, tool, args, why, approved, decided_at"
                    f" FROM {self._p}_approvals WHERE run_id = ANY(%s) ORDER BY id",
                    (ids,),
                )
        except Exception as exc:  # noqa: BLE001 - audit is not on the critical path
            logger.warning("audit.load_failed", error=str(exc))
            return 0

        loaded: dict[str, dict[str, Any]] = {}
        for run_id, session_id, prompt, model, served_by, answer, usage, started_at, finished_at in reversed(runs):
            if run_id in self._index:
                continue
            loaded[run_id] = {
                "run_id": run_id,
                "session_id": session_id,
                "prompt": prompt,
                "model": model or "",
                "served_by": served_by or "",
                "pending": None,
                "answer": answer or "",
                "usage": usage or Usage().model_dump(),
                "tool_calls": [],
                "approvals": [],
                "started_at": _iso(started_at),
                "finished_at": _iso(finished_at) if finished_at else None,
            }
        for run_id, tool, risk, args, ok, summary, output, duration_ms, at in calls:
            if run_id in loaded:
                loaded[run_id]["tool_calls"].append(
                    {
                        "tool": tool,
                        "risk": risk,
                        "args": args or {},
                        "ok": ok,
                        # Rows written before 0.4.1 have no summary: derive it the way ToolResult does.
                        "summary": summary if summary is not None else _first_line(output or ""),
                        "duration_ms": round(duration_ms or 0.0, 1),
                        "at": _iso(at),
                    }
                )
        for run_id, tool, args, why, approved, decided_at in approvals:
            if run_id in loaded:
                loaded[run_id]["approvals"].append(
                    {"tool": tool, "args": args or {}, "why": why, "approved": approved, "decided_at": _iso(decided_at)}
                )

        current = list(self._runs)
        self._runs.clear()
        self._index.clear()
        for record in [*loaded.values(), *current][-(self._runs.maxlen or 0) :]:
            self._runs.append(record)
            self._index[record["run_id"]] = record
        logger.info("audit.loaded", runs=len(loaded))
        return len(loaded)

    @staticmethod
    async def _fetch(conn: Any, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        cursor = await conn.execute(sql, params)
        return list(await cursor.fetchall())

    # --- persistence --------------------------------------------------------

    @staticmethod
    def _json(value: Any) -> Any:
        """Adapt a dict for a JSONB column."""
        try:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        except ImportError:  # pragma: no cover - optional dependency
            return value

    async def _write(self, sql: str, params: tuple[Any, ...]) -> None:
        """Best-effort persistence. Losing an audit row must not fail a run."""
        if self._pool is None:
            return
        try:
            async with self._pool.connection() as conn:
                await conn.execute(sql, params)
        except Exception as exc:  # noqa: BLE001 - audit is not on the critical path
            logger.warning("audit.write_failed", error=str(exc))


def _iso(value: Any) -> str:
    """A timestamp from Postgres (``datetime``) or memory (already a string) as ISO 8601."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _first_line(text: str) -> str:
    """Same as :attr:`ToolResult.summary`, for rows that predate the summary column."""
    return ToolResult(id="", name="", ok=True, content=text).summary


async def build_audit_log(dsn: str | None, *, table_prefix: str = "agent", capacity: int = 200) -> tuple[AuditLog, Any]:
    """Build the audit log and, when ``dsn`` is set, the pool backing it.

    Creates the tables if they do not exist, then loads the most recent
    ``capacity`` runs into memory, so a restart keeps the run list.

    Args:
        dsn: A libpq connection string, or None/empty for memory only.
        table_prefix: See :class:`AuditLog`.
        capacity: How many recent runs to keep in memory.

    Returns:
        The log and the connection pool to close on shutdown (None without Postgres).
    """
    if not dsn:
        logger.warning("audit.in_memory", reason="no Postgres DSN configured")
        return AuditLog(table_prefix=table_prefix, capacity=capacity), None

    try:
        from psycopg_pool import AsyncConnectionPool
    except ImportError:  # pragma: no cover - optional dependency
        logger.error("audit.psycopg_missing", hint="pip install 'quaerium[agents-postgres]'")
        return AuditLog(table_prefix=table_prefix, capacity=capacity), None

    pool = AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False)
    await pool.open(wait=True, timeout=15)
    async with pool.connection() as conn:
        await conn.execute(schema(table_prefix))
    logger.info("audit.postgres", table_prefix=table_prefix)
    audit = AuditLog(table_prefix=table_prefix, capacity=capacity, pool=pool)
    await audit.load()
    return audit, pool


__all__ = ["AuditLog", "build_audit_log", "schema"]
