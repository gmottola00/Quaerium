"""Session store.

The checkpointer already holds the message history the *model* sees. This holds
the much smaller thing the *browser* needs: the plain user/assistant text of a
conversation, plus enough metadata to list sessions in a sidebar, cheap to read
and easy to expire.

Redis when configured, an in-process dict otherwise. Both satisfy
:class:`~quaerium.agents.protocols.SessionStore`.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Deque

from quaerium.agents.logging import get_logger
from quaerium.agents.types import Turn

logger = get_logger(__name__)

MAX_TURNS = 200


class MemorySessionStore:
    """In-process session history. Lost on restart, which is fine for dev."""

    def __init__(self, *, max_turns: int = MAX_TURNS) -> None:
        self._turns: dict[str, Deque[Turn]] = defaultdict(lambda: deque(maxlen=max_turns))

    async def append(self, session_id: str, turn: Turn) -> None:
        self._turns[session_id].append(turn)

    async def history(self, session_id: str, *, limit: int = 20) -> list[Turn]:
        return list(self._turns[session_id])[-limit:]

    async def clear(self, session_id: str) -> None:
        self._turns.pop(session_id, None)

    async def sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": sid,
                "turns": len(turns),
                "last_at": turns[-1].at.isoformat() if turns else None,
                "title": next((t.content for t in turns if t.role == "user"), "")[:60],
            }
            for sid, turns in sorted(self._turns.items())
            if turns
        ]

    async def aclose(self) -> None:
        return None


class RedisSessionStore:
    """Redis-backed session history with a TTL.

    Keys are ``{prefix}:session:{id}`` (a list of turns) and ``{prefix}:sessions``
    (the set of ids). An app keeps its existing data by passing its old prefix.

    Args:
        url: Redis URL.
        prefix: Key namespace, e.g. the app name.
        ttl: Seconds a session survives without activity.
        max_turns: Trim length, so a long conversation cannot grow without bound.
    """

    def __init__(self, url: str, *, prefix: str = "agent", ttl: int = 86_400, max_turns: int = MAX_TURNS) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._index = f"{prefix}:sessions"
        self._ttl = ttl
        self._max_turns = max_turns

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:session:{session_id}"

    async def append(self, session_id: str, turn: Turn) -> None:
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, turn.model_dump_json())
        pipe.ltrim(key, -self._max_turns, -1)
        pipe.expire(key, self._ttl)
        pipe.sadd(self._index, session_id)
        await pipe.execute()

    async def history(self, session_id: str, *, limit: int = 20) -> list[Turn]:
        raw = await self._redis.lrange(self._key(session_id), -limit, -1)
        return [Turn(**json.loads(item)) for item in raw]

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
        await self._redis.srem(self._index, session_id)

    async def sessions(self) -> list[dict[str, Any]]:
        ids: list[str] = sorted(str(sid) for sid in await self._redis.smembers(self._index))
        out: list[dict[str, Any]] = []
        for sid in ids:
            turns = await self.history(sid, limit=self._max_turns)
            if not turns:
                continue
            out.append(
                {
                    "id": sid,
                    "turns": len(turns),
                    "last_at": turns[-1].at.isoformat(),
                    "title": next((t.content for t in turns if t.role == "user"), "")[:60],
                }
            )
        return out

    async def aclose(self) -> None:
        await self._redis.aclose()


def build_session_store(redis_url: str | None, *, prefix: str = "agent", ttl: int = 86_400) -> Any:
    """Return a Redis store when ``redis_url`` is set, otherwise the in-process one.

    Args:
        redis_url: Redis URL, or None/empty for the in-process store.
        prefix: Key namespace; see :class:`RedisSessionStore`.
        ttl: Seconds a session survives without activity.
    """
    if not redis_url:
        logger.warning("sessions.in_memory", reason="no Redis URL configured")
        return MemorySessionStore()
    try:
        store = RedisSessionStore(redis_url, prefix=prefix, ttl=ttl)
    except ImportError:  # pragma: no cover - optional dependency
        logger.error("sessions.redis_missing", hint="pip install 'quaerium[agents-redis]'")
        return MemorySessionStore()
    logger.info("sessions.redis", prefix=prefix)
    return store


__all__ = ["MAX_TURNS", "MemorySessionStore", "RedisSessionStore", "build_session_store"]
