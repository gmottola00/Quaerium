"""Session stores: the in-process one, and Redis against a hand-written fake."""

from __future__ import annotations

from typing import Any

from quaerium.agents.memory import MemorySessionStore, RedisSessionStore, build_session_store
from quaerium.agents.protocols import SessionStore
from quaerium.agents.types import Turn


class FakeRedis:
    """The handful of Redis commands the store uses, over dicts."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttl: dict[str, int] = {}

    def pipeline(self) -> "FakeRedis":
        return self

    def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        items = self.lists.get(key, [])
        self.lists[key] = items[start:] if end == -1 else items[start : end + 1]

    def expire(self, key: str, seconds: int) -> None:
        self.ttl[key] = seconds

    def sadd(self, key: str, value: str) -> None:
        self.sets.setdefault(key, set()).add(value)

    async def execute(self) -> None:
        return None

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def delete(self, key: str) -> None:
        self.lists.pop(key, None)

    async def srem(self, key: str, value: str) -> None:
        self.sets.get(key, set()).discard(value)

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    async def aclose(self) -> None:
        return None


def redis_store(prefix: str = "jarvis", max_turns: int = 200) -> tuple[RedisSessionStore, FakeRedis]:
    store = RedisSessionStore.__new__(RedisSessionStore)
    fake = FakeRedis()
    store._redis = fake
    store._prefix = prefix
    store._index = f"{prefix}:sessions"
    store._ttl = 60
    store._max_turns = max_turns
    return store, fake


async def exercise(store: Any) -> None:
    await store.append("s1", Turn(role="user", content="first question"))
    await store.append("s1", Turn(role="assistant", content="answer"))
    await store.append("s2", Turn(role="user", content="other"))

    assert [t.content for t in await store.history("s1")] == ["first question", "answer"]
    assert [t.content for t in await store.history("s1", limit=1)] == ["answer"]
    sessions = {s["id"]: s for s in await store.sessions()}
    assert sessions["s1"]["turns"] == 2
    assert sessions["s1"]["title"] == "first question"

    await store.clear("s1")
    assert await store.history("s1") == []
    assert [s["id"] for s in await store.sessions()] == ["s2"]


class TestMemoryStore:
    async def test_behaviour(self) -> None:
        await exercise(MemorySessionStore())

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(MemorySessionStore(), SessionStore)

    async def test_turns_are_trimmed(self) -> None:
        store = MemorySessionStore(max_turns=2)
        for i in range(5):
            await store.append("s", Turn(role="user", content=str(i)))
        assert [t.content for t in await store.history("s")] == ["3", "4"]


class TestRedisStore:
    async def test_behaviour(self) -> None:
        store, _ = redis_store()
        await exercise(store)

    async def test_keys_use_the_apps_prefix(self) -> None:
        """An app keeps its sessions across the migration by passing its old prefix."""
        store, fake = redis_store(prefix="alfred")
        await store.append("s1", Turn(role="user", content="x"))

        assert list(fake.lists) == ["alfred:session:s1"]
        assert fake.sets == {"alfred:sessions": {"s1"}}
        assert fake.ttl["alfred:session:s1"] == 60

    def test_satisfies_the_protocol(self) -> None:
        store, _ = redis_store()
        assert isinstance(store, SessionStore)


class TestFactory:
    def test_no_url_means_in_memory(self) -> None:
        assert isinstance(build_session_store(None), MemorySessionStore)
        assert isinstance(build_session_store(""), MemorySessionStore)

    def test_a_url_means_redis(self) -> None:
        store = build_session_store("redis://localhost:6379/0", prefix="jarvis")
        assert isinstance(store, RedisSessionStore)
        assert store._key("s") == "jarvis:session:s"
