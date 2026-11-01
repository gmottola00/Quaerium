"""ClientProvider: fake, real or a clear error, decided once and cached."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from quaerium.agents.exceptions import ConfigurationError
from quaerium.agents.tools.provider import ClientProvider


@dataclass
class Settings:
    token: str = ""
    fake_tools: bool = False


class Client:
    def __init__(self, label: str) -> None:
        self.label = label
        self.closed = False

    def close(self) -> None:
        self.closed = True


class AsyncClient(Client):
    async def aclose(self) -> None:
        self.closed = True


def provider(settings: Settings, **kwargs: object) -> ClientProvider[Settings, Client]:
    return ClientProvider(
        name="toy",
        settings=lambda: settings,
        real=lambda s: Client(f"real:{s.token}"),
        requires=lambda s: [] if s.token else ["TOY_TOKEN"],
        **kwargs,  # type: ignore[arg-type]
    )


class TestResolution:
    def test_real_client_is_built_once(self) -> None:
        p = provider(Settings(token="t"))
        assert p.get().label == "real:t"
        assert p.get() is p.get()

    def test_missing_settings_name_what_to_set(self) -> None:
        with pytest.raises(ConfigurationError, match="toy is not configured: set TOY_TOKEN"):
            provider(Settings()).get()

    def test_the_hint_is_appended(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\(or JARVIS_FAKE_TOOLS=true\)"):
            provider(Settings(), hint="or JARVIS_FAKE_TOOLS=true").get()

    def test_fake_when_the_app_says_so(self) -> None:
        p = provider(Settings(fake_tools=True), fake=lambda: Client("fake"), use_fake=lambda s: s.fake_tools)
        assert p.get().label == "fake"
        assert p.has_fake

    def test_use_fake_without_a_fake_falls_through_to_real(self) -> None:
        p = provider(Settings(token="t", fake_tools=True), use_fake=lambda s: s.fake_tools)
        assert p.get().label == "real:t"

    def test_missing(self) -> None:
        assert provider(Settings()).missing() == ["TOY_TOKEN"]
        assert provider(Settings(token="t")).missing() == []


class TestLifecycle:
    def test_set_injects_an_instance(self) -> None:
        p = provider(Settings())
        injected = Client("injected")
        p.set(injected)
        assert p.get() is injected

    def test_reset_closes_a_sync_client(self) -> None:
        p = provider(Settings(token="t"))
        client = p.get()
        p.reset()
        assert client.closed
        assert p.get() is not client

    async def test_aclose_closes_an_async_client(self) -> None:
        p = provider(Settings())
        client = AsyncClient("a")
        p.set(client)
        await p.aclose()
        assert client.closed
