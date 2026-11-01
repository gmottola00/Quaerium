"""Client resolution for capabilities.

Every capability needs the same decision on first use: serve a fake, build the
real client, or explain what is missing. This puts that decision in one place
so each capability's tools stay about the tools.

The provider knows nothing about the app's settings class: it is handed a
function that returns the current settings, and functions that read from them.

Example:
    >>> provider = ClientProvider(                                 # doctest: +SKIP
    ...     name="argocd",
    ...     settings=get_settings,
    ...     real=lambda s: ArgoCDClient(s.argocd_url, s.argocd_token),
    ...     fake=FakeArgoCDClient,
    ...     use_fake=lambda s: s.fake_tools,
    ...     requires=lambda s: [] if s.argocd_token else ["JARVIS_ARGOCD_TOKEN"],
    ... )
    >>> client = provider.get()                                    # doctest: +SKIP
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Generic, TypeVar

from quaerium.agents.exceptions import ConfigurationError
from quaerium.agents.logging import get_logger

logger = get_logger(__name__)

S = TypeVar("S")
T = TypeVar("T")


def _never(_settings: Any) -> bool:
    return False


def _nothing_missing(_settings: Any) -> list[str]:
    return []


class ClientProvider(Generic[S, T]):
    """Lazily builds and caches one capability's client.

    Args:
        name: Capability name, used in log lines and error messages.
        settings: Returns the app's current settings object.
        real: Builds the live client from settings.
        fake: Builds an offline stand-in. Optional: a capability that owns
            local data can be tested against the real thing instead.
        use_fake: Whether to serve the fake, given the settings. Ignored
            without ``fake``.
        requires: Returns the names of settings that are missing. A non-empty
            result means the live client cannot be built.
        hint: Appended to the "not configured" error, e.g. how to enable fakes.
    """

    def __init__(
        self,
        *,
        name: str,
        settings: Callable[[], S],
        real: Callable[[S], T],
        fake: Callable[[], T] | None = None,
        use_fake: Callable[[S], bool] = _never,
        requires: Callable[[S], list[str]] = _nothing_missing,
        hint: str = "",
    ) -> None:
        self._name = name
        self._settings = settings
        self._real = real
        self._fake = fake
        self._use_fake = use_fake
        self._requires = requires
        self._hint = hint
        self._cached: T | None = None

    def get(self) -> T:
        """Return the client, building it on first use.

        Raises:
            ConfigurationError: Live mode was needed but settings are missing.
                The message names the exact settings to set.
        """
        if self._cached is not None:
            return self._cached

        settings = self._settings()
        if self._fake is not None and self._use_fake(settings):
            logger.info("capability.fake", capability=self._name)
            self._cached = self._fake()
            return self._cached

        missing = self._requires(settings)
        if missing:
            message = f"{self._name} is not configured: set {', '.join(missing)}"
            raise ConfigurationError(f"{message} ({self._hint})" if self._hint else message)

        logger.info("capability.live", capability=self._name)
        self._cached = self._real(settings)
        return self._cached

    def set(self, client: T) -> None:
        """Use this instance from now on. For the app's lifespan and for tests."""
        self._cached = client

    @property
    def name(self) -> str:
        """The capability this provider serves, e.g. ``github``."""
        return self._name

    @property
    def has_fake(self) -> bool:
        """Whether an offline stand-in exists."""
        return self._fake is not None

    def missing(self) -> list[str]:
        """The settings the live client still needs. Empty when it can be built."""
        return self._requires(self._settings())

    def reset(self) -> None:
        """Drop the cached client, closing it best-effort if it has a sync ``close``.

        Used by tests and by settings reloads. A client that needs an async
        close should be closed with :meth:`aclose` instead.
        """
        closer = getattr(self._cached, "close", None)
        if closer is not None and not inspect.iscoroutinefunction(closer):
            try:
                result = closer()
                if inspect.iscoroutine(result):  # pragma: no cover - close() returned a coroutine
                    result.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("capability.close_failed", capability=self._name)
        self._cached = None

    async def aclose(self) -> None:
        """Close the cached client, if it has an async ``aclose``, and drop it."""
        client: Any = self._cached
        if client is not None and hasattr(client, "aclose"):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("capability.close_failed", capability=self._name)
        self._cached = None


__all__ = ["ClientProvider"]
