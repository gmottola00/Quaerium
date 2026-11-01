"""Exception hierarchy of the agents kernel.

One root, ``AgentError``, so a caller can catch everything the kernel raises
without catching everything Python raises. Apps subclass it for their own root
(``class JarvisError(AgentError)``) and add leaves below that.

Nothing here is used for control flow inside the agent loop: a tool that fails
returns an error *result* to the model rather than raising, so the model can
recover. These exceptions are for wiring failures — bad config, unreachable
services, a tool that does not exist.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base for every error the agents kernel raises."""


class ConfigurationError(AgentError):
    """A required setting is missing or inconsistent."""


class ToolError(AgentError):
    """Base for tool-layer failures."""


class ToolNotFoundError(ToolError):
    """The model asked for a tool that is not in the registry."""


class ToolExecutionError(ToolError):
    """A tool raised while executing."""


class UpstreamError(ToolError):
    """A capability's upstream service returned an error or was unreachable."""

    def __init__(self, service: str, message: str, *, status: int | None = None) -> None:
        self.service = service
        self.status = status
        super().__init__(f"{service}: {message}" + (f" (HTTP {status})" if status else ""))


class PolicyError(AgentError):
    """An action was refused by the policy gate."""


class OrchestrationError(AgentError):
    """Base for orchestration failures."""


class MaxStepsExceededError(OrchestrationError):
    """The graph hit its step ceiling without reaching an answer."""


__all__ = [
    "AgentError",
    "ConfigurationError",
    "MaxStepsExceededError",
    "OrchestrationError",
    "PolicyError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "UpstreamError",
]
