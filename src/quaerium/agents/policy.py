"""The policy gate.

The single place where an agent decides whether an action may proceed. Its
whole job is a mapping from ``RiskLevel`` to ``Decision``, and the value of
keeping it here — rather than scattered across tool implementations — is that
the answer to "what is the agent allowed to do" is one readable object.

The default policy:

    read = auto · write = auto · destructive = human confirm

The ``why`` texts are shown verbatim in the confirmation dialog, so an app
passes its own: in the user's language, and in terms of what *write* and
*destructive* mean in that app.

Example:
    >>> gate = PolicyGate()
    >>> gate.decide(RiskLevel.READ).decision
    <Decision.ALLOW: 'allow'>
    >>> gate.decide(RiskLevel.DESTRUCTIVE).decision
    <Decision.CONFIRM: 'confirm'>
    >>> PolicyGate(why={RiskLevel.DESTRUCTIVE: "Serve la tua conferma."}).decide(RiskLevel.DESTRUCTIVE).why
    'Serve la tua conferma.'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from quaerium.agents.types import Decision, RiskLevel

DEFAULT_WHY: Mapping[RiskLevel, str] = MappingProxyType(
    {
        RiskLevel.READ: "Read-only: no side effects.",
        RiskLevel.WRITE: "Has a side effect that is easy to see and undo.",
        RiskLevel.DESTRUCTIVE: "Hard to reverse. This one is yours to approve.",
    }
)

DEFAULT_CONFIRM_TEXT = "Every write requires confirmation in this configuration."
DEFAULT_DENY_TEXT = "Destructive actions are disabled in this configuration."


@dataclass(frozen=True)
class Ruling:
    """The gate's answer about one pending call."""

    decision: Decision
    why: str

    @property
    def needs_human(self) -> bool:
        return self.decision is Decision.CONFIRM

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


@dataclass(frozen=True)
class PolicyGate:
    """Maps risk to a decision.

    Args:
        confirm_writes: Escalate ``WRITE`` to a human confirmation too. Off by
            default; turn it on while you are still learning to trust the agent.
        block_destructive: Refuse destructive calls outright instead of asking.
            Useful for an unattended run.
        why: Explanation per risk level, shown in the confirmation dialog.
            Levels left out fall back to :data:`DEFAULT_WHY`.
        confirm_text: The explanation when ``confirm_writes`` escalates a write.
        deny_text: The explanation when ``block_destructive`` refuses a call.
    """

    confirm_writes: bool = False
    block_destructive: bool = False
    why: Mapping[RiskLevel, str] | None = field(default=None, hash=False)
    confirm_text: str = DEFAULT_CONFIRM_TEXT
    deny_text: str = DEFAULT_DENY_TEXT

    def _why(self, risk: RiskLevel) -> str:
        if self.why and risk in self.why:
            return self.why[risk]
        return DEFAULT_WHY[risk]

    def decide(self, risk: RiskLevel) -> Ruling:
        """Rule on a single risk level."""
        if risk is RiskLevel.READ:
            return Ruling(Decision.ALLOW, self._why(RiskLevel.READ))

        if risk is RiskLevel.WRITE:
            if self.confirm_writes:
                return Ruling(Decision.CONFIRM, self.confirm_text)
            return Ruling(Decision.ALLOW, self._why(RiskLevel.WRITE))

        if self.block_destructive:
            return Ruling(Decision.DENY, self.deny_text)
        return Ruling(Decision.CONFIRM, self._why(RiskLevel.DESTRUCTIVE))


__all__ = ["DEFAULT_WHY", "PolicyGate", "Ruling"]
