"""The policy gate's mapping."""

from __future__ import annotations

from quaerium.agents.policy import DEFAULT_WHY, PolicyGate
from quaerium.agents.types import Decision, RiskLevel


class TestPolicyGate:
    def test_default_policy(self, policy) -> None:
        assert policy.decide(RiskLevel.READ).decision is Decision.ALLOW
        assert policy.decide(RiskLevel.WRITE).decision is Decision.ALLOW
        assert policy.decide(RiskLevel.DESTRUCTIVE).decision is Decision.CONFIRM

    def test_a_question_is_asked_even_when_destructive_is_blocked(self) -> None:
        for gate in (PolicyGate(), PolicyGate(block_destructive=True), PolicyGate(confirm_writes=True)):
            ruling = gate.decide(RiskLevel.ASK)
            assert ruling.decision is Decision.ASK
            assert ruling.needs_human

    def test_confirm_writes_escalates(self) -> None:
        gate = PolicyGate(confirm_writes=True)
        assert gate.decide(RiskLevel.WRITE).needs_human
        assert gate.decide(RiskLevel.READ).allowed

    def test_block_destructive_refuses_instead_of_asking(self) -> None:
        gate = PolicyGate(block_destructive=True)
        assert gate.decide(RiskLevel.DESTRUCTIVE).decision is Decision.DENY


class TestTexts:
    def test_defaults(self, policy) -> None:
        assert policy.decide(RiskLevel.DESTRUCTIVE).why == DEFAULT_WHY[RiskLevel.DESTRUCTIVE]

    def test_the_app_supplies_its_own_why(self) -> None:
        gate = PolicyGate(why={RiskLevel.WRITE: "Scrive solo su Git."})
        assert gate.decide(RiskLevel.WRITE).why == "Scrive solo su Git."
        assert gate.decide(RiskLevel.READ).why == DEFAULT_WHY[RiskLevel.READ], "unset levels fall back"

    def test_confirm_and_deny_texts(self) -> None:
        gate = PolicyGate(confirm_writes=True, block_destructive=True, confirm_text="Conferma.", deny_text="No.")
        assert gate.decide(RiskLevel.WRITE).why == "Conferma."
        assert gate.decide(RiskLevel.DESTRUCTIVE).why == "No."

    def test_a_gate_with_texts_is_still_hashable(self) -> None:
        """Frozen dataclass: usable as a dict key or in a set, the mapping notwithstanding."""
        assert hash(PolicyGate(why={RiskLevel.READ: "x"})) == hash(PolicyGate(why={RiskLevel.READ: "y"}))
