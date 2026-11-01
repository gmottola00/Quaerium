"""The model boundary: what each kind of model is sent, and what is kept of its reply."""

from __future__ import annotations

from typing import Any

from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage

import pytest

from quaerium.agents.exceptions import ConfigurationError
from quaerium.agents.llm import LOCAL_MAX_TOKENS, AnthropicLLM, build_request, is_native
from quaerium.agents.types import RiskLevel

THINKING = {"type": "thinking", "thinking": "hmm", "signature": "sig"}
HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "what is the policy?"},
    {
        "role": "assistant",
        "content": [THINKING, {"type": "tool_use", "id": "toolu_1", "name": "kb__search", "input": {}}],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "14 days"}]},
]


def request_for(model: str) -> dict[str, Any]:
    return build_request(model=model, max_tokens=64000, system=[], messages=HISTORY, tools=None, effort="high")


def message(*blocks: dict[str, Any]) -> ParsedBetaMessage[None]:
    """What ``stream.get_final_message()`` returns: the parsed variant."""
    return ParsedBetaMessage[None].model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "content": list(blocks),
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class TestRequest:
    def test_claude_is_native_and_local_is_not(self) -> None:
        assert is_native("claude-haiku-4-5")
        assert not is_native("local/qwen3-30b")

    def test_claude_gets_the_full_request_and_its_history_untouched(self) -> None:
        request = request_for("claude-opus-5")

        assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert request["output_config"] == {"effort": "high"}
        assert request["betas"] == ["server-side-fallback-2026-07-01"]
        assert request["max_tokens"] == 64000
        assert request["messages"] is HISTORY

    def test_a_local_model_gets_the_portable_subset(self) -> None:
        request = request_for("local/qwen3-30b")

        assert not {"thinking", "output_config", "betas", "fallbacks"} & request.keys()
        assert request["max_tokens"] == LOCAL_MAX_TOKENS

    def test_a_local_model_never_sees_claude_thinking(self) -> None:
        request = request_for("local/qwen3-30b")

        assistant = request["messages"][1]["content"]
        assert [b["type"] for b in assistant] == ["tool_use"]
        assert HISTORY[1]["content"][0] is THINKING, "the stored history must not be mutated"


class TestReply:
    llm = AnthropicLLM(api_key="sk-test", model="claude-opus-5", risk_of=lambda _name: RiskLevel.READ)

    def test_claude_thinking_is_kept_for_the_next_request(self) -> None:
        turn = self.llm._to_turn(message(THINKING, {"type": "text", "text": "ok"}), served="claude-opus-5")
        assert [b["type"] for b in turn.content] == ["thinking", "text"]

    def test_local_thinking_is_dropped_so_claude_never_receives_it(self) -> None:
        turn = self.llm._to_turn(
            message(THINKING, {"type": "tool_use", "id": "call_1", "name": "kb__search", "input": {}}),
            served="local/qwen3-30b",
            keep_thinking=False,
        )

        assert [b["type"] for b in turn.content] == ["tool_use"]
        assert turn.tool_calls[0].name == "kb__search"
        assert turn.model == "local/qwen3-30b"

    def test_stored_blocks_are_valid_to_send_back(self) -> None:
        """A text block echoed with the SDK's ``parsed_output`` is a 400 on the next turn."""
        turn = self.llm._to_turn(message({"type": "text", "text": "ok"}), served="claude-opus-5")
        assert turn.content == [{"type": "text", "text": "ok"}]

    def test_a_reply_that_was_only_thinking_still_leaves_valid_content(self) -> None:
        turn = self.llm._to_turn(message(THINKING), served="local/qwen3-30b", keep_thinking=False)
        assert turn.content == [{"type": "text", "text": "(empty reply)"}]


class TestConstruction:
    @pytest.mark.parametrize("key", [None, "", "   "])
    def test_a_missing_or_blank_key_is_a_configuration_error(self, key: str | None) -> None:
        """A blank key in a test environment must not reach the network as a 401."""
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicLLM(api_key=key, model="claude-opus-5")

    def test_without_risk_of_every_call_fails_closed(self) -> None:
        llm = AnthropicLLM(api_key="sk-test", model="claude-opus-5")
        turn = llm._to_turn(message({"type": "tool_use", "id": "t1", "name": "x__y", "input": {}}), served="claude-opus-5")
        assert turn.tool_calls[0].risk is RiskLevel.DESTRUCTIVE

    async def test_a_local_model_without_gateway_is_a_configuration_error(self) -> None:
        llm = AnthropicLLM(api_key="sk-test", model="claude-opus-5")
        with pytest.raises(ConfigurationError, match="gateway_url"):
            await llm.complete(
                system=[], messages=[], tools=None, effort="low", emit=lambda *_: None, model="local/qwen3-30b"
            )
