"""Tests for ReAct prompt building and output parsing."""

import pytest

from quaerium.agents.prompts import build_react_prompt, parse_react_output


class TestBuildReactPrompt:
    def test_includes_task(self):
        prompt = build_react_prompt("What is X?", "No tools", [])
        assert "What is X?" in prompt

    def test_includes_tool_block(self):
        prompt = build_react_prompt("task", "Available tools:\n- search", [])
        assert "search" in prompt

    def test_includes_history(self):
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        prompt = build_react_prompt("new task", "", history)
        assert "prev question" in prompt
        assert "prev answer" in prompt


class TestParseReactOutput:
    def test_parse_final_answer(self):
        output = "Thought: I know the answer.\nFinal Answer: Paris is the capital."
        parsed = parse_react_output(output)
        assert parsed["is_final"] is True
        assert "Paris" in parsed["final_answer"]

    def test_parse_action(self):
        output = (
            "Thought: I should search for this.\n"
            "Action: rag_search\n"
            'Action Input: {"query": "capital of France"}'
        )
        parsed = parse_react_output(output)
        assert parsed["is_final"] is False
        assert parsed["action"] == "rag_search"
        assert parsed["action_input"] == {"query": "capital of France"}

    def test_parse_action_fallback_input(self):
        output = (
            "Thought: search it\n"
            "Action: search\n"
            "Action Input: some free text"
        )
        parsed = parse_react_output(output)
        assert parsed["action_input"] == {"query": "some free text"}

    def test_empty_output_treated_as_final(self):
        parsed = parse_react_output("just a plain response")
        assert parsed["is_final"] is True

    def test_thought_extracted(self):
        output = (
            "Thought: Let me think about this carefully.\n"
            "Action: search\n"
            'Action Input: {"query": "test"}'
        )
        parsed = parse_react_output(output)
        assert "Let me think" in parsed["thought"]
