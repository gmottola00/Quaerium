"""Tests for the ReActExecutor."""

import pytest

from quaerium.agents.executor import ReActExecutor
from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.tool import tool


class MockLLM:
    """LLM that returns pre-scripted responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        return next(self._responses, "Thought: done\nFinal Answer: fallback")


class TestReActExecutor:
    def test_single_step_final_answer(self):
        llm = MockLLM(["Thought: I know.\nFinal Answer: 42"])
        executor = ReActExecutor()
        trace = executor.execute(
            task="What is the answer?",
            tools={},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[],
            max_steps=5,
        )
        assert trace.success is True
        assert trace.final_answer == "42"
        assert len(trace.steps) == 1

    def test_tool_call_then_final_answer(self):
        @tool(description="returns a number")
        def get_number() -> str:
            return "99"

        llm = MockLLM([
            "Thought: check tool\nAction: get_number\nAction Input: {}",
            "Thought: got 99\nFinal Answer: The number is 99",
        ])
        executor = ReActExecutor()
        trace = executor.execute(
            task="What number?",
            tools={"get_number": get_number},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[],
            max_steps=5,
        )
        assert trace.success is True
        assert "99" in trace.final_answer
        assert len(trace.steps) == 2

    def test_max_steps_exceeded(self):
        # LLM keeps calling a tool indefinitely
        @tool(description="infinite loop")
        def loop_tool() -> str:
            return "keep going"

        llm = MockLLM([
            "Thought: loop\nAction: loop_tool\nAction Input: {}",
        ] * 10)
        executor = ReActExecutor()
        trace = executor.execute(
            task="loop",
            tools={"loop_tool": loop_tool},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[],
            max_steps=3,
        )
        assert trace.error == "max_steps_reached"
        assert trace.success is False

    def test_unknown_tool_treated_as_final(self):
        llm = MockLLM([
            "Thought: call unknown\nAction: nonexistent\nAction Input: {}",
        ])
        executor = ReActExecutor()
        trace = executor.execute(
            task="test",
            tools={},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[],
            max_steps=5,
        )
        assert trace.success is True

    def test_observer_called(self):
        events = []

        class TestObserver:
            def on_step_start(self, step): events.append(("start", step.step_number))
            def on_step_end(self, step): events.append(("end", step.step_number))
            def on_agent_finish(self, trace): events.append(("finish",))

        llm = MockLLM(["Thought: done\nFinal Answer: hello"])
        executor = ReActExecutor()
        executor.execute(
            task="test",
            tools={},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[TestObserver()],
            max_steps=5,
        )
        assert ("start", 1) in events
        assert ("end", 1) in events
        assert ("finish",) in events

    def test_trace_latency_recorded(self):
        llm = MockLLM(["Thought: quick\nFinal Answer: done"])
        executor = ReActExecutor()
        trace = executor.execute(
            task="speed test",
            tools={},
            memory=InMemoryConversationMemory(),
            llm=llm,
            observers=[],
            max_steps=5,
        )
        assert trace.total_latency_ms >= 0
        assert trace.steps[0].latency_ms >= 0
