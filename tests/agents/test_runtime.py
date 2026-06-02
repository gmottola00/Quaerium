"""Tests for AgentRuntime."""

import pytest

from quaerium.agents.runtime import AgentRuntime
from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.tool import tool


class MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        return next(self._responses, "Thought: done\nFinal Answer: default")


class TestAgentRuntime:
    def test_basic_run(self):
        llm = MockLLM(["Thought: simple\nFinal Answer: hello world"])
        runtime = AgentRuntime(llm=llm)
        response = runtime.run("Say hello")
        assert "hello world" in response.answer

    def test_memory_updated_after_run(self):
        llm = MockLLM(["Thought: ok\nFinal Answer: done"])
        mem = InMemoryConversationMemory()
        runtime = AgentRuntime(llm=llm, memory=mem)
        runtime.run("test task")
        history = mem.get_history()
        assert any(m["role"] == "user" and "test task" in m["content"] for m in history)
        assert any(m["role"] == "assistant" for m in history)

    def test_reset_memory(self):
        llm = MockLLM(["Thought: ok\nFinal Answer: done"] * 5)
        runtime = AgentRuntime(llm=llm)
        runtime.run("task 1")
        runtime.reset_memory()
        assert runtime.memory.get_history() == []

    def test_register_tool_dynamic(self):
        @tool(description="dynamic tool")
        def dyn() -> str:
            return "dynamic"

        llm = MockLLM([
            "Thought: use dyn\nAction: dyn\nAction Input: {}",
            "Thought: got result\nFinal Answer: dynamic",
        ])
        runtime = AgentRuntime(llm=llm)
        runtime.register_tool(dyn)
        response = runtime.run("test dynamic")
        assert "dynamic" in response.answer

    def test_tools_passed_at_init(self):
        @tool(description="counter tool")
        def counter() -> str:
            return "count=1"

        llm = MockLLM([
            "Thought: count\nAction: counter\nAction Input: {}",
            "Thought: got it\nFinal Answer: count=1",
        ])
        runtime = AgentRuntime(llm=llm, tools=[counter])
        response = runtime.run("count")
        assert "count=1" in response.answer

    def test_response_has_trace(self):
        llm = MockLLM(["Thought: done\nFinal Answer: finished"])
        runtime = AgentRuntime(llm=llm)
        response = runtime.run("task")
        assert response.trace is not None
        assert len(response.trace.steps) >= 1
        assert response.trace.task == "task"
