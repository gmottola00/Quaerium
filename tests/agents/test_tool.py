"""Tests for the agent tool system."""

import pytest

from quaerium.agents.tool import FunctionTool, ToolDefinition, ToolRegistry, tool
from quaerium.agents.protocols import Tool


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool
        def add(x: int, y: int) -> int:
            return x + y

        assert add.name == "add"
        assert isinstance(add, FunctionTool)

    def test_decorator_with_description(self):
        @tool(description="Returns today's date")
        def get_date() -> str:
            return "2026-03-16"

        assert get_date.description == "Returns today's date"
        assert get_date.run() == "2026-03-16"

    def test_decorator_schema_auto_derived(self):
        @tool
        def greet(name: str, loud: bool) -> str:
            return f"Hello, {name}!"

        schema = greet.parameters_schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "loud" in schema["properties"]
        assert "name" in schema["required"]

    def test_run_executes_function(self):
        @tool
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply.run(a=3, b=4)
        assert result == "12"

    def test_run_handles_exception(self):
        @tool
        def fail() -> str:
            raise ValueError("oops")

        result = fail.run()
        assert "Error" in result

    def test_protocol_satisfaction(self):
        @tool(description="test")
        def noop() -> str:
            return "ok"

        assert isinstance(noop, Tool)


class TestToolDefinition:
    def test_basic_run(self):
        td = ToolDefinition(
            name="upper",
            description="Uppercase text",
            fn=lambda text: text.upper(),
            parameters_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        assert td.run(text="hello") == "HELLO"

    def test_fallback_on_error(self):
        td = ToolDefinition(
            name="bad",
            description="always fails",
            fn=lambda: 1 / 0,
            max_retries=2,
            fallback="unavailable",
        )
        result = td.run()
        assert result == "unavailable"

    def test_retry_count(self):
        calls = []

        def flaky() -> str:
            calls.append(1)
            raise RuntimeError("retry")

        td = ToolDefinition(
            name="flaky",
            description="",
            fn=flaky,
            max_retries=3,
            fallback="gave up",
        )
        td.run()
        assert len(calls) == 3


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()

        @tool(description="says hi")
        def hello() -> str:
            return "hi"

        registry.register(hello)
        assert "hello" in registry
        assert registry.get("hello") is hello

    def test_all_returns_all(self):
        registry = ToolRegistry()

        @tool
        def t1() -> str: return "t1"

        @tool
        def t2() -> str: return "t2"

        registry.register(t1)
        registry.register(t2)
        assert len(registry.all()) == 2

    def test_to_prompt_block(self):
        registry = ToolRegistry()

        @tool(description="A useful tool")
        def useful_tool(query: str) -> str:
            return query

        registry.register(useful_tool)
        block = registry.to_prompt_block()
        assert "useful_tool" in block
        assert "A useful tool" in block
