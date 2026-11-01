"""Schema derivation, tool execution and the registry."""

from __future__ import annotations

from typing import Literal

import pytest

from quaerium.agents.exceptions import ToolNotFoundError
from quaerium.agents.tools.base import build_schema, split_docstring, tool
from quaerium.agents.types import RiskLevel, ToolCall


class TestDocstringParsing:
    def test_summary_only(self) -> None:
        summary, args = split_docstring("Do a thing.")
        assert summary == "Do a thing."
        assert args == {}

    def test_args_section(self) -> None:
        summary, args = split_docstring("""Do a thing.

            Args:
                name: Who to do it to.
                count: How many times.

            Returns:
                Nothing useful.
            """)
        assert summary == "Do a thing."
        assert args == {"name": "Who to do it to.", "count": "How many times."}

    def test_wrapped_description_is_joined(self) -> None:
        _, args = split_docstring("""S.

            Args:
                name: A description that
                    continues on the next line.
            """)
        assert args["name"] == "A description that continues on the next line."


class TestSchema:
    def test_types_defaults_and_descriptions(self) -> None:
        async def fn(a: str, b: int = 3, c: bool = False) -> str:
            """S.

            Args:
                a: First.
            """
            return ""

        schema = build_schema(fn)
        assert schema["properties"]["a"] == {"type": "string", "description": "First."}
        assert schema["properties"]["b"] == {"type": "integer", "default": 3}
        assert schema["properties"]["c"]["type"] == "boolean"
        assert schema["required"] == ["a"]
        assert schema["additionalProperties"] is False

    def test_literal_becomes_enum(self) -> None:
        async def fn(mode: Literal["fast", "slow"] = "fast") -> str:
            return ""

        assert build_schema(fn)["properties"]["mode"]["enum"] == ["fast", "slow"]

    def test_optional_unwraps_and_is_not_required(self) -> None:
        async def fn(ref: str | None = None) -> str:
            return ""

        schema = build_schema(fn)
        assert schema["properties"]["ref"]["type"] == "string"
        assert schema["required"] == []

    def test_list_gets_item_type(self) -> None:
        async def fn(names: list[str]) -> str:
            return ""

        assert build_schema(fn)["properties"]["names"] == {"type": "array", "items": {"type": "string"}}


class TestToolSpec:
    def test_name_is_namespaced(self, toy_registry) -> None:
        spec = toy_registry.get("toy__peek")
        assert spec.namespace == "toy"
        assert spec.description == "Look at something."

    def test_sync_function_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be an async function"):

            @tool(namespace="toy", risk=RiskLevel.READ)
            def sync_tool() -> str:  # type: ignore[misc]
                return ""

    async def test_run_returns_result(self, toy_registry) -> None:
        result = await toy_registry.get("toy__peek").run("c1", {"what": "the sky"})
        assert result.ok
        assert result.content == "saw the sky"
        assert result.id == "c1"

    async def test_failure_becomes_a_result_not_an_exception(self, toy_registry) -> None:
        result = await toy_registry.get("toy__explode").run("c2", {})
        assert not result.ok
        assert "RuntimeError" in result.content

    async def test_bad_arguments_become_a_result(self, toy_registry) -> None:
        result = await toy_registry.get("toy__peek").run("c3", {"nope": 1})
        assert not result.ok
        assert "invalid arguments" in result.content


class TestRegistry:
    def test_unknown_name_raises(self, toy_registry) -> None:
        with pytest.raises(ToolNotFoundError):
            toy_registry.get("nope__nope")

    def test_unknown_risk_fails_closed(self, toy_registry) -> None:
        """An unrecognised tool must not sail through the gate as read-only."""
        assert toy_registry.risk_of("nope__nope") is RiskLevel.DESTRUCTIVE

    def test_risk_of_known_tools(self, toy_registry) -> None:
        assert toy_registry.risk_of("toy__peek") is RiskLevel.READ
        assert toy_registry.risk_of("toy__draft") is RiskLevel.WRITE
        assert toy_registry.risk_of("toy__demolish") is RiskLevel.DESTRUCTIVE

    def test_anthropic_definitions_are_stable_and_sorted(self, toy_registry) -> None:
        """A reordered tool list silently invalidates the prompt cache prefix."""
        assert toy_registry.to_anthropic() == toy_registry.to_anthropic()
        assert [d["name"] for d in toy_registry.to_anthropic()] == toy_registry.names()

    def test_by_namespace(self, toy_registry) -> None:
        assert list(toy_registry.by_namespace()) == ["toy"]

    async def test_invoke_unknown_returns_error_result(self, toy_registry) -> None:
        result = await toy_registry.invoke(ToolCall(id="x", name="nope__nope", args={}))
        assert not result.ok
        assert "no tool named" in result.content
