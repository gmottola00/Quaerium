"""Tool system for Quaerium Agents.

Three ways to define tools:
1. @tool decorator — zero boilerplate, schema auto-derived from type hints
2. ToolDefinition dataclass — explicit schema, retry, fallback
3. Any object satisfying the Tool Protocol — pure duck typing
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PYTHON_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _build_schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Auto-derive a JSON Schema from a function's type-annotated signature."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        ann = param.annotation
        type_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        json_type = _PYTHON_TO_JSON_TYPE.get(type_name, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class FunctionTool:
    """Wraps a plain function as a Tool-Protocol-compatible object."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        self._fn = fn
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema or _build_schema_from_signature(fn)

    def run(self, **kwargs: Any) -> str:
        try:
            result = self._fn(**kwargs)
            return str(result)
        except Exception as exc:
            logger.warning("Tool '%s' raised: %s", self.name, exc)
            return f"Error: {exc}"

    async def arun(self, **kwargs: Any) -> str:
        if asyncio.iscoroutinefunction(self._fn):
            try:
                result = await self._fn(**kwargs)
                return str(result)
            except Exception as exc:
                logger.warning("Tool '%s' (async) raised: %s", self.name, exc)
                return f"Error: {exc}"
        return await asyncio.get_event_loop().run_in_executor(None, lambda: self.run(**kwargs))

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r})"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    parameters_schema: dict[str, Any] | None = None,
) -> Any:
    """Decorator to turn a plain function into a Tool.

    Usage::

        @tool(description="Returns today's ISO date")
        def get_date() -> str:
            from datetime import date
            return date.today().isoformat()

        # Or without arguments:
        @tool
        def get_date() -> str: ...
    """

    def decorator(f: Callable[..., Any]) -> FunctionTool:
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "").strip()
        return FunctionTool(f, tool_name, tool_desc, parameters_schema)

    if fn is not None:
        return decorator(fn)
    return decorator


@dataclass
class ToolDefinition:
    """Explicit tool definition with full control over schema, retry, and fallback.

    Use this when you need fine-grained control over the tool behaviour.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 1
    fallback: str = "Tool unavailable."

    def run(self, **kwargs: Any) -> str:
        last_error: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                result = self.fn(**kwargs)
                return str(result)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Tool '%s' attempt %d/%d failed: %s",
                    self.name, attempt + 1, self.max_retries, exc,
                )
        return self.fallback if self.fallback else f"Error: {last_error}"

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.get_event_loop().run_in_executor(None, lambda: self.run(**kwargs))


class ToolRegistry:
    """Registry mapping tool names to Tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool_obj: Any) -> None:
        """Register a tool by its name."""
        self._tools[tool_obj.name] = tool_obj

    def get(self, name: str) -> Any | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def all(self) -> dict[str, Any]:
        """Return all registered tools."""
        return dict(self._tools)

    def to_prompt_block(self) -> str:
        """Generate a formatted tool description block for the LLM prompt."""
        lines: list[str] = ["Available tools:"]
        for t in self._tools.values():
            schema_str = json.dumps(t.parameters_schema, ensure_ascii=False)
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  Parameters: {schema_str}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


__all__ = ["FunctionTool", "ToolDefinition", "ToolRegistry", "tool"]
