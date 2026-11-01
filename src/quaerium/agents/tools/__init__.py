"""Tools: the ``@tool`` decorator, the registry, client providers and RAG bridges."""

from __future__ import annotations

from quaerium.agents.tools.base import ToolSpec, build_schema, split_docstring, tool
from quaerium.agents.tools.provider import ClientProvider
from quaerium.agents.tools.rag import graph_query_tool, rag_tool
from quaerium.agents.tools.registry import ToolRegistry

__all__ = [
    "ClientProvider",
    "ToolRegistry",
    "ToolSpec",
    "build_schema",
    "graph_query_tool",
    "rag_tool",
    "split_docstring",
    "tool",
]
