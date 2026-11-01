"""RAG as tools.

The bridge between the two halves of Quaerium: a :class:`~quaerium.rag.RagPipeline`
or a graph store becomes a tool the agent can call, with the same risk model as
any other capability.

Example:
    >>> registry = ToolRegistry()                                   # doctest: +SKIP
    >>> registry.register_many([
    ...     rag_tool(pipeline, namespace="kb"),
    ...     graph_query_tool(neo4j, namespace="kb"),
    ... ])
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from quaerium.agents.tools.base import ToolSpec, tool
from quaerium.agents.types import RiskLevel

_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b|\bCALL\s+(?!db\.)",
    re.IGNORECASE,
)
"""Cypher that writes, or calls a procedure that might (``db.*`` introspection is allowed)."""


def _format_citations(response: Any, *, max_chars: int) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(getattr(response, "citations", []) or [], start=1):
        where = getattr(chunk, "section_path", None) or getattr(chunk, "id", "")
        pages = getattr(chunk, "page_numbers", None) or []
        if pages:
            where = f"{where}, p. {', '.join(str(p) for p in pages)}"
        text = " ".join(str(getattr(chunk, "text", "")).split())
        lines.append(f"[{index}] {where}: {text[:max_chars]}{'…' if len(text) > max_chars else ''}")
    return "\n".join(lines)


def rag_tool(
    pipeline: Any,
    *,
    namespace: str = "kb",
    name: str = "search",
    description: str | None = None,
    top_k: int = 5,
    max_source_chars: int = 500,
) -> ToolSpec:
    """Expose a RAG pipeline as a read-only tool.

    The tool returns the pipeline's answer followed by its numbered sources, so
    the model can quote them. The pipeline's ``run`` is synchronous; it is run
    in a worker thread so the event loop keeps streaming.

    Args:
        pipeline: Anything with ``run(question, *, top_k) -> RagResponse``.
        namespace: Tool namespace; the tool is ``{namespace}__{name}``.
        name: Tool verb.
        description: What the model is told the knowledge base contains. Be
            specific: it is how the model decides to call the tool.
        top_k: Chunks to retrieve.
        max_source_chars: Characters kept per source excerpt.
    """

    async def search(question: str) -> str:
        """Search the knowledge base and answer from it, with numbered sources.

        Args:
            question: A self-contained question; include names and dates from the conversation.
        """
        response = await asyncio.to_thread(pipeline.run, question, top_k=top_k)
        sources = _format_citations(response, max_chars=max_source_chars)
        answer = str(getattr(response, "answer", response))
        return f"{answer}\n\nSources:\n{sources}" if sources else answer

    return tool(namespace=namespace, risk=RiskLevel.READ, name=name, description=description, tags=("rag",))(search)


def graph_query_tool(
    store: Any,
    *,
    namespace: str = "kb",
    name: str = "graph_query",
    description: str | None = None,
    read_only: bool = True,
    max_rows: int = 50,
) -> ToolSpec:
    """Expose a graph store's native query language (Cypher for Neo4j) as a tool.

    Args:
        store: Anything with ``async query(query, parameters) -> list[dict]``,
            e.g. a :class:`~quaerium.core.graphstore.GraphStoreClient`.
        namespace: Tool namespace.
        name: Tool verb.
        description: What the graph contains: node labels, relationships and
            their properties. The model writes better queries the more it knows.
        read_only: Reject writing clauses before they reach the store, and mark
            the tool ``READ``. With False the tool is ``DESTRUCTIVE``: every
            query waits for a human.
        max_rows: Rows returned to the model; the rest are counted, not shown.
    """

    async def graph_query(query: str, parameters: dict[str, Any] | None = None) -> str:
        """Run a Cypher query against the knowledge graph and return the rows as JSON.

        Args:
            query: The Cypher query. Use $name placeholders for values.
            parameters: Values for the placeholders.
        """
        if read_only and _WRITE_CLAUSE.search(query):
            return "Error: this tool is read-only; the query contains a writing clause."
        rows = await store.query(query, parameters or {})
        shown = rows[:max_rows]
        body = json.dumps(shown, default=str, ensure_ascii=False)
        extra = len(rows) - len(shown)
        return f"{body}\n({extra} more rows not shown)" if extra > 0 else body

    risk = RiskLevel.READ if read_only else RiskLevel.DESTRUCTIVE
    return tool(namespace=namespace, risk=risk, name=name, description=description, tags=("rag", "graph"))(
        graph_query
    )


__all__ = ["graph_query_tool", "rag_tool"]
