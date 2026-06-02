"""RAGAgent — agent that uses RagPipeline as a retrieval tool.

Pre-registers a `rag_search` tool that wraps a RagPipeline instance.
Additional custom tools can be passed via `extra_tools`.
"""

from __future__ import annotations

from typing import Any

from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.models import AgentResponse
from quaerium.agents.runtime import AgentRuntime
from quaerium.agents.tool import FunctionTool


class RAGAgent:
    """Agent that wraps a RagPipeline as a search tool.

    Args:
        llm: LLMClient instance for agent reasoning
        rag_pipeline: RagPipeline instance to wrap as `rag_search` tool
        extra_tools: Additional Tool-compatible objects to register
        memory: AgentMemory instance (default: InMemoryConversationMemory)
        observers: List of AgentObserver instances
        max_steps: Maximum ReAct iterations (default: 8)
    """

    def __init__(
        self,
        *,
        llm: Any,
        rag_pipeline: Any,
        extra_tools: list[Any] | None = None,
        memory: Any | None = None,
        observers: list[Any] | None = None,
        max_steps: int = 8,
    ) -> None:
        self._rag_pipeline = rag_pipeline
        rag_tool = self._build_rag_tool(rag_pipeline)
        tools = [rag_tool] + (extra_tools or [])
        self._runtime = AgentRuntime(
            llm=llm,
            tools=tools,
            memory=memory or InMemoryConversationMemory(),
            observers=observers,
            max_steps=max_steps,
        )

    def _build_rag_tool(self, pipeline: Any) -> FunctionTool:
        """Wrap the RagPipeline.run() method as a FunctionTool."""

        def rag_search(query: str) -> str:
            """Search the knowledge base using RAG pipeline."""
            response = pipeline.run(query)
            # Format citations into the result string
            answer = response.answer
            citations = getattr(response, "citations", [])
            if citations:
                sources = ", ".join(
                    c.section_path or c.id for c in citations[:3]
                )
                return f"{answer}\n[Sources: {sources}]"
            return answer

        return FunctionTool(
            fn=rag_search,
            name="rag_search",
            description=(
                "Search the knowledge base for information relevant to the query. "
                "Returns an answer with source citations."
            ),
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        )

    def run(self, task: str) -> AgentResponse:
        """Run the RAG agent on a task and return an AgentResponse.

        The response includes `sources` populated from the last rag_search call.

        Args:
            task: Natural-language question or task

        Returns:
            AgentResponse with answer, trace, and retrieved chunks as sources
        """
        response = self._runtime.run(task)
        # Attach retrieved chunks from the last rag call if available
        response.sources = self._extract_last_sources(response)
        return response

    def _extract_last_sources(self, response: AgentResponse) -> list[Any]:
        """Extract retrieved chunks from the last rag_search step in the trace."""
        for step in reversed(response.trace.steps):
            if step.tool_call and step.tool_call.tool_name == "rag_search":
                query = step.tool_call.arguments.get("query", "")
                if query:
                    try:
                        rag_response = self._rag_pipeline.run(query)
                        return getattr(rag_response, "citations", [])
                    except Exception:
                        pass
        return []

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self._runtime.reset_memory()


__all__ = ["RAGAgent"]
