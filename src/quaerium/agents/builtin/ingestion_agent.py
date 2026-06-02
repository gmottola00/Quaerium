"""IngestionAgent — agent that automates document ingestion tasks.

Pre-registers tools for parsing documents and inspecting supported formats.
"""

from __future__ import annotations

from typing import Any

from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.models import AgentResponse
from quaerium.agents.runtime import AgentRuntime
from quaerium.agents.tool import FunctionTool

_SUPPORTED_FORMATS = [".pdf", ".docx", ".doc"]


class IngestionAgent:
    """Agent that wraps an IngestionService to automate document processing.

    Pre-registered tools:
    - parse_document: Parse a document file path
    - list_formats: List supported file formats
    - check_status: Check if a file is parseable

    Args:
        llm: LLMClient for agent reasoning
        ingestion_service: IngestionService instance
        extra_tools: Additional Tool-compatible objects
        memory: AgentMemory instance
        observers: List of AgentObserver instances
        max_steps: Maximum ReAct iterations
    """

    def __init__(
        self,
        *,
        llm: Any,
        ingestion_service: Any,
        extra_tools: list[Any] | None = None,
        memory: Any | None = None,
        observers: list[Any] | None = None,
        max_steps: int = 8,
    ) -> None:
        tools = self._build_tools(ingestion_service) + (extra_tools or [])
        self._runtime = AgentRuntime(
            llm=llm,
            tools=tools,
            memory=memory or InMemoryConversationMemory(),
            observers=observers,
            max_steps=max_steps,
        )

    def _build_tools(self, service: Any) -> list[FunctionTool]:
        def parse_document(file_path: str) -> str:
            """Parse a document at the given file path and return a summary."""
            try:
                result = service.parse_document(file_path)
                pages = result.get("total_pages", 0)
                lang = result.get("language", "unknown")
                filename = result.get("filename", file_path)
                return (
                    f"Parsed '{filename}': {pages} page(s), language={lang}. "
                    f"doc_id={result.get('doc_id', 'N/A')}"
                )
            except Exception as exc:
                return f"Error parsing document: {exc}"

        def list_formats() -> str:
            """List supported document formats for ingestion."""
            return f"Supported formats: {', '.join(_SUPPORTED_FORMATS)}"

        def check_status(file_path: str) -> str:
            """Check if a file exists and is in a supported format."""
            from pathlib import Path
            p = Path(file_path)
            if not p.exists():
                return f"File not found: {file_path}"
            if p.suffix.lower() not in _SUPPORTED_FORMATS:
                return f"Unsupported format '{p.suffix}'. Supported: {', '.join(_SUPPORTED_FORMATS)}"
            return f"File '{p.name}' is ready for ingestion ({p.suffix})."

        return [
            FunctionTool(
                fn=parse_document,
                name="parse_document",
                description="Parse a document file (PDF or DOCX) and return a summary.",
                parameters_schema={
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            ),
            FunctionTool(
                fn=list_formats,
                name="list_formats",
                description="List all supported document formats for ingestion.",
                parameters_schema={"type": "object", "properties": {}, "required": []},
            ),
            FunctionTool(
                fn=check_status,
                name="check_status",
                description="Check if a file exists and is supported for ingestion.",
                parameters_schema={
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            ),
        ]

    def run(self, task: str) -> AgentResponse:
        """Run the ingestion agent on a task."""
        return self._runtime.run(task)

    def reset_memory(self) -> None:
        self._runtime.reset_memory()


__all__ = ["IngestionAgent"]
