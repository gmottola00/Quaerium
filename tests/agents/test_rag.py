"""RAG as tools: a pipeline and a graph store behind the same risk model as anything else."""

from __future__ import annotations

import threading
from typing import Any

from quaerium.agents.tools.rag import graph_query_tool, rag_tool
from quaerium.agents.types import RiskLevel
from quaerium.rag.models import RagResponse, RetrievedChunk


class FakePipeline:
    """Synchronous, like :class:`~quaerium.rag.RagPipeline`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.thread: str = ""

    def run(self, question: str, *, top_k: int = 5) -> RagResponse:
        self.calls.append((question, top_k))
        self.thread = threading.current_thread().name
        chunk = RetrievedChunk(
            id="c1",
            text="Refunds are issued   within 14 days.",
            section_path="Policy > Refunds",
            metadata={},
            page_numbers=[3],
            source_chunk_id=None,
        )
        return RagResponse(answer="Within 14 days [1].", citations=[chunk])


class FakeGraph:
    def __init__(self, rows: int = 3) -> None:
        self.rows = [{"n": i} for i in range(rows)]
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((query, parameters or {}))
        return self.rows


class TestRagTool:
    def test_spec(self) -> None:
        spec = rag_tool(FakePipeline(), namespace="kb", description="Company policies.")
        assert spec.name == "kb__search"
        assert spec.risk is RiskLevel.READ
        assert spec.description == "Company policies."
        assert spec.input_schema["required"] == ["question"]

    async def test_answer_with_numbered_sources(self) -> None:
        pipeline = FakePipeline()
        result = await rag_tool(pipeline, top_k=3).run("c1", {"question": "refund time?"})

        assert result.ok
        assert result.content.startswith("Within 14 days [1].")
        assert "[1] Policy > Refunds, p. 3: Refunds are issued within 14 days." in result.content
        assert pipeline.calls == [("refund time?", 3)]

    async def test_the_pipeline_runs_off_the_event_loop(self) -> None:
        pipeline = FakePipeline()
        await rag_tool(pipeline).run("c1", {"question": "q"})
        assert pipeline.thread != threading.main_thread().name


class TestGraphQueryTool:
    async def test_read_query_runs(self) -> None:
        store = FakeGraph()
        result = await graph_query_tool(store).run(
            "c1", {"query": "MATCH (n) WHERE n.id = $id RETURN n", "parameters": {"id": 1}}
        )
        assert result.content == '[{"n": 0}, {"n": 1}, {"n": 2}]'
        assert store.queries == [("MATCH (n) WHERE n.id = $id RETURN n", {"id": 1})]

    async def test_read_only_rejects_writes_before_the_store(self) -> None:
        store = FakeGraph()
        for query in ("MATCH (n) DETACH DELETE n", "CREATE (n:X)", "MATCH (n) SET n.a = 1", "CALL apoc.do.it()"):
            result = await graph_query_tool(store).run("c1", {"query": query})
            assert "read-only" in result.content
        assert store.queries == []

    async def test_db_introspection_is_allowed(self) -> None:
        result = await graph_query_tool(FakeGraph(rows=1)).run("c1", {"query": "CALL db.labels()"})
        assert "read-only" not in result.content

    async def test_rows_are_capped(self) -> None:
        result = await graph_query_tool(FakeGraph(rows=10), max_rows=2).run("c1", {"query": "MATCH (n) RETURN n"})
        assert result.content.endswith("(8 more rows not shown)")

    def test_writable_means_destructive(self) -> None:
        assert graph_query_tool(FakeGraph()).risk is RiskLevel.READ
        assert graph_query_tool(FakeGraph(), read_only=False).risk is RiskLevel.DESTRUCTIVE
