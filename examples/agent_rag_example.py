"""Example: RAGAgent with RagPipeline.

Demonstrates using RAGAgent to answer questions over a document knowledge base
with a custom extra tool.

Usage:
    uv run python examples/agent_rag_example.py
"""

from __future__ import annotations

from datetime import date

from quaerium.agents import RAGAgent, tool
from quaerium.agents.memory import InMemoryConversationMemory

# ---------------------------------------------------------------------------
# 1. Define a custom extra tool using the @tool decorator
# ---------------------------------------------------------------------------

@tool(description="Returns today's ISO date (YYYY-MM-DD)")
def get_date() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# 2. Set up LLM and RagPipeline (replace with real instances)
# ---------------------------------------------------------------------------

class StubLLM:
    """Minimal stub LLM for demonstration — replace with OllamaLLMClient etc."""
    model_name = "stub"

    def generate(self, prompt: str, **kwargs) -> str:
        # In a real scenario this calls your LLM provider
        return "Thought: I have all the information needed.\nFinal Answer: This is a stub answer."


class StubRagPipeline:
    """Stub RagPipeline — replace with a real RagPipeline instance."""

    def run(self, question: str, **kwargs):
        from quaerium.rag.models import RagResponse, RetrievedChunk
        chunk = RetrievedChunk(
            id="doc-1",
            text="Section 4 deadline: 30 days from contract signature.",
            section_path="Section 4",
            metadata={},
            page_numbers=[4],
            source_chunk_id=None,
            score=0.95,
        )
        return RagResponse(
            answer="The deadline in section 4 is 30 days from contract signature.",
            citations=[chunk],
        )


# ---------------------------------------------------------------------------
# 3. Build and run the agent
# ---------------------------------------------------------------------------

def main() -> None:
    llm = StubLLM()
    pipeline = StubRagPipeline()

    agent = RAGAgent(
        llm=llm,
        rag_pipeline=pipeline,
        extra_tools=[get_date],
        memory=InMemoryConversationMemory(max_turns=20),
        max_steps=8,
    )

    question = "What are the deadlines in section 4?"
    print(f"Question: {question}\n")

    response = agent.run(question)

    print(f"Answer: {response.answer}\n")
    print("Trace:")
    for step in response.trace.steps:
        tool_info = ""
        if step.tool_call:
            tool_info = f" → [{step.tool_call.tool_name}]"
        print(f"  [{step.step_number}] {step.thought[:80]}{tool_info}")

    print(f"\nLatency: {response.trace.total_latency_ms:.1f}ms")
    if response.sources:
        print(f"Sources: {[s.section_path or s.id for s in response.sources]}")


if __name__ == "__main__":
    main()
