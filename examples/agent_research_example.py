"""Example: ResearchAgent for multi-hop research.

Demonstrates composing multiple specialized agents (each as a sub-agent tool)
inside a ResearchAgent that can delegate queries to them.

Usage:
    uv run python examples/agent_research_example.py
"""

from __future__ import annotations

from quaerium.agents import RAGAgent, ResearchAgent
from quaerium.agents.memory import InMemoryConversationMemory


# ---------------------------------------------------------------------------
# Stubs — replace with real LLM + pipelines in production
# ---------------------------------------------------------------------------

class ReasoningLLM:
    """Stub LLM that produces a multi-hop research answer."""
    model_name = "stub-reasoner"
    _step = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self._step += 1
        if self._step == 1:
            return (
                "Thought: I should check the contracts knowledge base first.\n"
                "Action: contracts_agent\n"
                'Action Input: {"query": "payment terms"}'
            )
        return (
            "Thought: I have enough information to answer.\n"
            "Final Answer: Payment terms are 30 days net, as per contract section 5."
        )


class KbLLM:
    model_name = "stub-kb"

    def generate(self, prompt: str, **kwargs) -> str:
        return "Thought: Found it.\nFinal Answer: Payment terms: 30 days net."


class StubKbPipeline:
    def run(self, question: str, **kwargs):
        from quaerium.rag.models import RagResponse, RetrievedChunk
        return RagResponse(
            answer="Payment terms: 30 days net.",
            citations=[
                RetrievedChunk(
                    id="kb-1", text="Payment: 30 days net.",
                    section_path="Section 5", metadata={},
                    page_numbers=[5], source_chunk_id=None, score=0.9,
                )
            ],
        )


# ---------------------------------------------------------------------------
# Build agents
# ---------------------------------------------------------------------------

def main() -> None:
    # Specialized RAG agent for contracts knowledge base
    contracts_agent = RAGAgent(
        llm=KbLLM(),
        rag_pipeline=StubKbPipeline(),
        memory=InMemoryConversationMemory(max_turns=5),
        max_steps=4,
    )

    # Research agent that orchestrates sub-agents
    researcher = ResearchAgent(
        llm=ReasoningLLM(),
        sub_agents={
            "contracts_agent": contracts_agent,
        },
        memory=InMemoryConversationMemory(max_turns=10),
        max_steps=8,
    )

    task = "What are the payment terms in the contract?"
    print(f"Research task: {task}\n")

    response = researcher.run(task)

    print(f"Final answer: {response.answer}\n")
    print("Research trace:")
    for step in response.trace.steps:
        tool_info = f" → [{step.tool_call.tool_name}]" if step.tool_call else ""
        print(f"  [{step.step_number}] {step.thought[:80]}{tool_info}")

    print(f"\nTotal latency: {response.trace.total_latency_ms:.1f}ms")


if __name__ == "__main__":
    main()
