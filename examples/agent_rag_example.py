"""An agent that answers from a knowledge base and asks before it deletes.

Runs offline: the model is scripted and the RAG pipeline is a stand-in, so there
is no API key and no vector store to set up. Swap in ``AnthropicLLM`` and a real
``RagPipeline`` and nothing else changes.

    pip install 'quaerium[agents]'
    python examples/agent_rag_example.py
"""

from __future__ import annotations

import asyncio

from quaerium.agents import AgentRunner, AuditLog, PolicyGate, RiskLevel, ToolRegistry, build_graph, rag_tool, tool
from quaerium.agents.testing import ScriptedLLM
from quaerium.rag.models import RagResponse, RetrievedChunk

# --- the knowledge base ----------------------------------------------------


class HandbookPipeline:
    """Stands in for a RagPipeline: same ``run(question, *, top_k)`` shape."""

    def run(self, question: str, *, top_k: int = 5) -> RagResponse:
        chunk = RetrievedChunk(
            id="handbook-12",
            text="Refunds are issued within 14 days of the request, to the original payment method.",
            section_path="Handbook > Billing > Refunds",
            metadata={},
            page_numbers=[12],
            source_chunk_id=None,
        )
        return RagResponse(answer="Refunds are issued within 14 days [1].", citations=[chunk])


# --- an app tool, with a risk level ----------------------------------------

TICKETS = {"T-1": "Refund for order 881", "T-2": "Password reset"}


@tool(namespace="tickets", risk=RiskLevel.DESTRUCTIVE)
async def close(ticket_id: str) -> str:
    """Close a support ticket.

    Args:
        ticket_id: The ticket to close, e.g. ``T-1``.
    """
    title = TICKETS.pop(ticket_id)
    return f"Closed {ticket_id}: {title}"


# --- wiring ------------------------------------------------------------------


async def main() -> None:
    registry = ToolRegistry()
    registry.register_many([rag_tool(HandbookPipeline(), namespace="kb", description="The support handbook."), close])

    # What a model would do with "how long do refunds take? then close T-1".
    llm = ScriptedLLM(
        [
            ScriptedLLM.call("kb__search", {"question": "How long do refunds take?"}),
            ScriptedLLM.call("tickets__close", {"ticket_id": "T-1"}, risk=RiskLevel.DESTRUCTIVE),
            ScriptedLLM.say("Refunds take 14 days (handbook p. 12). T-1 is closed."),
        ]
    )

    audit = AuditLog()
    graph = build_graph(
        llm=llm,
        registry=registry,
        system=[{"type": "text", "text": "You are a support assistant. Cite the handbook."}],
        policy=PolicyGate(),
        **audit.hooks(),
    )
    runner = AgentRunner(graph=graph, audit=audit)

    print("> how long do refunds take? then close T-1")
    async for event in runner.start("demo", "how long do refunds take? then close T-1", run_id="run_demo"):
        print(f"  {event['type']:<18} {event.get('name') or event.get('tool') or event.get('text') or ''}")

    # The run is suspended at the gate. A UI would show event["why"] and two buttons.
    print("\n> (user approves)")
    async for event in runner.resume("demo", "run_demo", approved=True):
        print(f"  {event['type']:<18} {event.get('name') or event.get('text') or ''}")

    run = audit.get("run_demo") or {}
    print(f"\naudit: {len(run['tool_calls'])} tool calls, approvals: {[a['approved'] for a in run['approvals']]}")
    print(f"tickets left: {sorted(TICKETS)}")


if __name__ == "__main__":
    asyncio.run(main())
