"""EvaluationAgent — agent that orchestrates RAG evaluation tasks.

Pre-registers tools wrapping RetrievalEvaluator and GenerationEvaluator.
"""

from __future__ import annotations

import json
from typing import Any

from quaerium.agents.memory import InMemoryConversationMemory
from quaerium.agents.models import AgentResponse
from quaerium.agents.runtime import AgentRuntime
from quaerium.agents.tool import FunctionTool


class EvaluationAgent:
    """Agent that drives evaluation workflows using injected evaluators.

    Pre-registered tools:
    - run_retrieval_eval: Evaluate retrieval quality
    - run_generation_eval: Evaluate generation quality

    Args:
        llm: LLMClient for agent reasoning
        retrieval_evaluator: RetrievalEvaluator instance (optional)
        generation_evaluator: GenerationEvaluator instance (optional)
        extra_tools: Additional Tool-compatible objects
        memory: AgentMemory instance
        observers: List of AgentObserver instances
        max_steps: Maximum ReAct iterations
    """

    def __init__(
        self,
        *,
        llm: Any,
        retrieval_evaluator: Any | None = None,
        generation_evaluator: Any | None = None,
        extra_tools: list[Any] | None = None,
        memory: Any | None = None,
        observers: list[Any] | None = None,
        max_steps: int = 10,
    ) -> None:
        tools = self._build_tools(retrieval_evaluator, generation_evaluator) + (extra_tools or [])
        self._runtime = AgentRuntime(
            llm=llm,
            tools=tools,
            memory=memory or InMemoryConversationMemory(),
            observers=observers,
            max_steps=max_steps,
        )

    def _build_tools(
        self,
        retrieval_eval: Any | None,
        generation_eval: Any | None,
    ) -> list[FunctionTool]:
        tools: list[FunctionTool] = []

        if retrieval_eval is not None:
            def run_retrieval_eval(
                query: str,
                retrieved_doc_ids: str,
                relevant_doc_ids: str,
            ) -> str:
                """Evaluate retrieval quality. Pass doc IDs as comma-separated strings."""
                try:
                    retrieved = [
                        {"id": d.strip(), "score": 1.0}
                        for d in retrieved_doc_ids.split(",")
                        if d.strip()
                    ]
                    relevant = [d.strip() for d in relevant_doc_ids.split(",") if d.strip()]
                    metrics = retrieval_eval.evaluate_retrieval(
                        query=query,
                        retrieved_docs=retrieved,
                        relevant_doc_ids=relevant,
                    )
                    return json.dumps({
                        "precision": getattr(metrics, "precision_at_k", {}),
                        "recall": getattr(metrics, "recall_at_k", {}),
                        "mrr": getattr(metrics, "mrr", None),
                        "ndcg": getattr(metrics, "ndcg", None),
                    }, default=str)
                except Exception as exc:
                    return f"Retrieval evaluation error: {exc}"

            tools.append(FunctionTool(
                fn=run_retrieval_eval,
                name="run_retrieval_eval",
                description=(
                    "Evaluate retrieval quality for a query. "
                    "Provide query, retrieved_doc_ids (comma-separated), "
                    "and relevant_doc_ids (comma-separated ground truth)."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "retrieved_doc_ids": {"type": "string"},
                        "relevant_doc_ids": {"type": "string"},
                    },
                    "required": ["query", "retrieved_doc_ids", "relevant_doc_ids"],
                },
            ))

        if generation_eval is not None:
            def run_generation_eval(
                question: str,
                generated_answer: str,
                context: str,
                reference_answer: str = "",
            ) -> str:
                """Evaluate generation quality. Pass context as a single string."""
                try:
                    context_list = [c.strip() for c in context.split("|||") if c.strip()]
                    metrics = generation_eval.evaluate_answer(
                        question=question,
                        generated_answer=generated_answer,
                        context=context_list,
                        reference_answer=reference_answer or None,
                    )
                    return json.dumps({
                        "relevance": getattr(metrics, "relevance_score", None),
                        "faithfulness": getattr(metrics, "faithfulness_score", None),
                        "hallucination": getattr(metrics, "hallucination_score", None),
                    }, default=str)
                except Exception as exc:
                    return f"Generation evaluation error: {exc}"

            tools.append(FunctionTool(
                fn=run_generation_eval,
                name="run_generation_eval",
                description=(
                    "Evaluate generation quality. Provide question, generated_answer, "
                    "context (chunks separated by '|||'), and optional reference_answer."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "generated_answer": {"type": "string"},
                        "context": {"type": "string"},
                        "reference_answer": {"type": "string"},
                    },
                    "required": ["question", "generated_answer", "context"],
                },
            ))

        return tools

    def run(self, task: str) -> AgentResponse:
        """Run the evaluation agent on a task."""
        return self._runtime.run(task)

    def reset_memory(self) -> None:
        self._runtime.reset_memory()


__all__ = ["EvaluationAgent"]
