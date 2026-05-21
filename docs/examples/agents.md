# Agents Examples

Practical, runnable examples for every major agent pattern in Quaerium.

!!! abstract "Prerequisites"
    All examples assume Quaerium v0.4.0+ and a running Ollama instance with `llama3.2`:
    ```bash
    pip install quaerium
    ollama pull llama3.2
    ```

---

## RAGAgent with Custom Tools

A complete, runnable example showing how to combine document retrieval with dynamic tools such as a date provider and a calculator.

=== "Full Example"

    ```python title="examples/agent_rag_example.py" linenums="1"
    from datetime import date

    from quaerium.agents import RAGAgent, tool
    from quaerium.agents.memory import InMemoryConversationMemory
    from quaerium.infra import get_ollama_embedding, get_ollama_llm
    from quaerium.infra.vectorstores import get_qdrant_service
    from quaerium import RagPipeline

    # ---------------------------------------------------------------------------
    # Infrastructure
    # ---------------------------------------------------------------------------
    llm = get_ollama_llm(model="llama3.2")
    embedding = get_ollama_embedding(model="nomic-embed-text")
    vector_store = get_qdrant_service(
        host="localhost",
        port=6333,
        collection_name="contracts",
    )

    # Build the RAG pipeline that backs the agent
    pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=vector_store,
    )

    # Index some documents
    pipeline.index_documents([
        "La sezione 4 scade il 2026-06-30.",
        "I requisiti tecnici sono descritti nell'allegato B.",
        "Il contratto prevede penali del 2% per ogni settimana di ritardo.",
    ])

    # ---------------------------------------------------------------------------
    # Custom tools
    # ---------------------------------------------------------------------------
    @tool(description="Returns today's date in ISO 8601 format (YYYY-MM-DD)")
    def get_today() -> str:
        return date.today().isoformat()

    @tool(description="Calculate the number of days between two ISO dates")
    def days_between(start: str, end: str) -> int:  # (1)!
        from datetime import datetime
        d1 = datetime.fromisoformat(start)
        d2 = datetime.fromisoformat(end)
        return abs((d2 - d1).days)

    # ---------------------------------------------------------------------------
    # Agent
    # ---------------------------------------------------------------------------
    agent = RAGAgent(
        llm=llm,
        rag_pipeline=pipeline,
        extra_tools=[get_today, days_between],
        memory=InMemoryConversationMemory(max_turns=20),
        max_steps=8,
    )

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    response = agent.run(
        "Quali sono le scadenze della sezione 4 e quanti giorni mancano?"
    )

    print("=== ANSWER ===")
    print(response.answer)

    print("\n=== SOURCES ===")
    for source in response.sources:
        print(f"  - {source.section_path}")

    print("\n=== TRACE ===")
    for step in response.trace.steps:
        print(f"\n[Step {step.step_number}]")
        print(f"  Thought: {step.thought}")
        if step.tool_call:
            print(f"  Action: {step.tool_call.tool_name}")
            print(f"  Input: {step.tool_call.arguments}")
            print(f"  Observation: {step.observation}")
        print(f"  Latency: {step.latency_ms:.0f}ms")

    print(f"\nTotal: {response.trace.total_latency_ms:.0f}ms | "
          f"Steps: {len(response.trace.steps)} | "
          f"Success: {response.trace.success}")
    ```

    1. The `@tool` decorator infers `start: str` and `end: str` as required string parameters automatically. No manual schema needed.

=== "Expected Output"

    ```
    === ANSWER ===
    La sezione 4 scade il 2026-06-30. Oggi è il 2026-03-16, quindi mancano
    106 giorni alla scadenza.

    === SOURCES ===
      - sezione_4

    === TRACE ===

    [Step 1]
      Thought: I need to find the deadline for sezione 4 in the documents.
      Action: rag_search
      Input: {"query": "scadenza sezione 4"}
      Observation: La sezione 4 scade il 2026-06-30.
      Latency: 312ms

    [Step 2]
      Thought: I have the deadline. Now I need today's date to calculate days.
      Action: get_today
      Input: {}
      Observation: 2026-03-16
      Latency: 4ms

    [Step 3]
      Thought: I can now calculate the days remaining.
      Action: days_between
      Input: {"start": "2026-03-16", "end": "2026-06-30"}
      Observation: 106
      Latency: 2ms

    Total: 318ms | Steps: 3 | Success: True
    ```

---

## ResearchAgent — Multi-Hop Reasoning

Use `ResearchAgent` to answer questions that span multiple knowledge bases. Each sub-agent becomes a named tool.

=== "Setup"

    ```python title="examples/agent_research_setup.py" linenums="1"
    from quaerium.agents import ResearchAgent, RAGAgent
    from quaerium.infra import get_ollama_embedding, get_ollama_llm
    from quaerium.infra.vectorstores import get_qdrant_service
    from quaerium import RagPipeline

    llm = get_ollama_llm(model="llama3.2")
    embedding = get_ollama_embedding(model="nomic-embed-text")

    # --- Contracts knowledge base ---
    contracts_store = get_qdrant_service(
        host="localhost",
        port=6333,
        collection_name="contracts",
    )
    contracts_pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=contracts_store,
    )
    contracts_agent = RAGAgent(
        llm=llm,
        rag_pipeline=contracts_pipeline,
        max_steps=5,
    )

    # --- Laws knowledge base ---
    laws_store = get_qdrant_service(
        host="localhost",
        port=6333,
        collection_name="laws",
    )
    laws_pipeline = RagPipeline(
        embedding_client=embedding,
        llm_client=llm,
        vector_store=laws_store,
    )
    laws_agent = RAGAgent(
        llm=llm,
        rag_pipeline=laws_pipeline,
        max_steps=5,
    )
    ```

=== "Research Run"

    ```python title="examples/agent_research_run.py" linenums="1"
    from quaerium.agents import ResearchAgent

    researcher = ResearchAgent(
        llm=llm,
        sub_agents={
            "contracts_agent": contracts_agent,  # (1)!
            "laws_agent": laws_agent,
        },
        max_steps=12,
    )

    response = researcher.run(
        "Analizza i requisiti legali e contrattuali per la gara X. "
        "Verifica se le clausole contrattuali rispettano il D.Lgs. 36/2023."
    )

    print(response.answer)

    # Show which sub-agents were consulted
    tools_used = [
        s.tool_call.tool_name
        for s in response.trace.steps
        if s.tool_call
    ]
    print(f"\nSub-agents consulted: {set(tools_used)}")
    ```

    1. The key `"contracts_agent"` becomes the tool name the LLM sees. Use names that clearly identify what each sub-agent knows about.

=== "Trace Analysis"

    ```python title="examples/agent_research_trace.py" linenums="1"
    # Analyse which sub-agents contributed to the answer
    from collections import Counter

    tool_counts = Counter(
        s.tool_call.tool_name
        for s in response.trace.steps
        if s.tool_call
    )

    print("Tool invocation counts:")
    for name, count in tool_counts.most_common():
        print(f"  {name}: {count}")

    # Check if all steps succeeded
    failed_steps = [
        s for s in response.trace.steps
        if s.observation and "error" in s.observation.lower()
    ]
    if failed_steps:
        print(f"\nWarning: {len(failed_steps)} step(s) produced errors")
    else:
        print("\nAll steps completed successfully")
    ```

---

## Custom Tool with ToolDefinition

Use `ToolDefinition` for explicit schema control, retry logic, and fallback behaviour.

=== "Definition"

    ```python title="examples/custom_tool_definition.py" linenums="1"
    import httpx
    from quaerium.agents.tool import ToolDefinition

    def fetch_exchange_rate(base: str, target: str) -> str:
        """Fetch current exchange rate from a public API."""
        url = f"https://api.exchangerate.host/latest"
        params = {"base": base, "symbols": target}
        resp = httpx.get(url, params=params, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"][target]
        return f"1 {base} = {rate:.4f} {target}"

    exchange_tool = ToolDefinition(
        name="get_exchange_rate",
        description=(
            "Fetch the current exchange rate between two currencies. "
            "Example: Action Input: {\"base\": \"EUR\", \"target\": \"USD\"}"
        ),
        fn=fetch_exchange_rate,
        parameters_schema={
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Source currency code (e.g. EUR, USD)"
                },
                "target": {
                    "type": "string",
                    "description": "Target currency code (e.g. USD, GBP)"
                },
            },
            "required": ["base", "target"],
        },
        max_retries=3,  # (1)!
        fallback="Exchange rate service is currently unavailable.",  # (2)!
    )
    ```

    1. On `httpx.HTTPError` or any other exception, the tool retries up to 3 times with a short backoff before returning the fallback string.
    2. The fallback string is returned to the agent as the observation. The agent can then decide to try a different approach.

=== "Usage"

    ```python title="examples/custom_tool_usage.py" linenums="1"
    from quaerium.agents import AgentRuntime
    from quaerium.infra import get_ollama_llm

    llm = get_ollama_llm(model="llama3.2")

    runtime = AgentRuntime(
        llm=llm,
        tools=[exchange_tool],
        max_steps=5,
    )

    response = runtime.run(
        "What is today's EUR to USD exchange rate?"
    )
    print(response.answer)
    ```

=== "Testing with Mock"

    ```python title="examples/custom_tool_mock.py" linenums="1"
    from unittest.mock import patch
    from quaerium.agents import AgentRuntime
    from quaerium.infra import get_ollama_llm

    llm = get_ollama_llm(model="llama3.2")

    def mock_fetch(base: str, target: str) -> str:
        return f"1 {base} = 1.0850 {target} (mocked)"

    mock_tool = ToolDefinition(
        name="get_exchange_rate",
        description="Fetch exchange rate (mock)",
        fn=mock_fetch,
        parameters_schema={
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["base", "target"],
        },
    )

    runtime = AgentRuntime(llm=llm, tools=[mock_tool], max_steps=5)
    response = runtime.run("EUR to USD rate?")
    print(response.answer)
    ```

---

## Custom Observer for Logging and Metrics

=== "Logging Observer"

    ```python title="observers/logging_observer.py" linenums="1"
    import logging
    from quaerium.agents.models import AgentStep, AgentTrace

    logger = logging.getLogger("quaerium.agent")

    class LoggingObserver:
        """Structured logging observer for production use."""

        def on_step_start(self, step: AgentStep) -> None:
            logger.debug(
                "agent_step_start",
                extra={"step": step.step_number}
            )

        def on_step_end(self, step: AgentStep) -> None:
            logger.info(
                "agent_step_end",
                extra={
                    "step": step.step_number,
                    "latency_ms": round(step.latency_ms, 2),
                    "has_tool_call": step.tool_call is not None,
                },
            )

        def on_tool_call(
            self, tool_name: str, kwargs: dict, result: str
        ) -> None:
            logger.info(
                "agent_tool_call",
                extra={
                    "tool": tool_name,
                    "kwargs": kwargs,
                    "result_preview": result[:120],
                },
            )

        def on_agent_finish(self, trace: AgentTrace) -> None:
            logger.info(
                "agent_finish",
                extra={
                    "success": trace.success,
                    "steps": len(trace.steps),
                    "total_latency_ms": round(trace.total_latency_ms, 2),
                    "error": trace.error,
                },
            )
    ```

=== "Metrics Observer"

    ```python title="observers/metrics_observer.py" linenums="1"
    from collections import defaultdict
    from quaerium.agents.models import AgentStep, AgentTrace

    class MetricsObserver:
        """Collects in-process metrics for monitoring."""

        def __init__(self):
            self.step_count = 0
            self.tool_calls: dict[str, int] = defaultdict(int)
            self.total_latency_ms = 0.0
            self.success_count = 0
            self.failure_count = 0

        def on_step_start(self, step: AgentStep) -> None:
            pass

        def on_step_end(self, step: AgentStep) -> None:
            self.step_count += 1
            self.total_latency_ms += step.latency_ms

        def on_tool_call(
            self, tool_name: str, kwargs: dict, result: str
        ) -> None:
            self.tool_calls[tool_name] += 1

        def on_agent_finish(self, trace: AgentTrace) -> None:
            if trace.success:
                self.success_count += 1
            else:
                self.failure_count += 1

        def report(self) -> dict:
            return {
                "step_count": self.step_count,
                "tool_calls": dict(self.tool_calls),
                "avg_latency_ms": (
                    self.total_latency_ms / max(self.step_count, 1)
                ),
                "success_count": self.success_count,
                "failure_count": self.failure_count,
            }
    ```

=== "Usage"

    ```python title="examples/observer_usage.py" linenums="1"
    from quaerium.agents import AgentRuntime
    from quaerium.agents import tool
    from quaerium.infra import get_ollama_llm

    llm = get_ollama_llm(model="llama3.2")

    @tool(description="Returns today's ISO date")
    def get_date() -> str:
        from datetime import date
        return date.today().isoformat()

    logging_obs = LoggingObserver()
    metrics_obs = MetricsObserver()

    runtime = AgentRuntime(
        llm=llm,
        tools=[get_date],
        observers=[logging_obs, metrics_obs],  # (1)!
        max_steps=5,
    )

    runtime.run("What day is today?")
    runtime.run("Is today a weekday?")

    # View aggregated metrics
    print(metrics_obs.report())
    # {'step_count': 4, 'tool_calls': {'get_date': 2},
    #  'avg_latency_ms': 180.5, 'success_count': 2, 'failure_count': 0}
    ```

    1. Both observers receive every callback independently. Observer exceptions are caught and logged but do not interrupt agent execution.

---

## Custom ExecutionStrategy

Implement a custom `ExecutionStrategy` to replace the default ReAct loop with any other prompting strategy.

```python title="examples/custom_strategy.py" linenums="1"
from typing import Any
from quaerium.agents.models import AgentTrace, AgentStep
from datetime import datetime, timezone

class ChainOfThoughtStrategy:
    """
    A simplified strategy that does a single LLM call
    with chain-of-thought reasoning, then extracts the answer.

    This is a skeleton — extend it for your use case.
    """

    def execute(
        self,
        *,
        task: str,
        tools: list[Any],
        memory: Any,
        llm: Any,
        observers: list[Any],
        max_steps: int,
    ) -> AgentTrace:
        start = datetime.now(timezone.utc)

        # Build tool descriptions for the prompt
        tool_block = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )

        # Single prompt with CoT instructions
        prompt = (
            f"You have access to these tools:\n{tool_block}\n\n"
            f"Think step by step, then answer:\n{task}\n\n"
            "Reasoning:"
        )

        # Notify observers
        step = AgentStep(
            step_number=1,
            thought="",
            tool_call=None,
            observation=None,
            timestamp=datetime.now(timezone.utc),
            latency_ms=0.0,
        )
        for obs in observers:
            obs.on_step_start(step)

        # Single LLM call
        raw = llm.generate(prompt)
        latency_ms = (
            (datetime.now(timezone.utc) - start).total_seconds() * 1000
        )

        # Parse answer after "Final Answer:" if present
        if "Final Answer:" in raw:
            answer = raw.split("Final Answer:", 1)[1].strip()
        else:
            answer = raw.strip()

        step.thought = raw
        step.latency_ms = latency_ms

        for obs in observers:
            obs.on_step_end(step)

        trace = AgentTrace(
            task=task,
            steps=[step],
            final_answer=answer,
            success=True,
            total_latency_ms=latency_ms,
            error=None,
        )
        for obs in observers:
            obs.on_agent_finish(trace)

        return trace


# Use the custom strategy
from quaerium.agents import AgentRuntime
from quaerium.infra import get_ollama_llm

runtime = AgentRuntime(
    llm=get_ollama_llm(model="llama3.2"),
    strategy=ChainOfThoughtStrategy(),  # (1)!
    max_steps=1,
)

response = runtime.run("Explain RAG in one sentence.")
print(response.answer)
```

1. Pass any object that satisfies the `ExecutionStrategy` protocol. Because it is `@runtime_checkable`, you can verify with `isinstance(strategy, ExecutionStrategy)`.

!!! note "ExecutionStrategy Protocol"
    ```python
    @runtime_checkable
    class ExecutionStrategy(Protocol):
        def execute(
            self,
            *,
            task: str,
            tools: list,
            memory: Any,
            llm: Any,
            observers: list,
            max_steps: int,
        ) -> AgentTrace: ...
    ```

---

## See Also

- [Agents Guide](../guides/agents.md) — Comprehensive conceptual documentation
- [Agents API Reference](../api/agents/index.md) — Complete class and protocol reference
- `examples/agent_rag_example.py` — Full runnable RAGAgent script
- `examples/agent_research_example.py` — Full runnable ResearchAgent script
