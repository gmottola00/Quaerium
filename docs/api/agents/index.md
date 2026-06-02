# Agents API Reference

Complete reference for all classes, protocols, and data models in the `quaerium.agents` module.

!!! abstract "Module path"
    ```python
    from quaerium.agents import (
        AgentRuntime, RAGAgent, IngestionAgent,
        EvaluationAgent, ResearchAgent, tool,
    )
    from quaerium.agents.tool import ToolDefinition, ToolRegistry
    from quaerium.agents.memory import (
        InMemoryConversationMemory, VectorLongTermMemory
    )
    from quaerium.agents.models import (
        ToolCall, AgentStep, AgentTrace, AgentResponse
    )
    ```

---

## Module Exports

| Name | Kind | Module | Description |
|------|------|--------|-------------|
| `tool` | Decorator | `agents` | Auto-schema tool decorator |
| `AgentRuntime` | Class | `agents` | Central orchestrator |
| `RAGAgent` | Class | `agents` | RAG-powered built-in agent |
| `IngestionAgent` | Class | `agents` | Document ingestion agent |
| `EvaluationAgent` | Class | `agents` | Evaluation metrics agent |
| `ResearchAgent` | Class | `agents` | Multi-hop research agent |
| `ToolDefinition` | Class | `agents.tool` | Explicit tool schema definition |
| `ToolRegistry` | Class | `agents.tool` | Tool collection with prompt generation |
| `FunctionTool` | Class | `agents.tool` | Callable wrapper for the Tool protocol |
| `InMemoryConversationMemory` | Class | `agents.memory` | Sliding-window in-memory memory |
| `VectorLongTermMemory` | Class | `agents.memory` | Vector-backed semantic memory |
| `ReActExecutor` | Class | `agents.executor` | Default ReAct execution strategy |
| `ToolCall` | Dataclass | `agents.models` | Tool invocation record |
| `AgentStep` | Dataclass | `agents.models` | Single ReAct iteration |
| `AgentTrace` | Dataclass | `agents.models` | Full execution trace |
| `AgentResponse` | Dataclass | `agents.models` | High-level response object |
| `Tool` | Protocol | `agents.protocols` | Tool interface |
| `AgentMemory` | Protocol | `agents.protocols` | Memory interface |
| `AgentObserver` | Protocol | `agents.protocols` | Observer hook interface |
| `ExecutionStrategy` | Protocol | `agents.protocols` | Execution strategy interface |

---

## Protocols

<div class="grid cards" markdown>

-   :material-tools: **Tool**

    ---

    `@runtime_checkable` protocol for agent tools. Any object with `name`, `description`, `parameters_schema`, `run`, and `arun` satisfies this protocol.

-   :material-memory: **AgentMemory**

    ---

    `@runtime_checkable` protocol for conversation memory. Requires `add_message`, `get_history`, and `clear`.

-   :material-eye-outline: **AgentObserver**

    ---

    `@runtime_checkable` protocol for execution hooks. Requires `on_step_start`, `on_step_end`, `on_tool_call`, and `on_agent_finish`.

-   :material-strategy: **ExecutionStrategy**

    ---

    `@runtime_checkable` protocol for execution loops. Requires a single `execute(*, task, tools, memory, llm, observers, max_steps)` method.

</div>

### Tool Protocol

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters_schema: dict[str, Any]

    def run(self, **kwargs: Any) -> str:
        """Synchronous tool execution. Must return a string observation."""
        ...

    async def arun(self, **kwargs: Any) -> str:
        """Asynchronous tool execution."""
        ...
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique tool identifier used in `Action:` lines |
| `description` | `str` | Human-readable description shown to the LLM |
| `parameters_schema` | `dict` | JSON Schema for `Action Input:` validation |

### AgentMemory Protocol

```python
@runtime_checkable
class AgentMemory(Protocol):
    def add_message(self, role: str, content: str) -> None:
        """Append a message. role is typically 'user' or 'assistant'."""
        ...

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history as a list of role/content dicts."""
        ...

    def clear(self) -> None:
        """Reset the memory to an empty state."""
        ...
```

### AgentObserver Protocol

```python
@runtime_checkable
class AgentObserver(Protocol):
    def on_step_start(self, step: AgentStep) -> None:
        """Called immediately before a step begins executing."""
        ...

    def on_step_end(self, step: AgentStep) -> None:
        """Called immediately after a step finishes (including tool call)."""
        ...

    def on_tool_call(
        self, tool_name: str, kwargs: dict, result: str
    ) -> None:
        """Called after each tool invocation with its inputs and output."""
        ...

    def on_agent_finish(self, trace: AgentTrace) -> None:
        """Called once when the agent produces its final answer or fails."""
        ...
```

### ExecutionStrategy Protocol

```python
@runtime_checkable
class ExecutionStrategy(Protocol):
    def execute(
        self,
        *,
        task: str,
        tools: list[Tool],
        memory: AgentMemory,
        llm: Any,
        observers: list[AgentObserver],
        max_steps: int,
    ) -> AgentTrace:
        """Run the agent loop and return a complete execution trace."""
        ...
```

---

## Data Models

### ToolCall

Captures a single tool invocation as parsed from the LLM output.

```python
from quaerium.agents.models import ToolCall

@dataclass(frozen=True)
class ToolCall:
    tool_name: str       # Matches Tool.name
    arguments: dict      # Parsed from Action Input JSON block
    raw_llm_output: str  # Verbatim LLM text before parsing
```

**Example:**

```python
# After parsing:
# Action: get_date
# Action Input: {}
tc = ToolCall(
    tool_name="get_date",
    arguments={},
    raw_llm_output="Action: get_date\nAction Input: {}",
)
```

### AgentStep

Represents one full iteration of the ReAct loop.

```python
from quaerium.agents.models import AgentStep
from datetime import datetime

@dataclass
class AgentStep:
    step_number: int            # 1-based iteration counter
    thought: str                # LLM reasoning (Thought: block)
    tool_call: ToolCall | None  # None for Final Answer steps
    observation: str | None     # Tool return value, or None
    timestamp: datetime         # UTC time this step started
    latency_ms: float           # Wall-clock duration in milliseconds
```

!!! note "Final Answer steps"
    When the LLM produces `Final Answer:`, `tool_call` and `observation` are both `None`. The thought contains the full reasoning text.

### AgentTrace

The complete execution record for one `AgentRuntime.run()` call.

```python
from quaerium.agents.models import AgentTrace

@dataclass
class AgentTrace:
    task: str                    # Original task string passed to run()
    steps: list[AgentStep]       # Ordered list of all steps
    final_answer: str | None     # Extracted from Final Answer: block
    success: bool                # True iff Final Answer was reached
    total_latency_ms: float      # Sum of all step latencies
    error: str | None            # Exception message if success=False
```

### AgentResponse

High-level response object returned by all agent `run()` methods.

```python
from quaerium.agents.models import AgentResponse

@dataclass
class AgentResponse:
    answer: str          # Convenience accessor for trace.final_answer
    trace: AgentTrace    # Full execution trace
    sources: list        # Citations (populated by RAGAgent only)
```

---

## Tool System

### @tool decorator

```python
from quaerium.agents import tool
```

Wraps any callable as a `FunctionTool` by deriving the JSON Schema from `inspect.signature()`.

**Signature:**

```python
def tool(
    description: str,
) -> Callable[[Callable], FunctionTool]: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | Human-readable description injected into the LLM prompt |

**Supported type mappings:**

| Python type | JSON Schema type |
|-------------|-----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| Untyped | `"string"` (fallback) |

**Example:**

```python
@tool(description="Return the square root of a number")
def sqrt(x: float) -> float:
    import math
    return math.sqrt(x)

print(sqrt.name)                # "sqrt"
print(sqrt.description)         # "Return the square root of a number"
print(sqrt.parameters_schema)   # {"type": "object", "properties": {"x": ...}, ...}
print(sqrt.run(x=9.0))          # "3.0"
```

### FunctionTool

Internal class produced by the `@tool` decorator. Satisfies the `Tool` protocol.

```python
from quaerium.agents.tool import FunctionTool

@dataclass
class FunctionTool:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    _fn: Callable

    def run(self, **kwargs: Any) -> str: ...
    async def arun(self, **kwargs: Any) -> str: ...
```

!!! note
    You rarely need to instantiate `FunctionTool` directly. Use the `@tool` decorator instead.

### ToolDefinition

Explicit tool definition with full control over schema, retry, and fallback.

```python
from quaerium.agents.tool import ToolDefinition

@dataclass
class ToolDefinition:
    name: str
    description: str
    fn: Callable
    parameters_schema: dict[str, Any]
    max_retries: int = 0
    fallback: str = ""
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Unique tool name |
| `description` | `str` | required | LLM-visible description |
| `fn` | `Callable` | required | The function to invoke |
| `parameters_schema` | `dict` | required | Full JSON Schema object |
| `max_retries` | `int` | `0` | Number of retry attempts on failure |
| `fallback` | `str` | `""` | String returned when all retries are exhausted |

### ToolRegistry

Manages a collection of tools and generates prompt blocks.

```python
from quaerium.agents.tool import ToolRegistry

registry = ToolRegistry()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(tool: Tool) -> None` | Add a tool to the registry |
| `get` | `(name: str) -> Tool \| None` | Retrieve a tool by name |
| `to_prompt_block` | `() -> str` | Generate formatted text for LLM system prompt |
| `__len__` | `() -> int` | Number of registered tools |
| `__iter__` | `() -> Iterator[Tool]` | Iterate over all tools |

**Example:**

```python
registry = ToolRegistry()
registry.register(get_date)
registry.register(search_docs)

print(len(registry))          # 2
print(registry.to_prompt_block())
# Available tools:
# - get_date: Returns today's ISO date
#   Parameters: {}
# - search_docs: Search documents by keyword
#   Parameters: {"query": "string", "top_k": "integer"}
```

---

## Memory

### InMemoryConversationMemory

```python
from quaerium.agents.memory import InMemoryConversationMemory

mem = InMemoryConversationMemory(max_turns: int = 20)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_turns` | `int` | `20` | Maximum number of messages to retain |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_message` | `(role: str, content: str) -> None` | Append a message |
| `get_history` | `() -> list[dict[str, str]]` | Return all retained messages |
| `clear` | `() -> None` | Erase all messages |
| `__len__` | `() -> int` | Current number of messages |

!!! warning "Sliding window"
    When `len(memory) > max_turns`, the oldest message is dropped. This means the agent loses the earliest turns of the conversation. Set `max_turns` high enough to cover your expected conversation depth.

### VectorLongTermMemory

```python
from quaerium.agents.memory import VectorLongTermMemory

mem = VectorLongTermMemory(
    vector_store: VectorStoreClient,
    embedding_client: EmbeddingClient,
    top_k: int = 5,
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_store` | `VectorStoreClient` | required | Backing vector store |
| `embedding_client` | `EmbeddingClient` | required | Embedding provider |
| `top_k` | `int` | `5` | Number of semantically relevant messages to retrieve |

**Methods:** Same interface as `AgentMemory` protocol (`add_message`, `get_history`, `clear`).

---

## Runtime and Executor

### AgentRuntime

```python
from quaerium.agents import AgentRuntime

runtime = AgentRuntime(
    llm: LLMClient,
    tools: list[Tool] = [],
    memory: AgentMemory | None = None,
    strategy: ExecutionStrategy | None = None,
    observers: list[AgentObserver] = [],
    max_steps: int = 10,
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `run` | `(task: str) -> AgentResponse` | `AgentResponse` | Execute the agent on a single task |

!!! note "Memory default"
    If `memory=None`, `AgentRuntime` initialises a fresh `InMemoryConversationMemory(max_turns=20)`. The check is `memory is not None` (not a truthiness check), so an empty memory object is preserved correctly.

### ReActExecutor

```python
from quaerium.agents.executor import ReActExecutor

executor = ReActExecutor()
```

The default `ExecutionStrategy`. Implements the standard Thought/Action/Observation loop.

**Behaviour:**

1. Builds a system prompt from tool descriptions and conversation history.
2. Calls `llm.generate(prompt)` to get the LLM's next step.
3. Parses `Thought:`, `Action:`, and `Action Input:` from the output.
4. Invokes the named tool and appends the observation.
5. Loops until `Final Answer:` is detected or `max_steps` is reached.

**ReAct prompt format:**

```
Thought: <reasoning text>
Action: <tool_name>
Action Input: {"param": "value"}
Observation: <tool result>
...
Final Answer: <final answer text>
```

---

## Built-in Agents

### RAGAgent

```python
from quaerium.agents import RAGAgent

agent = RAGAgent(
    llm: LLMClient,
    rag_pipeline: RagPipeline,
    extra_tools: list[Tool] = [],
    memory: AgentMemory | None = None,
    observers: list[AgentObserver] = [],
    max_steps: int = 8,
)
```

**Built-in tools:**

| Tool name | Description |
|-----------|-------------|
| `rag_search` | Search the RAG pipeline with a query string |

**Returns:** `AgentResponse` with populated `sources` list.

### IngestionAgent

```python
from quaerium.agents import IngestionAgent

agent = IngestionAgent(
    llm: LLMClient,
    ingestion_service: IngestionService,
    memory: AgentMemory | None = None,
    observers: list[AgentObserver] = [],
    max_steps: int = 6,
)
```

**Built-in tools:**

| Tool name | Description |
|-----------|-------------|
| `parse_document` | Parse and extract text from a file path |
| `list_formats` | List all supported ingestion file formats |
| `check_status` | Check the ingestion status of a document |

### EvaluationAgent

```python
from quaerium.agents import EvaluationAgent

agent = EvaluationAgent(
    llm: LLMClient,
    retrieval_evaluator: RetrievalEvaluator | None = None,
    generation_evaluator: GenerationEvaluator | None = None,
    memory: AgentMemory | None = None,
    observers: list[AgentObserver] = [],
    max_steps: int = 10,
)
```

**Built-in tools** (each is only registered if the corresponding evaluator is provided):

| Tool name | Requires | Description |
|-----------|----------|-------------|
| `evaluate_retrieval` | `retrieval_evaluator` | Run retrieval metrics on a query |
| `evaluate_generation` | `generation_evaluator` | Run generation metrics on an answer |

### ResearchAgent

```python
from quaerium.agents import ResearchAgent

agent = ResearchAgent(
    llm: LLMClient,
    sub_agents: dict[str, Any],
    memory: AgentMemory | None = None,
    observers: list[AgentObserver] = [],
    max_steps: int = 12,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sub_agents` | `dict[str, Any]` | Mapping from tool name to sub-agent. Each value must have a `run(task: str) -> AgentResponse` method. |

Each sub-agent is wrapped as a `FunctionTool` using its dict key as the tool name.

---

## Prompts

### parse_react_output

```python
from quaerium.agents.prompts import parse_react_output

action, action_input, final_answer = parse_react_output(llm_output: str)
```

Parses LLM output using two `re.DOTALL` regexes:

- Extracts `Action: <name>` into `action` (`str | None`)
- Extracts `Action Input: <json>` into `action_input` (`dict | None`) with JSON fallback to `{"query": raw}`
- Extracts `Final Answer: <text>` into `final_answer` (`str | None`)

!!! note "Fallback behaviour"
    If JSON parsing of `Action Input` fails, the raw text is wrapped as `{"query": raw_text}`. This prevents crashes on imperfect LLM formatting.

---

## Error Handling

All agent methods raise standard Python exceptions:

| Exception | Cause |
|-----------|-------|
| `ValueError` | Invalid constructor arguments (e.g. empty `sub_agents`) |
| `RuntimeError` | Executor reached `max_steps` without `Final Answer` (recorded in `trace.error`, not raised) |
| `ToolError` | Tool raised an exception and `max_retries` was exhausted (recorded as observation) |

!!! tip "Checking for failures"
    The recommended pattern is to check `response.trace.success` rather than catching exceptions:
    ```python
    response = runtime.run(task)
    if not response.trace.success:
        logger.error(f"Agent failed: {response.trace.error}")
    ```

---

## See Also

- [Agents Guide](../../guides/agents.md) — Conceptual documentation and worked examples
- [Agents Examples](../../examples/agents.md) — Runnable code patterns
- [Evaluation API](../evaluation/index.md) — Evaluators used by `EvaluationAgent`
- [RAG Pipeline API](../rag/pipeline.md) — Pipelines used by `RAGAgent`
