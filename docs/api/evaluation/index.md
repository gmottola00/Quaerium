# Evaluation API Reference

## Core Protocols

### MetricCalculator

Protocol for individual metric calculation.

```python
from quaerium.core.evaluation.protocols import MetricCalculator

@runtime_checkable
class MetricCalculator(Protocol):
    @property
    def name(self) -> str:
        """Metric name (e.g., "Precision@5")"""
        ...

    def calculate(
        self,
        predictions: Any,
        ground_truth: Any,
        **kwargs: Any,
    ) -> MetricScore:
        """Calculate metric from predictions and ground truth."""
        ...
```

### RetrievalEvaluator

Protocol for evaluating retrieval quality.

```python
from quaerium.core.evaluation.protocols import RetrievalEvaluator

@runtime_checkable
class RetrievalEvaluator(Protocol):
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        relevant_doc_ids: list[str],
        **kwargs: Any,
    ) -> RetrievalMetrics:
        """Evaluate retrieval for a single query."""
        ...
```

**Parameters:**
- `query`: Original search query
- `retrieved_docs`: Retrieved documents with 'id' and 'score' fields
- `relevant_doc_ids`: Ground truth relevant document IDs

**Returns:** `RetrievalMetrics` object

### GenerationEvaluator

Protocol for evaluating generation quality.

```python
from quaerium.core.evaluation.protocols import GenerationEvaluator

@runtime_checkable
class GenerationEvaluator(Protocol):
    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        context: list[str],
        reference_answer: str | None = None,
        **kwargs: Any,
    ) -> GenerationMetrics:
        """Evaluate generated answer quality."""
        ...
```

**Parameters:**
- `question`: Original question
- `generated_answer`: Answer from RAG system
- `context`: Context chunks used for generation
- `reference_answer`: Optional ground truth answer

**Returns:** `GenerationMetrics` object

### LLMJudge

Protocol for LLM-based evaluation.

```python
from quaerium.core.evaluation.protocols import LLMJudge

@runtime_checkable
class LLMJudge(Protocol):
    def judge(self, prompt: str, **kwargs: Any) -> float:
        """Get numerical judgment (0-1)."""
        ...

    def judge_with_reasoning(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[float, str]:
        """Get judgment with explanation."""
        ...
```

### PipelineObserver

Protocol for observing pipeline execution.

```python
from quaerium.core.evaluation.protocols import PipelineObserver

@runtime_checkable
class PipelineObserver(Protocol):
    def on_query_rewrite(
        self, original: str, rewritten: str, metadata: dict
    ) -> None: ...

    def on_retrieval(
        self, query: str, results: list[dict], metadata: dict
    ) -> None: ...

    def on_reranking(
        self, query: str, before: list, after: list, metadata: dict
    ) -> None: ...

    def on_context_assembly(
        self, chunks: list, token_count: int, metadata: dict
    ) -> None: ...

    def on_generation(
        self, question: str, answer: str, context: list, metadata: dict
    ) -> None: ...

    def get_results(self) -> EvaluationResult: ...
```

## Data Models

### MetricScore

Individual metric result.

```python
from quaerium.core.evaluation.metrics import MetricScore

@dataclass(frozen=True)
class MetricScore:
    name: str              # Metric name
    value: float           # Metric value
    metadata: dict = {}    # Additional info
```

### RetrievalMetrics

Retrieval evaluation results.

```python
from quaerium.core.evaluation.metrics import RetrievalMetrics

@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: dict[int, float]  # {5: 0.8, 10: 0.7}
    recall_at_k: dict[int, float]     # {5: 0.4, 10: 0.6}
    mrr: float                         # 0-1
    ndcg: float                        # 0-1
    hit_rate: float                    # 0 or 1
    metadata: dict = {}
```

### GenerationMetrics

Generation evaluation results.

```python
from quaerium.core.evaluation.metrics import GenerationMetrics

@dataclass(frozen=True)
class GenerationMetrics:
    relevance_score: float           # 0-1
    faithfulness_score: float        # 0-1
    hallucination_score: float       # 0-1 (lower is better)
    answer_similarity: float | None  # Optional
    metadata: dict = {}
```

### EndToEndMetrics

Pipeline performance metrics.

```python
from quaerium.core.evaluation.metrics import EndToEndMetrics

@dataclass
class EndToEndMetrics:
    total_latency_ms: float          # Total time (ms)
    retrieval_latency_ms: float      # Retrieval time (ms)
    generation_latency_ms: float     # Generation time (ms)
    total_tokens: int                # Total tokens used
    cost_usd: float                  # Estimated cost
    context_utilization: float       # 0-1
    metadata: dict = {}
```

### EvaluationResult

Complete evaluation result.

```python
from quaerium.core.evaluation.metrics import EvaluationResult

@dataclass
class EvaluationResult:
    query: str
    retrieval_metrics: RetrievalMetrics | None = None
    generation_metrics: GenerationMetrics | None = None
    end_to_end_metrics: EndToEndMetrics | None = None
    success: bool = True
    error: str | None = None
```

### EvaluationExample

Single evaluation example.

```python
from quaerium.core.evaluation.dataset import EvaluationExample

@dataclass(frozen=True)
class EvaluationExample:
    query: str                    # Question/query
    relevant_doc_ids: list[str]  # Relevant document IDs
    reference_answer: str | None = None
    metadata: dict = {}
```

## Retrieval Metrics

### PrecisionAtK

```python
from quaerium.infra.evaluation.retrieval import PrecisionAtK

calculator = PrecisionAtK(k=5)
score = calculator.calculate(
    predictions=["doc1", "doc2", "doc3"],
    ground_truth=["doc1", "doc5"]
)
# score.value = 0.33 (1 out of 3)
```

**Parameters:**
- `k`: Number of top results to consider

**Methods:**
- `calculate(predictions, ground_truth)`: Returns `MetricScore`

### RecallAtK

```python
from quaerium.infra.evaluation.retrieval import RecallAtK

calculator = RecallAtK(k=10)
score = calculator.calculate(
    predictions=["doc1", "doc2"],
    ground_truth=["doc1", "doc3", "doc5"]
)
# score.value = 0.33 (1 out of 3 found)
```

### MRRCalculator

```python
from quaerium.infra.evaluation.retrieval import MRRCalculator

calculator = MRRCalculator()
score = calculator.calculate(
    predictions=["doc1", "doc2", "doc3"],
    ground_truth=["doc2"]
)
# score.value = 0.5 (found at rank 2, so 1/2)
```

### NDCGCalculator

```python
from quaerium.infra.evaluation.retrieval import NDCGCalculator

calculator = NDCGCalculator(k=10)
score = calculator.calculate(
    predictions=["doc1", "doc2", "doc3"],
    ground_truth=["doc1", "doc3"]
)
# score.value = NDCG score (0-1)
```

### HitRateCalculator

```python
from quaerium.infra.evaluation.retrieval import HitRateCalculator

calculator = HitRateCalculator(k=5)
score = calculator.calculate(
    predictions=["doc1", "doc2"],
    ground_truth=["doc3"]
)
# score.value = 0.0 (no hits)
```

### StandardRetrievalEvaluator

```python
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator

evaluator = StandardRetrievalEvaluator(k_values=[5, 10])
metrics = evaluator.evaluate_retrieval(
    query="test query",
    retrieved_docs=[
        {"id": "doc1", "score": 0.9},
        {"id": "doc2", "score": 0.8}
    ],
    relevant_doc_ids=["doc1", "doc3"]
)
```

**Parameters:**
- `k_values`: List of K values to evaluate (default: [5, 10])

**Returns:** `RetrievalMetrics` with all metrics computed

## Generation Metrics

### OpenAIJudge

```python
from quaerium.infra.evaluation.generation import OpenAIJudge
from quaerium.infra.llm import OpenAIClient

llm = OpenAIClient(model="gpt-4")
judge = OpenAIJudge(llm=llm)

# Simple judgment
score = judge.judge("Rate this answer: ...")  # Returns 0-1

# With reasoning
score, reasoning = judge.judge_with_reasoning("Evaluate: ...")
```

**Supported score formats:**
- "8/10" → 0.8
- "Score: 7" → 0.7
- "0.75" → 0.75
- Standalone numbers

### AnswerRelevanceEvaluator

```python
from quaerium.infra.evaluation.generation import AnswerRelevanceEvaluator

evaluator = AnswerRelevanceEvaluator(judge)
score = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a programming language."
)
# score = 0-1 (relevance)
```

### FaithfulnessEvaluator

```python
from quaerium.infra.evaluation.generation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(judge)
score = evaluator.evaluate(
    answer="Python was created in 1991.",
    context=["Python was created by Guido in 1991."]
)
# score = 0-1 (faithfulness)
```

### HallucinationDetector

```python
from quaerium.infra.evaluation.generation import HallucinationDetector

detector = HallucinationDetector(judge)
score = detector.detect(
    answer="Python supports quantum computing.",
    context=["Python is a general-purpose language."]
)
# score = 0-1 (higher = more hallucination)
```

### StandardGenerationEvaluator

```python
from quaerium.infra.evaluation.generation import StandardGenerationEvaluator

evaluator = StandardGenerationEvaluator(judge)
metrics = evaluator.evaluate_answer(
    question="What is Python?",
    generated_answer="Python is a programming language.",
    context=["Python is a high-level language."],
    reference_answer="Python is a language."  # Optional
)
```

**Returns:** `GenerationMetrics` with all metrics

## Pipeline Integration

### MetricsObserver

```python
from quaerium.infra.evaluation.pipeline import MetricsObserver

observer = MetricsObserver()

# Use with pipeline
pipeline = RagPipeline(..., observers=[observer])
response = pipeline.run(query)

# Get results
result = observer.get_results()
```

**Methods:**
- `on_query_rewrite(original, rewritten, metadata)`: Called after query rewriting
- `on_retrieval(query, results, metadata)`: Called after retrieval
- `on_reranking(query, before, after, metadata)`: Called after reranking
- `on_context_assembly(chunks, token_count, metadata)`: Called after assembly
- `on_generation(question, answer, context, metadata)`: Called after generation
- `get_results()`: Returns `EvaluationResult`

### LatencyTracker

```python
from quaerium.infra.evaluation.pipeline.trackers import LatencyTracker

tracker = LatencyTracker()
tracker.start("stage_name")
# ... do work ...
tracker.end("stage_name")
latency = tracker.get_latency("stage_name")  # ms
```

### TokenTracker

```python
from quaerium.infra.evaluation.pipeline.trackers import TokenTracker

tracker = TokenTracker()
tracker.add_usage(
    prompt_tokens=100,
    completion_tokens=50,
    model="gpt-4"
)
print(tracker.total_tokens)       # 150
print(tracker.estimated_cost_usd) # $0.0045
```

**Supported models:** GPT-4, GPT-4 Turbo, GPT-3.5, GPT-4o, GPT-4o Mini

## Dataset Management

### JSONLDataset

```python
from quaerium.infra.evaluation.datasets import JSONLDataset

dataset = JSONLDataset("path/to/dataset.jsonl")

# Access
print(len(dataset))           # Number of examples
example = dataset[0]          # Get by index
print(dataset.metadata)       # Dataset metadata

# Iterate
for example in dataset:
    print(example.query)
    print(example.relevant_doc_ids)
    print(example.reference_answer)
```

**JSONL format:**
```json
{"query": "...", "relevant_doc_ids": [...], "reference_answer": "..."}
```

**Properties:**
- `metadata`: `DatasetMetadata` object
- Supports indexing: `dataset[0]`
- Supports iteration: `for ex in dataset`
- Supports `len()`: `len(dataset)`

## Usage Examples

### Basic Retrieval Evaluation

```python
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator

evaluator = StandardRetrievalEvaluator(k_values=[5, 10])
metrics = evaluator.evaluate_retrieval(
    query="test",
    retrieved_docs=[{"id": "d1", "score": 0.9}],
    relevant_doc_ids=["d1", "d2"]
)

print(f"P@5: {metrics.precision_at_k[5]:.2f}")
print(f"R@5: {metrics.recall_at_k[5]:.2f}")
```

### Basic Generation Evaluation

```python
from quaerium.infra.evaluation.generation import (
    StandardGenerationEvaluator, OpenAIJudge
)
from quaerium.infra.llm import OpenAIClient

judge = OpenAIJudge(llm=OpenAIClient(model="gpt-4"))
evaluator = StandardGenerationEvaluator(judge)

metrics = evaluator.evaluate_answer(
    question="What is RAG?",
    generated_answer="RAG combines retrieval with generation.",
    context=["RAG is retrieval-augmented generation."]
)

print(f"Relevance: {metrics.relevance_score:.2f}")
```

### Pipeline Monitoring

```python
from quaerium.infra.evaluation.pipeline import MetricsObserver

observer = MetricsObserver()
pipeline = RagPipeline(..., observers=[observer])

response = pipeline.run("query")
result = observer.get_results()

print(f"Latency: {result.end_to_end_metrics.total_latency_ms}ms")
print(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")
```

### Dataset Evaluation

```python
from quaerium.infra.evaluation.datasets import JSONLDataset

dataset = JSONLDataset("eval.jsonl")
results = []

for example in dataset:
    # Run evaluation
    metrics = evaluate(example)
    results.append(metrics)

# Aggregate
avg_score = sum(m.score for m in results) / len(results)
```

## Type Annotations

All evaluation code includes complete type annotations for IDE support:

```python
from quaerium.core.evaluation.protocols import RetrievalEvaluator
from quaerium.core.evaluation.metrics import RetrievalMetrics

def evaluate(evaluator: RetrievalEvaluator) -> RetrievalMetrics:
    return evaluator.evaluate_retrieval(...)
```

## Error Handling

All evaluation functions raise appropriate exceptions:

```python
try:
    metrics = evaluator.evaluate_retrieval(...)
except ValueError as e:
    # Missing required fields
    print(f"Validation error: {e}")
except Exception as e:
    # Other errors
    print(f"Evaluation failed: {e}")
```

Common exceptions:
- `ValueError`: Invalid parameters or missing required fields
- `FileNotFoundError`: Dataset file not found
- `KeyError`: Missing expected keys in data structures
