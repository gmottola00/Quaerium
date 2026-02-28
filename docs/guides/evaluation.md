# Evaluation & Metrics Guide

## Overview

Quaerium provides a comprehensive evaluation framework for measuring and improving RAG system quality. The framework covers three key areas:

1. **Retrieval Metrics** - Measure how well documents are retrieved (Precision@K, Recall@K, MRR, NDCG)
2. **Generation Metrics** - Evaluate answer quality using LLM-as-judge (relevance, faithfulness, hallucination detection)
3. **End-to-End Metrics** - Track performance (latency, cost, token usage)

## Quick Start

```python
from quaerium import RagPipeline
from quaerium.infra.evaluation.pipeline import MetricsObserver

# Create observer
observer = MetricsObserver()

# Create pipeline with observer
pipeline = RagPipeline(
    vector_searcher=searcher,
    rewriter=rewriter,
    reranker=reranker,
    assembler=assembler,
    generator_llm=llm,
    observers=[observer]  # Add observer here
)

# Run pipeline
response = pipeline.run("What is RAG?")

# Get metrics
result = observer.get_results()
print(f"Latency: {result.end_to_end_metrics.total_latency_ms:.0f}ms")
print(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")
print(f"Tokens: {result.end_to_end_metrics.total_tokens}")
```

## Retrieval Metrics

### Overview

Retrieval metrics measure how well your RAG system finds relevant documents. These metrics compare retrieved documents against ground truth relevance labels.

### Available Metrics

#### Precision@K

Measures what fraction of the top-K retrieved documents are relevant.

**Formula:** `Precision@K = (# relevant docs in top-K) / K`

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.retrieval import PrecisionAtK

calc = PrecisionAtK(k=5)
score = calc.calculate(
    predictions=["doc1", "doc2", "doc3", "doc4", "doc5"],
    ground_truth=["doc1", "doc3", "doc7"]
)

print(f"{score.name}: {score.value:.2f}")  # Precision@5: 0.40
```

**Use case:** Useful when you care about precision in the top results (e.g., only showing top 5 results to users).

#### Recall@K

Measures what fraction of all relevant documents are found in the top-K results.

**Formula:** `Recall@K = (# relevant docs in top-K) / (total # relevant docs)`

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.retrieval import RecallAtK

calc = RecallAtK(k=10)
score = calc.calculate(
    predictions=["doc1", "doc2", "doc3"],
    ground_truth=["doc1", "doc3", "doc7", "doc9"]
)

print(f"{score.name}: {score.value:.2f}")  # Recall@10: 0.50 (found 2 out of 4)
```

**Use case:** Useful when you need to find as many relevant documents as possible.

#### MRR (Mean Reciprocal Rank)

Measures the rank of the first relevant document. Higher rank (earlier position) gives higher score.

**Formula:** `MRR = 1 / (rank of first relevant doc)`

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.retrieval import MRRCalculator

calc = MRRCalculator()
score = calc.calculate(
    predictions=["doc1", "doc2", "doc3", "doc4"],
    ground_truth=["doc3", "doc7"]
)

print(f"{score.name}: {score.value:.3f}")  # MRR: 0.333 (found at rank 3)
```

**Use case:** Useful when users typically only look at the first relevant result.

#### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality with position-based discounting. Relevant documents at higher ranks contribute more to the score.

**Formula:** `NDCG@K = DCG@K / IDCG@K`

where DCG = Σ(relevance / log2(rank + 1))

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.retrieval import NDCGCalculator

calc = NDCGCalculator(k=10)
score = calc.calculate(
    predictions=["doc1", "doc2", "doc3", "doc4"],
    ground_truth=["doc1", "doc3", "doc7"]
)

print(f"{score.name}: {score.value:.3f}")  # NDCG@10: 0.xxx
```

**Use case:** Gold standard metric for ranking quality, especially when you care about the order of results.

#### Hit Rate

Binary metric: 1 if at least one relevant document was retrieved, 0 otherwise.

**Range:** 0 or 1

**Example:**
```python
from quaerium.infra.evaluation.retrieval import HitRateCalculator

calc = HitRateCalculator(k=10)
score = calc.calculate(
    predictions=["doc1", "doc2"],
    ground_truth=["doc1", "doc5"]
)

print(f"{score.name}: {score.value}")  # Hit@10: 1.0
```

**Use case:** Useful for ensuring your system can find at least one relevant document.

### Standard Retrieval Evaluator

Compute all metrics at once:

```python
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator

evaluator = StandardRetrievalEvaluator(k_values=[5, 10])

metrics = evaluator.evaluate_retrieval(
    query="What is RAG?",
    retrieved_docs=[
        {"id": "doc1", "score": 0.95},
        {"id": "doc2", "score": 0.87},
        {"id": "doc3", "score": 0.75}
    ],
    relevant_doc_ids=["doc1", "doc3", "doc7"]
)

print(f"Precision@5: {metrics.precision_at_k[5]:.2f}")
print(f"Recall@5: {metrics.recall_at_k[5]:.2f}")
print(f"MRR: {metrics.mrr:.2f}")
print(f"NDCG: {metrics.ndcg:.2f}")
print(f"Hit Rate: {metrics.hit_rate:.0f}")
```

## Generation Metrics

### Overview

Generation metrics evaluate the quality of generated answers using LLM-as-judge. These metrics measure subjective qualities that are difficult to capture with traditional metrics.

### Available Metrics

#### Answer Relevance

Measures how well the generated answer addresses the question.

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.generation import OpenAIJudge, AnswerRelevanceEvaluator
from quaerium.infra.llm import OpenAIClient

llm = OpenAIClient(model="gpt-4")
judge = OpenAIJudge(llm=llm)
evaluator = AnswerRelevanceEvaluator(judge)

score = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a high-level programming language."
)

print(f"Relevance: {score:.2f}")  # e.g., 0.90
```

**Evaluation prompt:** Asks the LLM to rate how well the answer addresses the question on a 0-10 scale.

#### Faithfulness

Measures whether the generated answer is supported by the provided context.

**Range:** 0-1 (higher is better)

**Example:**
```python
from quaerium.infra.evaluation.generation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(judge)

score = evaluator.evaluate(
    answer="Python was created in 1991.",
    context=["Python was created by Guido van Rossum in 1991."]
)

print(f"Faithfulness: {score:.2f}")  # e.g., 1.0 (fully supported)
```

**Evaluation prompt:** Asks the LLM to rate whether all claims in the answer are supported by the context.

#### Hallucination Detection

Detects claims in the answer that are NOT supported by the context.

**Range:** 0-1 (lower is better, 0 = no hallucination)

**Example:**
```python
from quaerium.infra.evaluation.generation import HallucinationDetector

detector = HallucinationDetector(judge)

score = detector.detect(
    answer="Python was created in 2020.",
    context=["Python was created in 1991."]
)

print(f"Hallucination: {score:.2f}")  # e.g., 0.90 (high hallucination)
```

**Note:** Hallucination score is inverse of faithfulness.

### Standard Generation Evaluator

Compute all generation metrics at once:

```python
from quaerium.infra.evaluation.generation import StandardGenerationEvaluator

evaluator = StandardGenerationEvaluator(judge)

metrics = evaluator.evaluate_answer(
    question="What is Python?",
    generated_answer="Python is a programming language created in 1991.",
    context=["Python is a high-level programming language.", "It was created in 1991."],
    reference_answer="Python is a programming language."  # Optional
)

print(f"Relevance: {metrics.relevance_score:.2f}")
print(f"Faithfulness: {metrics.faithfulness_score:.2f}")
print(f"Hallucination: {metrics.hallucination_score:.2f}")
```

### LLM-as-Judge

The `OpenAIJudge` class handles score parsing from LLM responses. It supports multiple formats:

```python
from quaerium.infra.evaluation.generation import OpenAIJudge

judge = OpenAIJudge(llm=my_llm)

# Supported formats:
# - "8/10" or "8 out of 10" → 0.8
# - "Score: 7" → 0.7
# - "0.75" → 0.75
# - Standalone number in text

score = judge.judge("Rate this answer: ...")
score, reasoning = judge.judge_with_reasoning("Evaluate and explain: ...")
```

## End-to-End Metrics

### Overview

End-to-end metrics track the performance of your entire RAG pipeline using the observer pattern.

### MetricsObserver

```python
from quaerium.infra.evaluation.pipeline import MetricsObserver

observer = MetricsObserver()
pipeline = RagPipeline(..., observers=[observer])

response = pipeline.run("What is RAG?")
result = observer.get_results()

# Access metrics
metrics = result.end_to_end_metrics
print(f"Total latency: {metrics.total_latency_ms:.0f}ms")
print(f"Retrieval latency: {metrics.retrieval_latency_ms:.0f}ms")
print(f"Generation latency: {metrics.generation_latency_ms:.0f}ms")
print(f"Total tokens: {metrics.total_tokens}")
print(f"Cost: ${metrics.cost_usd:.4f}")
print(f"Context utilization: {metrics.context_utilization:.2%}")
```

### Tracked Metrics

| Metric | Description |
|--------|-------------|
| `total_latency_ms` | Total pipeline execution time (ms) |
| `retrieval_latency_ms` | Time spent in retrieval stage (ms) |
| `generation_latency_ms` | Time spent in generation stage (ms) |
| `total_tokens` | Total tokens used (prompt + completion) |
| `cost_usd` | Estimated cost in USD |
| `context_utilization` | Fraction of context window used (0-1) |

### Cost Estimation

The observer automatically estimates costs based on token usage and model:

**Supported models:**
- GPT-4: $30/$60 per 1M tokens (prompt/completion)
- GPT-4 Turbo: $10/$30 per 1M tokens
- GPT-3.5 Turbo: $0.5/$1.5 per 1M tokens
- GPT-4o: $5/$15 per 1M tokens
- GPT-4o Mini: $0.15/$0.6 per 1M tokens

**Example:**
```python
# After pipeline run
metrics = observer.get_results().end_to_end_metrics

print(f"Prompt tokens: {metrics.metadata['prompt_tokens']}")
print(f"Completion tokens: {metrics.metadata['completion_tokens']}")
print(f"Estimated cost: ${metrics.cost_usd:.4f}")
```

## Dataset Management

### Overview

Evaluation datasets contain queries with ground truth labels (relevant documents, reference answers) for systematic testing.

### JSONL Format

Each line is a JSON object:

```json
{"query": "What is RAG?", "relevant_doc_ids": ["doc1", "doc3"], "reference_answer": "RAG combines retrieval with generation."}
{"query": "How to install?", "relevant_doc_ids": ["doc5"], "metadata": {"category": "setup"}}
```

**Required fields:**
- `query`: The question/query
- `relevant_doc_ids`: List of relevant document IDs

**Optional fields:**
- `reference_answer`: Ground truth answer
- `metadata`: Additional metadata (category, difficulty, etc.)

### Loading Datasets

```python
from quaerium.infra.evaluation.datasets import JSONLDataset

dataset = JSONLDataset("data/eval_dataset.jsonl")

print(f"Dataset: {dataset.metadata.name}")
print(f"Examples: {len(dataset)}")

# Iterate
for example in dataset:
    print(f"Query: {example.query}")
    print(f"Relevant docs: {example.relevant_doc_ids}")
    print(f"Reference: {example.reference_answer}")
```

### Sample Dataset

Quaerium includes a sample dataset:

```python
dataset = JSONLDataset("examples/data/sample_eval_dataset.jsonl")
# 10 RAG evaluation examples included
```

## Complete Evaluation Workflow

### Step 1: Load Dataset

```python
from quaerium.infra.evaluation.datasets import JSONLDataset

dataset = JSONLDataset("data/my_eval_dataset.jsonl")
```

### Step 2: Create Evaluators

```python
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator
from quaerium.infra.evaluation.generation import StandardGenerationEvaluator, OpenAIJudge
from quaerium.infra.evaluation.pipeline import MetricsObserver
from quaerium.infra.llm import OpenAIClient

# Retrieval evaluator
retrieval_eval = StandardRetrievalEvaluator(k_values=[5, 10])

# Generation evaluator
llm = OpenAIClient(model="gpt-4")
judge = OpenAIJudge(llm=llm)
generation_eval = StandardGenerationEvaluator(judge)

# Pipeline observer
observer = MetricsObserver()
```

### Step 3: Create Pipeline with Observer

```python
from quaerium import RagPipeline

pipeline = RagPipeline(
    vector_searcher=searcher,
    rewriter=rewriter,
    reranker=reranker,
    assembler=assembler,
    generator_llm=llm,
    observers=[observer]
)
```

### Step 4: Run Evaluation

```python
results = []

for example in dataset:
    # Run pipeline
    response = pipeline.run(example.query)

    # Get end-to-end metrics
    e2e_result = observer.get_results()

    # Evaluate retrieval (you need to capture retrieved docs)
    # This requires modifying the pipeline or observer to track retrieved docs
    retrieval_metrics = retrieval_eval.evaluate_retrieval(
        query=example.query,
        retrieved_docs=...,  # Get from pipeline/observer
        relevant_doc_ids=example.relevant_doc_ids
    )

    # Evaluate generation
    generation_metrics = generation_eval.evaluate_answer(
        question=example.query,
        generated_answer=response.answer,
        context=[c.text for c in response.citations],
        reference_answer=example.reference_answer
    )

    results.append({
        "query": example.query,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "e2e": e2e_result.end_to_end_metrics
    })

# Aggregate results
avg_precision = sum(r["retrieval"].precision_at_k[5] for r in results) / len(results)
avg_relevance = sum(r["generation"].relevance_score for r in results) / len(results)
avg_latency = sum(r["e2e"].total_latency_ms for r in results) / len(results)

print(f"Average Precision@5: {avg_precision:.2%}")
print(f"Average Relevance: {avg_relevance:.2%}")
print(f"Average Latency: {avg_latency:.0f}ms")
```

## Best Practices

### 1. Choose Appropriate Metrics

- **Precision@K**: When you care about top-K accuracy
- **Recall@K**: When you need comprehensive coverage
- **NDCG**: When ranking order matters
- **MRR**: When users look at first relevant result

### 2. Use Multiple K Values

```python
evaluator = StandardRetrievalEvaluator(k_values=[3, 5, 10, 20])
```

This helps understand performance at different cutoffs.

### 3. Balance Cost and Quality

LLM-as-judge evaluation can be expensive. Consider:
- Using cheaper models (GPT-3.5) for initial evaluation
- Sampling a subset of examples
- Caching judge responses

### 4. Validate Ground Truth

Ensure your evaluation dataset has accurate labels:
- Multiple annotators for reliability
- Clear annotation guidelines
- Regular review and updates

### 5. Monitor in Production

Use observers to track production metrics:
```python
# Log metrics for monitoring
result = observer.get_results()
logger.info(f"Latency: {result.end_to_end_metrics.total_latency_ms}ms")
logger.info(f"Cost: ${result.end_to_end_metrics.cost_usd:.4f}")

# Alert on degradation
if result.end_to_end_metrics.total_latency_ms > 2000:
    alert("High latency detected")
```

## Troubleshooting

### Observer Not Collecting Metrics

**Problem:** `observer.get_results()` returns error or empty metrics

**Solutions:**
1. Ensure observer is passed to pipeline: `RagPipeline(..., observers=[observer])`
2. Check that pipeline actually ran: `response = pipeline.run(query)`
3. Verify observer callbacks are being called

### LLM Judge Returns Errors

**Problem:** Score parsing fails or LLM returns unexpected format

**Solutions:**
1. Check LLM response format in logs
2. Adjust prompts to be more explicit
3. Use `judge_with_reasoning()` to see full LLM response
4. Try different LLM models

### High Evaluation Costs

**Problem:** LLM-as-judge evaluation is expensive

**Solutions:**
1. Use cheaper models (GPT-3.5 instead of GPT-4)
2. Sample evaluation dataset
3. Cache judge responses
4. Use traditional metrics where possible

### Retrieval Metrics Don't Match Expectations

**Problem:** Metrics seem too low/high

**Solutions:**
1. Verify ground truth labels are correct
2. Check that `relevant_doc_ids` match actual document IDs
3. Ensure retrieved docs have correct ID format
4. Review sample results manually

## API Reference

See [API Documentation](../api/evaluation/index.md) for complete reference.

## Examples

See [evaluation_example.py](../../examples/evaluation_example.py) for a complete working example.
