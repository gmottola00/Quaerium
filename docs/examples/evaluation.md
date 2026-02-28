# Evaluation & Metrics Example

This example demonstrates how to evaluate RAG pipeline quality using Quaerium's comprehensive evaluation framework.

## Overview

Quaerium provides three types of evaluation:

1. **Retrieval Metrics** - Measure how well the system retrieves relevant documents
2. **Generation Metrics** - Measure answer quality using LLM-as-judge
3. **End-to-End Metrics** - Track latency, cost, and token usage

## Prerequisites

```bash
# Install Quaerium with evaluation support
pip install quaerium[ollama]

# Or with OpenAI for LLM-as-judge
pip install quaerium[openai]
```

## Quick Start

### 1. Create Evaluation Dataset

First, create a JSONL dataset with test queries and ground truth:

```python title="create_dataset.py"
import json
from pathlib import Path

# Define evaluation examples
examples = [
    {
        "query": "What is RAG?",
        "relevant_doc_ids": ["doc_rag_intro", "doc_rag_architecture"],
        "reference_answer": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation."
    },
    {
        "query": "How do vector stores work?",
        "relevant_doc_ids": ["doc_vectorstore_intro", "doc_embeddings"],
        "reference_answer": "Vector stores enable semantic search by storing embeddings of text."
    },
    {
        "query": "What are embeddings?",
        "relevant_doc_ids": ["doc_embeddings", "doc_similarity"],
        "reference_answer": "Embeddings are numerical vector representations of text that capture semantic meaning."
    }
]

# Save as JSONL
output_path = Path("data/eval_dataset.jsonl")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"✓ Created evaluation dataset: {output_path}")
```

### 2. Basic Evaluation

```python title="basic_evaluation.py"
from quaerium.infra.evaluation.datasets import JSONLDataset
from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator
from quaerium.infra.evaluation.generation import (
    StandardGenerationEvaluator,
    OpenAIJudge
)
from quaerium.infra.llm.openai import OpenAILLMClient

# Load dataset
dataset = JSONLDataset("data/eval_dataset.jsonl")
print(f"Loaded {len(dataset)} evaluation examples")

# Create retrieval evaluator
retrieval_eval = StandardRetrievalEvaluator(k_values=[5, 10])

# Create generation evaluator with LLM judge
llm = OpenAILLMClient(model="gpt-4", api_key="your-api-key")
judge = OpenAIJudge(llm=llm)
generation_eval = StandardGenerationEvaluator(judge)

# Evaluate single example
example = dataset[0]

# Simulated retrieval results
retrieved_docs = [
    {"id": "doc_rag_intro", "score": 0.95},
    {"id": "doc_rag_architecture", "score": 0.87},
    {"id": "doc_other", "score": 0.45}
]

# Evaluate retrieval
retrieval_metrics = retrieval_eval.evaluate_retrieval(
    query=example.query,
    retrieved_docs=retrieved_docs,
    relevant_doc_ids=example.relevant_doc_ids
)

print(f"\nRetrieval Metrics:")
print(f"  Precision@5: {retrieval_metrics.precision_at_k[5]:.2%}")
print(f"  Recall@5: {retrieval_metrics.recall_at_k[5]:.2%}")
print(f"  MRR: {retrieval_metrics.mrr:.3f}")
print(f"  NDCG: {retrieval_metrics.ndcg:.3f}")

# Evaluate generation
generated_answer = "RAG combines retrieval with generation for better answers."
context = ["RAG is a technique...", "Vector stores enable search..."]

generation_metrics = generation_eval.evaluate_answer(
    question=example.query,
    generated_answer=generated_answer,
    context=context,
    reference_answer=example.reference_answer
)

print(f"\nGeneration Metrics:")
print(f"  Relevance: {generation_metrics.relevance_score:.2%}")
print(f"  Faithfulness: {generation_metrics.faithfulness_score:.2%}")
print(f"  Hallucination: {generation_metrics.hallucination_score:.2%}")
```

### 3. Pipeline Evaluation with Observer

```python title="pipeline_evaluation.py"
from quaerium import RagPipeline
from quaerium.infra.evaluation.pipeline import MetricsObserver
from quaerium.infra.llm.ollama import OllamaLLMClient
from quaerium.infra.embedding.ollama import OllamaEmbeddingClient
from quaerium.infra.vectorstore.milvus import MilvusVectorStore

# Create observer
observer = MetricsObserver()

# Create pipeline with observer
embedding = OllamaEmbeddingClient(model="nomic-embed-text")
llm = OllamaLLMClient(model="llama3.2")
vectorstore = MilvusVectorStore(host="localhost", collection_name="docs")

pipeline = RagPipeline(
    vectorstore=vectorstore,
    embedding=embedding,
    llm=llm,
    observers=[observer]  # Add observer
)

# Run pipeline
response = pipeline.run("What is RAG?")

# Get end-to-end metrics
result = observer.get_results()

if result.success:
    metrics = result.end_to_end_metrics
    print(f"\nEnd-to-End Metrics:")
    print(f"  Total Latency: {metrics.total_latency_ms:.0f}ms")
    print(f"  Retrieval Latency: {metrics.retrieval_latency_ms:.0f}ms")
    print(f"  Generation Latency: {metrics.generation_latency_ms:.0f}ms")
    print(f"  Total Tokens: {metrics.total_tokens:,}")
    print(f"  Estimated Cost: ${metrics.cost_usd:.4f}")
    print(f"  Context Utilization: {metrics.context_utilization:.2%}")
else:
    print(f"Pipeline failed: {result.error}")
```

## Complete Example

See the [full evaluation example](../../examples/evaluation_example.py) for a comprehensive demonstration including:

- Loading evaluation datasets
- Creating all types of evaluators
- Running RAG pipeline with metrics observer
- Evaluating retrieval and generation quality
- Aggregating results across multiple queries
- Quality assessment and reporting

```bash
# Run the complete example
python examples/evaluation_example.py
```

## Dataset Evaluation Loop

```python title="batch_evaluation.py"
from quaerium.infra.evaluation.datasets import JSONLDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
dataset = JSONLDataset("data/eval_dataset.jsonl")

# Initialize evaluators (same as above)
retrieval_eval = StandardRetrievalEvaluator(k_values=[5, 10])
generation_eval = StandardGenerationEvaluator(judge)
observer = MetricsObserver()

# Create pipeline with observer
pipeline = RagPipeline(..., observers=[observer])

# Evaluate all examples
results = []

for idx, example in enumerate(dataset):
    logger.info(f"[{idx+1}/{len(dataset)}] Evaluating: {example.query}")

    # Run pipeline
    response = pipeline.run(example.query)

    # Get metrics from observer
    e2e_result = observer.get_results()

    if not e2e_result.success:
        logger.warning(f"Pipeline failed: {e2e_result.error}")
        continue

    # Extract retrieved docs from observer (implementation-specific)
    retrieved_docs = [...]  # From pipeline/observer

    # Evaluate retrieval
    retrieval_metrics = retrieval_eval.evaluate_retrieval(
        query=example.query,
        retrieved_docs=retrieved_docs,
        relevant_doc_ids=example.relevant_doc_ids
    )

    # Evaluate generation
    generation_metrics = generation_eval.evaluate_answer(
        question=example.query,
        generated_answer=response.answer,
        context=[c.text for c in response.citations],
        reference_answer=example.reference_answer
    )

    # Store results
    results.append({
        "query": example.query,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "e2e": e2e_result.end_to_end_metrics
    })

# Aggregate results
avg_precision = sum(r["retrieval"].precision_at_k[5] for r in results) / len(results)
avg_recall = sum(r["retrieval"].recall_at_k[5] for r in results) / len(results)
avg_relevance = sum(r["generation"].relevance_score for r in results) / len(results)
avg_latency = sum(r["e2e"].total_latency_ms for r in results) / len(results)
total_cost = sum(r["e2e"].cost_usd for r in results)

print(f"\n{'='*80}")
print("EVALUATION SUMMARY")
print(f"{'='*80}")
print(f"\nRetrieval Metrics:")
print(f"  Average Precision@5: {avg_precision:.2%}")
print(f"  Average Recall@5: {avg_recall:.2%}")
print(f"\nGeneration Metrics:")
print(f"  Average Relevance: {avg_relevance:.2%}")
print(f"\nPerformance Metrics:")
print(f"  Average Latency: {avg_latency:.0f}ms")
print(f"  Total Cost: ${total_cost:.4f}")
print(f"\nEvaluated {len(results)} queries")
```

## Mock LLM for Testing

If you don't have an OpenAI API key, use a mock LLM for demonstration:

```python title="mock_evaluation.py"
class MockLLMClient:
    """Mock LLM for testing without API key."""

    def __init__(self):
        self._responses = []
        self.model_name = "mock-llm"

    def set_responses(self, responses):
        """Set canned responses."""
        self._responses = responses

    def generate(self, prompt, **kwargs):
        """Return next canned response."""
        if self._responses:
            return self._responses.pop(0)
        return "8/10"  # Default score

# Use mock LLM
mock_llm = MockLLMClient()
mock_llm.set_responses([
    "8/10",  # Relevance
    "9/10",  # Faithfulness
    "1/10",  # Hallucination
])

judge = OpenAIJudge(llm=mock_llm)
generation_eval = StandardGenerationEvaluator(judge)
```

## Interpreting Results

### Retrieval Metrics

| Metric | Range | Good Threshold | Interpretation |
|--------|-------|----------------|----------------|
| **Precision@K** | 0-1 | ≥0.7 | Fraction of retrieved docs that are relevant |
| **Recall@K** | 0-1 | ≥0.5 | Fraction of relevant docs that were retrieved |
| **MRR** | 0-1 | ≥0.8 | How high the first relevant doc ranks |
| **NDCG** | 0-1 | ≥0.7 | Quality of ranking (higher relevant docs = better) |

### Generation Metrics

| Metric | Range | Good Threshold | Interpretation |
|--------|-------|----------------|----------------|
| **Relevance** | 0-1 | ≥0.7 | Answer addresses the question |
| **Faithfulness** | 0-1 | ≥0.7 | Answer grounded in context |
| **Hallucination** | 0-1 | ≤0.3 | Lower is better - info not in context |

### Performance Metrics

| Metric | Good Threshold | Interpretation |
|--------|----------------|----------------|
| **Latency** | ≤1000ms | Total response time |
| **Cost** | Varies | Estimated API cost |
| **Context Utilization** | 0.5-0.9 | Percentage of context window used |

## Quality Assessment Example

```python title="assess_quality.py"
def assess_quality(results):
    """Assess overall quality from evaluation results."""

    # Aggregate metrics
    avg_precision = sum(r["retrieval"].precision_at_k[5] for r in results) / len(results)
    avg_relevance = sum(r["generation"].relevance_score for r in results) / len(results)
    avg_faithfulness = sum(r["generation"].faithfulness_score for r in results) / len(results)
    avg_hallucination = sum(r["generation"].hallucination_score for r in results) / len(results)
    avg_latency = sum(r["e2e"].total_latency_ms for r in results) / len(results)

    print("\nQuality Assessment:")

    # Retrieval quality
    if avg_precision >= 0.7:
        print("  ✓ Retrieval precision is GOOD (≥70%)")
    elif avg_precision >= 0.5:
        print("  ⚠ Retrieval precision is FAIR (50-70%)")
    else:
        print("  ✗ Retrieval precision is POOR (<50%)")

    # Answer relevance
    if avg_relevance >= 0.7:
        print("  ✓ Answer relevance is GOOD (≥70%)")
    elif avg_relevance >= 0.5:
        print("  ⚠ Answer relevance is FAIR (50-70%)")
    else:
        print("  ✗ Answer relevance is POOR (<50%)")

    # Faithfulness
    if avg_faithfulness >= 0.7:
        print("  ✓ Faithfulness is GOOD (≥70%)")
    elif avg_faithfulness >= 0.5:
        print("  ⚠ Faithfulness is FAIR (50-70%)")
    else:
        print("  ✗ Faithfulness is POOR (<50%)")

    # Hallucination
    if avg_hallucination <= 0.3:
        print("  ✓ Hallucination rate is LOW (≤30%)")
    elif avg_hallucination <= 0.5:
        print("  ⚠ Hallucination rate is MODERATE (30-50%)")
    else:
        print("  ✗ Hallucination rate is HIGH (>50%)")

    # Latency
    if avg_latency <= 1000:
        print("  ✓ Latency is GOOD (≤1s)")
    elif avg_latency <= 2000:
        print("  ⚠ Latency is FAIR (1-2s)")
    else:
        print("  ✗ Latency is SLOW (>2s)")

    return {
        "retrieval_quality": "good" if avg_precision >= 0.7 else "fair" if avg_precision >= 0.5 else "poor",
        "generation_quality": "good" if avg_relevance >= 0.7 else "fair" if avg_relevance >= 0.5 else "poor",
        "performance": "good" if avg_latency <= 1000 else "fair" if avg_latency <= 2000 else "poor"
    }

# Usage
assessment = assess_quality(results)
print(f"\nOverall Assessment: {assessment}")
```

## Best Practices

### 1. Create Representative Datasets

```python
# Good: Diverse queries covering different use cases
examples = [
    {"query": "What is X?", ...},           # Factual
    {"query": "How do I Y?", ...},          # Procedural
    {"query": "Compare A and B", ...},      # Comparison
    {"query": "Why does Z happen?", ...},   # Causal
]

# Bad: Only one type of query
examples = [
    {"query": "What is X?", ...},
    {"query": "What is Y?", ...},
    {"query": "What is Z?", ...},
]
```

### 2. Use Multiple Metrics

```python
# Good: Evaluate multiple dimensions
retrieval_metrics = retrieval_eval.evaluate_retrieval(...)
generation_metrics = generation_eval.evaluate_answer(...)
e2e_metrics = observer.get_results().end_to_end_metrics

# Bad: Only one metric
precision = calculate_precision(...)
```

### 3. Test with Real Data

```python
# Good: Use actual production queries
dataset = JSONLDataset("production_queries.jsonl")

# Bad: Only synthetic data
dataset = create_synthetic_data()
```

### 4. Monitor Over Time

```python
# Save results for trend analysis
import json
from datetime import datetime

results_file = f"eval_results_{datetime.now():%Y%m%d}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### LLM Judge Returns Invalid Scores

```python
# Ensure proper prompt formatting
judge = OpenAIJudge(llm=llm)

# Check LLM response
response = llm.generate("Rate this answer: ...")
print(f"Raw response: {response}")

# Should contain score like "8/10" or "Score: 7"
```

### Missing Relevant Documents

```python
# Check ground truth IDs match retrieved IDs
print(f"Ground truth: {example.relevant_doc_ids}")
print(f"Retrieved: {[doc['id'] for doc in retrieved_docs]}")

# IDs must match exactly (case-sensitive)
```

### High Latency

```python
# Use faster models for LLM judge
judge = OpenAIJudge(llm=OpenAILLMClient(model="gpt-3.5-turbo"))

# Or reduce top_k for retrieval
retrieval_eval = StandardRetrievalEvaluator(k_values=[5])  # Instead of [5, 10, 20]
```

## See Also

- [Evaluation Guide](../guides/evaluation.md) - Comprehensive evaluation documentation
- [Evaluation API Reference](../api/evaluation/index.md) - Complete API documentation
- [evaluation_example.py](../../examples/evaluation_example.py) - Full working example
- [RAG Pipeline](../guides/rag_pipeline.md) - Pipeline integration
