"""
Complete evaluation example for Quaerium RAG.

This example demonstrates:
1. Loading an evaluation dataset
2. Creating retrieval and generation evaluators
3. Running a RAG pipeline with metrics observer
4. Evaluating retrieval and generation quality
5. Aggregating results across the dataset

Usage:
    python examples/evaluation_example.py
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run complete evaluation example."""

    # ========================================================================
    # Step 1: Import required modules
    # ========================================================================
    logger.info("Importing Quaerium modules...")

    from quaerium.infra.evaluation.datasets import JSONLDataset
    from quaerium.infra.evaluation.retrieval import StandardRetrievalEvaluator
    from quaerium.infra.evaluation.generation import (
        StandardGenerationEvaluator,
        OpenAIJudge
    )
    from quaerium.infra.evaluation.pipeline import MetricsObserver

    # NOTE: This example uses simulated pipeline execution for demonstration.
    # In production, you would use actual RAG pipeline components:
    #   from quaerium import RagPipeline
    #   from quaerium.infra.llm import OpenAIClient
    #   llm = OpenAIClient(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    #   pipeline = RagPipeline(..., observers=[observer])

    # Simple mock for demonstration purposes
    class MockLLMClient:
        def __init__(self):
            self._responses = []
            self.model_name = "mock-llm"
        def set_responses(self, r):
            self._responses = r
        def generate(self, prompt, **kwargs):
            return self._responses.pop(0) if self._responses else "8/10"

    # Check if we can use real OpenAI
    import os
    use_real_llm = False
    if os.getenv("OPENAI_API_KEY"):
        try:
            from quaerium.infra.llm.openai import OpenAILLMClient
            use_real_llm = True
        except ImportError:
            logger.warning("OpenAI not available, using mock for demonstration")

    # ========================================================================
    # Step 2: Load evaluation dataset
    # ========================================================================
    logger.info("Loading evaluation dataset...")

    dataset_path = Path("examples/data/sample_eval_dataset.jsonl")
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please ensure sample_eval_dataset.jsonl exists in examples/data/")
        return

    dataset = JSONLDataset(dataset_path)
    logger.info(f"Loaded dataset: {dataset.metadata.name}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"  Version: {dataset.metadata.version}")

    # ========================================================================
    # Step 3: Create evaluators
    # ========================================================================
    logger.info("Creating evaluators...")

    # Retrieval evaluator
    retrieval_eval = StandardRetrievalEvaluator(k_values=[5, 10])
    logger.info("  ✓ Retrieval evaluator (Precision@K, Recall@K, MRR, NDCG)")

    # Generation evaluator with LLM judge
    if use_real_llm:
        # Production usage with real LLM
        llm = OpenAILLMClient(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("  Using real OpenAI for LLM-as-judge")
    else:
        # Demo usage with mock
        llm = MockLLMClient()
        llm.set_responses([
            "8/10",  # Relevance score
            "9/10",  # Faithfulness score
            "1/10",  # Hallucination score
        ] * 100)  # Repeat for multiple examples
        logger.info("  Using mock LLM for demonstration")

    judge = OpenAIJudge(llm=llm)
    generation_eval = StandardGenerationEvaluator(judge)
    logger.info("  ✓ Generation evaluator (Relevance, Faithfulness, Hallucination)")

    # Pipeline observer for end-to-end metrics
    observer = MetricsObserver()
    logger.info("  ✓ Pipeline observer (Latency, Cost, Tokens)")

    # ========================================================================
    # Step 4: Simulate RAG pipeline execution
    # ========================================================================
    logger.info("\nRunning evaluation on dataset...")

    results = []

    for idx, example in enumerate(dataset):
        logger.info(f"\n[{idx+1}/{len(dataset)}] Evaluating: {example.query[:50]}...")

        # --- Simulate pipeline execution ---
        # In production, replace with:
        # pipeline = RagPipeline(..., observers=[observer])
        # response = pipeline.run(example.query)

        # For this example, we'll simulate the pipeline stages
        observer.on_query_rewrite(
            original=example.query,
            rewritten=example.query + " (expanded)",
            metadata={}
        )

        # Simulate retrieval
        retrieved_docs = [
            {"id": doc_id, "score": 0.9 - i * 0.1}
            for i, doc_id in enumerate(example.relevant_doc_ids[:3])
        ]
        observer.on_retrieval(
            query=example.query,
            results=retrieved_docs,
            metadata={}
        )

        # Simulate reranking
        observer.on_reranking(
            query=example.query,
            before=retrieved_docs,
            after=retrieved_docs,
            metadata={
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "model": "gpt-4"
            }
        )

        # Simulate context assembly
        context_chunks = [
            {"text": f"Context for {doc['id']}"} for doc in retrieved_docs
        ]
        observer.on_context_assembly(
            chunks=context_chunks,
            token_count=500,
            metadata={"max_tokens": 8192}
        )

        # Simulate generation
        generated_answer = f"Answer to: {example.query}"
        observer.on_generation(
            question=example.query,
            answer=generated_answer,
            context=[c["text"] for c in context_chunks],
            metadata={
                "usage": {"prompt_tokens": 600, "completion_tokens": 150},
                "model": "gpt-4"
            }
        )

        # --- Get end-to-end metrics ---
        e2e_result = observer.get_results()

        if not e2e_result.success:
            logger.warning(f"  ⚠ Pipeline failed: {e2e_result.error}")
            continue

        # --- Evaluate retrieval ---
        retrieval_metrics = retrieval_eval.evaluate_retrieval(
            query=example.query,
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=example.relevant_doc_ids
        )

        logger.info(f"  Retrieval:")
        logger.info(f"    Precision@5: {retrieval_metrics.precision_at_k[5]:.2%}")
        logger.info(f"    Recall@5: {retrieval_metrics.recall_at_k[5]:.2%}")
        logger.info(f"    MRR: {retrieval_metrics.mrr:.3f}")
        logger.info(f"    NDCG: {retrieval_metrics.ndcg:.3f}")

        # --- Evaluate generation ---
        generation_metrics = generation_eval.evaluate_answer(
            question=example.query,
            generated_answer=generated_answer,
            context=[c["text"] for c in context_chunks],
            reference_answer=example.reference_answer
        )

        logger.info(f"  Generation:")
        logger.info(f"    Relevance: {generation_metrics.relevance_score:.2%}")
        logger.info(f"    Faithfulness: {generation_metrics.faithfulness_score:.2%}")
        logger.info(f"    Hallucination: {generation_metrics.hallucination_score:.2%}")

        # --- End-to-end metrics ---
        e2e_metrics = e2e_result.end_to_end_metrics
        logger.info(f"  Performance:")
        logger.info(f"    Latency: {e2e_metrics.total_latency_ms:.0f}ms")
        logger.info(f"    Tokens: {e2e_metrics.total_tokens}")
        logger.info(f"    Cost: ${e2e_metrics.cost_usd:.4f}")
        logger.info(f"    Context util: {e2e_metrics.context_utilization:.2%}")

        # Store results
        results.append({
            "query": example.query,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "e2e": e2e_metrics
        })

    # ========================================================================
    # Step 5: Aggregate and report results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)

    if not results:
        logger.error("No results to aggregate")
        return

    # Aggregate retrieval metrics
    avg_precision_5 = sum(r["retrieval"].precision_at_k[5] for r in results) / len(results)
    avg_recall_5 = sum(r["retrieval"].recall_at_k[5] for r in results) / len(results)
    avg_mrr = sum(r["retrieval"].mrr for r in results) / len(results)
    avg_ndcg = sum(r["retrieval"].ndcg for r in results) / len(results)

    logger.info("\nRetrieval Metrics (Average):")
    logger.info(f"  Precision@5: {avg_precision_5:.2%}")
    logger.info(f"  Recall@5: {avg_recall_5:.2%}")
    logger.info(f"  MRR: {avg_mrr:.3f}")
    logger.info(f"  NDCG: {avg_ndcg:.3f}")

    # Aggregate generation metrics
    avg_relevance = sum(r["generation"].relevance_score for r in results) / len(results)
    avg_faithfulness = sum(r["generation"].faithfulness_score for r in results) / len(results)
    avg_hallucination = sum(r["generation"].hallucination_score for r in results) / len(results)

    logger.info("\nGeneration Metrics (Average):")
    logger.info(f"  Relevance: {avg_relevance:.2%}")
    logger.info(f"  Faithfulness: {avg_faithfulness:.2%}")
    logger.info(f"  Hallucination: {avg_hallucination:.2%}")

    # Aggregate performance metrics
    avg_latency = sum(r["e2e"].total_latency_ms for r in results) / len(results)
    total_tokens = sum(r["e2e"].total_tokens for r in results)
    total_cost = sum(r["e2e"].cost_usd for r in results)
    avg_context_util = sum(r["e2e"].context_utilization for r in results) / len(results)

    logger.info("\nPerformance Metrics:")
    logger.info(f"  Average Latency: {avg_latency:.0f}ms")
    logger.info(f"  Total Tokens: {total_tokens:,}")
    logger.info(f"  Total Cost: ${total_cost:.4f}")
    logger.info(f"  Average Context Utilization: {avg_context_util:.2%}")

    # Quality assessment
    logger.info("\nQuality Assessment:")

    if avg_precision_5 >= 0.7:
        logger.info("  ✓ Retrieval precision is GOOD (≥70%)")
    elif avg_precision_5 >= 0.5:
        logger.info("  ⚠ Retrieval precision is FAIR (50-70%)")
    else:
        logger.info("  ✗ Retrieval precision is POOR (<50%)")

    if avg_relevance >= 0.7:
        logger.info("  ✓ Answer relevance is GOOD (≥70%)")
    elif avg_relevance >= 0.5:
        logger.info("  ⚠ Answer relevance is FAIR (50-70%)")
    else:
        logger.info("  ✗ Answer relevance is POOR (<50%)")

    if avg_faithfulness >= 0.7:
        logger.info("  ✓ Answer faithfulness is GOOD (≥70%)")
    elif avg_faithfulness >= 0.5:
        logger.info("  ⚠ Answer faithfulness is FAIR (50-70%)")
    else:
        logger.info("  ✗ Answer faithfulness is POOR (<50%)")

    if avg_hallucination <= 0.3:
        logger.info("  ✓ Hallucination rate is LOW (≤30%)")
    elif avg_hallucination <= 0.5:
        logger.info("  ⚠ Hallucination rate is MODERATE (30-50%)")
    else:
        logger.info("  ✗ Hallucination rate is HIGH (>50%)")

    if avg_latency <= 1000:
        logger.info("  ✓ Latency is GOOD (≤1s)")
    elif avg_latency <= 2000:
        logger.info("  ⚠ Latency is FAIR (1-2s)")
    else:
        logger.info("  ✗ Latency is SLOW (>2s)")

    logger.info("\n" + "="*80)
    logger.info(f"Evaluated {len(results)} examples from {dataset.metadata.name}")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}", exc_info=True)
