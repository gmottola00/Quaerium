"""End-to-end RAG pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from quaerium.core.index.base import SearchStrategy
from quaerium.core.llm import LLMClient
from quaerium.rag.assembler import ContextAssembler
from quaerium.rag.models import RagResponse, RetrievedChunk
from quaerium.rag.rerankers import LLMReranker
from quaerium.rag.rewriter import QueryRewriter


class RagPipeline:
    """RAG pipeline with query rewriting, vector retrieval, reranking, assembly, generation."""

    def __init__(
        self,
        *,
        vector_searcher: SearchStrategy,
        rewriter: QueryRewriter,
        reranker: LLMReranker,
        assembler: ContextAssembler,
        generator_llm: LLMClient,
        observers: List[Any] | None = None,
    ) -> None:
        self.vector_searcher = vector_searcher
        self.rewriter = rewriter
        self.reranker = reranker
        self.assembler = assembler
        self.generator_llm = generator_llm
        self.observers = observers or []

    def run(self, question: str, *, metadata_hint: Dict[str, str] | None = None, top_k: int = 5) -> RagResponse:
        """Execute the RAG flow."""
        # Query rewriting
        rewritten = self.rewriter.rewrite(question, metadata_hint=metadata_hint)
        self._notify_observers("on_query_rewrite", question, rewritten, {})

        # Vector retrieval
        vec_hits = self.vector_searcher.search(rewritten, top_k=top_k)
        self._notify_observers("on_retrieval", rewritten, vec_hits, {})

        # Reranking
        reranked_hits = self.reranker.rerank(question, vec_hits, top_k=top_k)
        self._notify_observers("on_reranking", question, vec_hits, reranked_hits, {})
        reranked = [self._to_retrieved_chunk(hit) for hit in reranked_hits]

        # Context assembly
        context_chunks = self.assembler.assemble(reranked)
        token_count = sum(len(c.text.split()) * 1.3 for c in context_chunks)  # Rough estimate
        self._notify_observers("on_context_assembly", [c.__dict__ for c in context_chunks], int(token_count), {})

        # Answer generation
        answer = self._generate_answer(question, context_chunks)
        context_texts = [c.text for c in context_chunks]
        self._notify_observers("on_generation", question, answer, context_texts, {})

        return RagResponse(answer=answer, citations=context_chunks)

    def _to_retrieved_chunk(self, hit: Dict[str, object]) -> RetrievedChunk:
        return RetrievedChunk(
            id=str(hit.get("id", "")),
            text=str(hit.get("text", "")),
            section_path=hit.get("section_path"),
            metadata=hit.get("metadata") or {},
            page_numbers=hit.get("page_numbers") or [],
            source_chunk_id=hit.get("source_chunk_id"),
            score=hit.get("score") if isinstance(hit.get("score"), (int, float)) else None,
        )

    def _generate_answer(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context_parts = [f"{c.section_path or ''}\n{c.text}" for c in chunks]
        context = "\n\n".join(context_parts)
        prompt = (
            "Sei un assistente per gare e appalti. Rispondi in modo conciso usando solo il contesto fornito.\n"
            "Includi riferimenti puntuali (sezione/path) se possibile. Se non trovi la risposta, di' che non è presente.\n"
            f"Domanda: {question}\n\n"
            f"Contesto:\n{context}\n\n"
            "Risposta:"
        )
        return self.generator_llm.generate(prompt)

    def _notify_observers(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Notify all observers of a pipeline event."""
        for observer in self.observers:
            if hasattr(observer, method_name):
                try:
                    getattr(observer, method_name)(*args, **kwargs)
                except Exception as e:
                    # Log but don't fail the pipeline
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Observer {observer.__class__.__name__}.{method_name} failed: {e}"
                    )


__all__ = ["RagPipeline"]
