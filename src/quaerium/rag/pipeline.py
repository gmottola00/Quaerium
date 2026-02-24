"""End-to-end RAG pipeline."""

from __future__ import annotations

from typing import Dict, List

from quaerium.core.index.base import SearchStrategy
from quaerium.rag.assembler import ContextAssembler
from quaerium.rag.models import RagResponse, RetrievedChunk
from quaerium.rag.rewriter import QueryRewriter
from quaerium.rag.rerankers import LLMReranker
from quaerium.core.llm import LLMClient


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
    ) -> None:
        self.vector_searcher = vector_searcher
        self.rewriter = rewriter
        self.reranker = reranker
        self.assembler = assembler
        self.generator_llm = generator_llm

    def run(self, question: str, *, metadata_hint: Dict[str, str] | None = None, top_k: int = 5) -> RagResponse:
        """Execute the RAG flow."""
        rewritten = self.rewriter.rewrite(question, metadata_hint=metadata_hint)

        vec_hits = self.vector_searcher.search(rewritten, top_k=top_k)

        reranked_hits = self.reranker.rerank(question, vec_hits, top_k=top_k)
        reranked = [self._to_retrieved_chunk(hit) for hit in reranked_hits]

        context_chunks = self.assembler.assemble(reranked)
        answer = self._generate_answer(question, context_chunks)
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


__all__ = ["RagPipeline"]
