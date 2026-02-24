# RAG Layer Overview

Pipeline RAG modulare e riusabile: riscrive la query, recupera e reranka chunk, assembla il contesto rispettando il budget, genera risposta con citazioni.

## Componenti

- `models.py`
  - `RetrievedChunk`: passaggio recuperato (id, text, section_path, metadata, page_numbers, score).
  - `RagResponse`: risposta finale con answer + citations.
- `rewriter.py`
  - `QueryRewriter`: usa un LLM per riscrivere/arricchire la query (es. chiarire lotto/tender).
- `retrieval` (da `src/core/index/search/`)
  - `VectorSearcher`: retrieval semantico (Milvus + embedding).
  - `KeywordSearcher`/`HybridSearcher`: opzionali per recall keyword o ibrido.
- `rerankers.py`
  - `LLMReranker`: rerank dei candidati via LLM (placeholder, sostituibile da cross-encoder che implementa `Reranker`).
- `assembler.py`
  - `ContextAssembler`: seleziona chunk ordinati entro il budget (approx word count; sostituibile con conteggio token reale).
- `pipeline.py`
  - `RagPipeline`: orchestration end-to-end (rewrite → retrieval → rerank → assemble → generate).
- LLM/Embedding
  - Interfacce `LLMClient` / `EmbeddingClient` (Ollama/OpenAI); configurabili via env e iniettate nel pipeline.

## Flusso architetturale

1) **Rewrite**: normalizza/arricchisce la query con `QueryRewriter` (LLM).
2) **Vector retrieval**: `VectorSearcher` (Milvus + embedding) → top-k grezzi.
3) **Reranking**: `LLMReranker` ordina i candidati (cross-encoder plug-in futuro).
4) **Assembly**: `ContextAssembler` tronca ai chunk entro budget.
5) **Generation**: LLM finale costruisce la risposta; citazioni derivate dai chunk selezionati.

## Esempio d’uso

```python
from src.core.rag.pipeline import RagPipeline
from src.core.rag.rewriter import QueryRewriter
from src.core.rag.assembler import ContextAssembler
from src.core.rag.rerankers import LLMReranker
from src.core.index.search.vector_searcher import VectorSearcher
from src.core.llm import OllamaLLMClient

llm = OllamaLLMClient()
vector_searcher = VectorSearcher(indexer, embed_client)
pipeline = RagPipeline(
    vector_searcher=vector_searcher,
    rewriter=QueryRewriter(llm),
    reranker=LLMReranker(llm),
    assembler=ContextAssembler(max_tokens=2000),
    generator_llm=llm,
)

resp = pipeline.run("Domanda utente", top_k=5)
print(resp.answer)
for c in resp.citations:
    print(c.id, c.section_path, c.page_numbers)
```

## Estensioni previste

- Sostituire `LLMReranker` con un cross-encoder dedicato (implementa `Reranker`).
- Usare `HybridSearcher` (vector+keyword) al posto del solo `VectorSearcher`.
- Integrare un conteggio token reale (tokenizer del modello) nell’assembler.
- Filtri/recall guidato da metadata prima del retrieval (lot_id, tender_code, ecc.).

## Integrazione API

- `src/api/providers.py` espone singleton per embedding/indexer/searcher/pipeline.
- `src/api/routers/ingestion.py`: endpoint `/rag/pipeline` esegue l’intero flow (rewrite → vector → rerank → assemble → generate) e restituisce answer + citations.

---
[Torna al README core](../README.md)
