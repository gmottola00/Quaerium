# Embedding Clients Architecture

Interfacce e client per generare embedding in modo intercambiabile.

## Architettura

- `base.py`: interfaccia `EmbeddingClient` (metodi `embed`, `embed_batch`, proprietà `model_name`, `dimension` opzionale).
- `ollama.py`: `OllamaEmbeddingClient` verso Ollama `/api/embeddings` (config via env `OLLAMA_URL`, `OLLAMA_EMBED_MODEL`).
- `openai_embedding.py`: `OpenAIEmbeddingClient` per API OpenAI (env `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBED_MODEL`).
- `__init__.py`: re-export dei client e dell’interfaccia.

## Perché questa architettura

- Interfaccia unica per cambiare modello/provider senza toccare l’applicazione.
- Iniezione della funzione di embedding ovunque serva (es. indicizzazione Milvus) senza dipendenza hard-coded.
- Configurazione via environment variables per switching rapido locale/remote.

## Uso rapido

```python
from core.embedding import OllamaEmbeddingClient, OpenAIEmbeddingClient

emb = OllamaEmbeddingClient()  # di default usa OLLAMA_URL e OLLAMA_EMBED_MODEL
vec = emb.embed("testo di esempio")

emb_openai = OpenAIEmbeddingClient()
vec2 = emb_openai.embed("testo di esempio")
```

## Installazione modelli (Ollama)

Esempi:
```
ollama pull nomic-embed-text
ollama pull all-minilm
ollama pull bge-small
```

## Combo consigliate (test locali su Mac)

- Embedding: `nomic-embed-text` oppure `all-minilm` (più rapido, dim 384), o `bge-small`.
- LLM companion: `phi3:mini`, `mistral:instruct` (7B), `llama3:instruct` (8B).

Se usi OpenAI: `text-embedding-3-small` + `gpt-4o-mini`.

