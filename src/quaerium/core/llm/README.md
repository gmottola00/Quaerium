# LLM Clients Architecture

Interfacce e client LLM intercambiabili per generazione testi/chat.

## Architettura

- `base.py`: interfaccia `LLMClient` (metodi `generate`, `generate_batch`, proprietà `model_name`).
- `ollama.py`: `OllamaLLMClient` verso Ollama `/api/generate` (env `OLLAMA_URL`, `OLLAMA_LLM_MODEL`).
- `openai_llm.py`: `OpenAILLMClient` per chat completions OpenAI (env `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_LLM_MODEL`).
- `__init__.py`: re-export dei client e dell’interfaccia.

## Perché questa architettura

- Separazione interfacce/implementazioni per sostituire provider senza toccare il codice a monte.
- Configurazione via environment variables per switching rapido (locale Ollama vs cloud OpenAI).
- Facilita test/mocking: si inietta l’interfaccia e si sostituisce l’implementazione.

## Uso rapido

```python
from core.llm import OllamaLLMClient, OpenAILLMClient

llm = OllamaLLMClient()
resp = llm.generate("Spiega in breve l'oggetto della gara.")

llm_openai = OpenAILLMClient()
resp2 = llm_openai.generate("Riassumi il testo seguente: ...")
```

## Combo embedding + LLM consigliate (local test)

- Embedding: `nomic-embed-text` o `all-minilm` / `bge-small`
- LLM: `phi3:mini`, `mistral:instruct` (7B), `llama3:instruct` (8B)

Remoto: `text-embedding-3-small` + `gpt-4o-mini` (OpenAI).

## Note operative

- Ollama: avvia con `ollama serve`, installa modelli con `ollama pull <modello>`.
- OpenAI: richiede `OPENAI_API_KEY`; opzionale `OPENAI_BASE_URL` per proxy/self-hosted compatibili.
