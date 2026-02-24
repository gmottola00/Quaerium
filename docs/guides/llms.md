# :material-robot: LLM Clients

Large Language Models (LLMs) power the generation phase of RAG systems. Master LLM integration for high-quality, contextual responses.

---

## :material-information: Overview

!!! abstract "What LLM Clients Do"
    LLM clients in RAG Toolkit handle the entire text generation pipeline.

**Capabilities:**

<div class="grid cards" markdown>

- :material-text: **Text Generation**

    ---

    Generate natural language answers from prompts

- :material-file-document-multiple: **Context Integration**

    ---

    Combine retrieved documents with user queries

- :material-wave: **Streaming**

    ---

    Real-time response generation

- :material-shield-alert: **Error Handling**

    ---

    Automatic retries, rate limiting, fallbacks

</div>

---

## :material-server: Supported LLM Providers

!!! info "Choose Your Provider"

### OpenAI (Recommended)

OpenAI provides state-of-the-art models with excellent quality and reliability.

**Models:**
- `gpt-4-turbo`: Latest GPT-4, best quality, 128k context
- `gpt-4`: Standard GPT-4, 8k context
- `gpt-3.5-turbo`: Fast and cost-effective, 16k context

**Installation:**

```bash
pip install quaerium[openai]
export OPENAI_API_KEY="your-api-key"
```

**Usage:**

```python
from quaerium.infra.llm import OpenAILLM

# Initialize
llm = OpenAILLM(
    model="gpt-4-turbo",
    api_key="your-api-key",  # Or set OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=1000,
)

# Generate response
response = await llm.generate(
    prompt="What is machine learning?",
)
print(response)

# Generate with system message
response = await llm.generate(
    prompt="Explain quantum computing",
    system_message="You are a helpful physics teacher."
)
```

**Pricing** (as of Dec 2024):
- `gpt-4-turbo`: $10 / 1M input tokens, $30 / 1M output tokens
- `gpt-4`: $30 / 1M input tokens, $60 / 1M output tokens
- `gpt-3.5-turbo`: $0.50 / 1M input tokens, $1.50 / 1M output tokens

### Ollama (Local, Free)

Run powerful LLMs locally with Ollama for privacy and zero API costs.

**Popular Models:**
- `llama3`: Meta's Llama 3, excellent quality
- `mistral`: Mistral 7B, fast and capable
- `phi3`: Microsoft Phi-3, efficient small model
- `gemma`: Google Gemma, strong performance

**Installation:**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Install quaerium with Ollama support
pip install quaerium[ollama]
```

**Usage:**

```python
from quaerium.infra.llm import OllamaLLM

# Initialize
llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7,
)

# Generate response
response = await llm.generate(
    prompt="What is machine learning?",
)
print(response)

# With system message
response = await llm.generate(
    prompt="Explain neural networks",
    system_message="You are a helpful AI teacher."
)
```

**Model Comparison:**

| Model | Size | Speed | Quality | Context | Use Case |
|-------|------|-------|---------|---------|----------|
| `llama3` | 8B | Medium | Excellent | 8k | General purpose |
| `mistral` | 7B | Fast | Very good | 32k | Long context |
| `phi3` | 3.8B | Very fast | Good | 4k | Speed critical |
| `gemma` | 7B | Medium | Very good | 8k | Balanced |

## Configuration

### Temperature

Control randomness in responses:

```python
# Deterministic (factual responses)
llm = OpenAILLM(
    model="gpt-4-turbo",
    temperature=0.0,  # No randomness
)

# Balanced (default)
llm = OpenAILLM(
    model="gpt-4-turbo",
    temperature=0.7,  # Some creativity
)

# Creative
llm = OpenAILLM(
    model="gpt-4-turbo",
    temperature=1.0,  # Maximum creativity
)
```

### Max Tokens

Limit response length:

```python
llm = OpenAILLM(
    model="gpt-4-turbo",
    max_tokens=500,  # Maximum 500 tokens in response
)

# For summaries
llm_summary = OpenAILLM(model="gpt-4-turbo", max_tokens=200)

# For detailed explanations
llm_detailed = OpenAILLM(model="gpt-4-turbo", max_tokens=2000)
```

### Timeout

Set request timeouts:

```python
llm = OpenAILLM(
    model="gpt-4-turbo",
    timeout=60.0,  # 60 seconds (default: 120)
)
```

### Retry Logic

Handle failures gracefully:

```python
llm = OpenAILLM(
    model="gpt-4-turbo",
    max_retries=3,  # Retry failed requests
    retry_delay=1.0,  # Wait 1 second between retries
)
```

## Advanced Usage

### Streaming Responses

Stream responses for real-time display:

```python
# Stream response tokens as they're generated
async for chunk in llm.generate_stream(
    prompt="Explain quantum mechanics in detail",
):
    print(chunk, end="", flush=True)
print()  # New line after streaming
```

### Chat History

Maintain conversation context:

```python
from quaerium.core.llm import Message

# Build chat history
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is Python?"),
    Message(role="assistant", content="Python is a programming language."),
    Message(role="user", content="Give me an example."),
]

# Generate with history
response = await llm.generate_with_history(messages=messages)
print(response)
```

### Function Calling

Use structured outputs (OpenAI only):

```python
# Define function
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Generate with function calling
response = await llm.generate(
    prompt="Find documents about machine learning",
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### JSON Mode

Force structured JSON output (OpenAI only):

```python
llm = OpenAILLM(
    model="gpt-4-turbo",
    response_format={"type": "json_object"}
)

response = await llm.generate(
    prompt="""
    Extract information from this text as JSON:
    "John Smith is 30 years old and works as a software engineer in San Francisco."
    
    Return JSON with fields: name, age, occupation, location
    """
)

import json
data = json.loads(response)
print(data)  # {"name": "John Smith", "age": 30, ...}
```

## Integration with RAG

### Basic RAG Query

```python
from quaerium import RagPipeline
from quaerium.infra.embedding import OpenAIEmbedding
from quaerium.infra.vectorstores.milvus import MilvusVectorStore
from quaerium.infra.llm import OpenAILLM

# Setup LLM
llm = OpenAILLM(
    model="gpt-4-turbo",
    temperature=0.7,
)

# Create RAG pipeline
pipeline = RagPipeline(
    embedding_client=OpenAIEmbedding(),
    vector_store=MilvusVectorStore(
        collection_name="documents",
        embedding_client=OpenAIEmbedding(),
    ),
    llm_client=llm,
)

# Query with automatic context retrieval
result = await pipeline.query(
    "What are the key findings in the research papers?"
)

print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)} documents used")
```

### Custom Prompts

Customize how context is presented to the LLM:

```python
# Custom prompt template
custom_prompt = """
You are a research assistant analyzing scientific papers.

Context from papers:
{context}

Question: {question}

Provide a detailed answer with citations.
"""

pipeline = RagPipeline(
    llm_client=llm,
    prompt_template=custom_prompt,
    # ... other config
)
```

### Multi-Step Reasoning

Break complex queries into steps:

```python
# Step 1: Decompose query
decomposition_prompt = """
Break this complex question into simpler sub-questions:
{question}
"""

sub_questions = await llm.generate(
    prompt=decomposition_prompt.format(question=query)
)

# Step 2: Answer each sub-question
answers = []
for sub_q in sub_questions:
    answer = await pipeline.query(sub_q)
    answers.append(answer)

# Step 3: Synthesize final answer
synthesis_prompt = """
Synthesize a final answer from these sub-answers:
{answers}

Original question: {question}
"""

final_answer = await llm.generate(
    prompt=synthesis_prompt.format(
        answers="\n\n".join(answers),
        question=query
    )
)
```

## Model Selection Guide

### By Quality

**Best Quality (OpenAI):**
```python
llm = OpenAILLM(model="gpt-4-turbo")
# 128k context, best reasoning
# Use for: Complex tasks, production systems
```

**Good Quality (Ollama, Free):**
```python
llm = OllamaLLM(model="llama3")
# 8k context, excellent for most tasks
# Use for: General purpose, privacy-sensitive
```

### By Speed

**Fastest (Ollama, Local):**
```python
llm = OllamaLLM(model="phi3")
# Very fast, good quality
# Use for: Real-time applications
```

**Fast (OpenAI):**
```python
llm = OpenAILLM(model="gpt-3.5-turbo")
# Fast API, good quality
# Use for: Most applications
```

### By Cost

**Free (Ollama):**
```python
llm = OllamaLLM(model="llama3")
# Zero API costs
# Cost: GPU/CPU time only
```

**Cost-Effective (OpenAI):**
```python
llm = OpenAILLM(model="gpt-3.5-turbo")
# $0.50 / 1M input tokens
# Use for: Budget-conscious applications
```

### By Context Window

**Longest Context:**
```python
# OpenAI GPT-4-turbo: 128k tokens
llm = OpenAILLM(model="gpt-4-turbo")

# Ollama Mistral: 32k tokens
llm = OllamaLLM(model="mistral")
```

## Custom LLM Clients

Implement your own LLM provider following the protocol:

```python
from typing import Protocol, runtime_checkable, AsyncIterator

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients."""
    
    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text completion."""
        ...
    
    async def generate_stream(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Generate text completion with streaming."""
        ...
```

### Example: Anthropic Claude

```python
from anthropic import AsyncAnthropic

class AnthropicLLM:
    """Anthropic Claude LLM client."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate completion."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            system=system_message or "",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    async def generate_stream(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Generate with streaming."""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            system=system_message or "",
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk

# Usage
llm = AnthropicLLM()
response = await llm.generate("Explain quantum computing")
```

## Performance Optimization

### Caching Responses

Cache common queries:

```python
from functools import lru_cache

class CachedLLM:
    """LLM client with response caching."""
    
    def __init__(self, llm_client):
        self.client = llm_client
        self._generate_cached = lru_cache(maxsize=1000)(
            self._generate_sync
        )
    
    def _generate_sync(self, prompt: str) -> str:
        """Synchronous wrapper for caching."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.client.generate(prompt)
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with caching."""
        # Only cache without extra parameters
        if not kwargs:
            return self._generate_cached(prompt)
        return await self.client.generate(prompt, **kwargs)

# Usage
base_llm = OpenAILLM(model="gpt-4-turbo")
cached_llm = CachedLLM(base_llm)

# First call: API request
r1 = await cached_llm.generate("What is AI?")  # API call

# Second call: from cache
r2 = await cached_llm.generate("What is AI?")  # No API call
```

### Batch Processing

Process multiple prompts efficiently:

```python
import asyncio

async def batch_generate(
    prompts: list[str],
    llm_client,
    max_concurrent: int = 5
):
    """Generate responses in parallel with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_limit(prompt: str):
        async with semaphore:
            return await llm_client.generate(prompt)
    
    tasks = [generate_with_limit(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Usage
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "Describe gradient descent"
]

responses = await batch_generate(
    prompts=prompts,
    llm_client=llm,
    max_concurrent=10
)
```

### Token Estimation

Estimate costs before calling API:

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(
    prompt: str,
    expected_response_length: int,
    model: str = "gpt-4-turbo"
) -> float:
    """Estimate API cost in USD."""
    # Pricing per 1M tokens
    prices = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }
    
    input_tokens = estimate_tokens(prompt, model)
    output_tokens = expected_response_length
    
    price = prices.get(model, prices["gpt-4-turbo"])
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]
    
    return input_cost + output_cost

# Usage
prompt = "Explain quantum computing in detail..."
cost = estimate_cost(prompt, expected_response_length=500)
print(f"Estimated cost: ${cost:.4f}")
```

## Monitoring and Debugging

### Response Tracking

Track API usage and costs:

```python
class TrackedLLM:
    """LLM client with usage tracking."""
    
    def __init__(self, llm_client):
        self.client = llm_client
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with tracking."""
        self.total_calls += 1
        
        # Estimate input tokens
        self.total_input_tokens += len(prompt) // 4
        
        response = await self.client.generate(prompt, **kwargs)
        
        # Estimate output tokens
        self.total_output_tokens += len(response) // 4
        
        return response
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }
    
    def estimate_cost(self, model: str = "gpt-4-turbo") -> float:
        """Estimate total cost."""
        prices = {
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        }
        price = prices.get(model, prices["gpt-4-turbo"])
        
        input_cost = (self.total_input_tokens / 1_000_000) * price["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * price["output"]
        
        return input_cost + output_cost

# Usage
tracked = TrackedLLM(OpenAILLM())
await tracked.generate("What is AI?")
await tracked.generate("Explain ML")

stats = tracked.get_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Estimated cost: ${tracked.estimate_cost():.4f}")
```

### Response Quality Check

Validate response quality:

```python
async def generate_with_validation(
    prompt: str,
    llm_client,
    max_retries: int = 3
) -> str:
    """Generate with quality validation."""
    for attempt in range(max_retries):
        response = await llm_client.generate(prompt)
        
        # Validation checks
        if len(response) < 10:
            print(f"Attempt {attempt + 1}: Response too short, retrying...")
            continue
        
        if "I don't know" in response and attempt < max_retries - 1:
            print(f"Attempt {attempt + 1}: Uncertain response, retrying...")
            continue
        
        return response
    
    raise ValueError("Failed to generate valid response")

# Usage
response = await generate_with_validation(
    "What is quantum computing?",
    llm_client=llm
)
```

## Troubleshooting

### API Key Issues

```python
from quaerium.infra.llm import OpenAILLM

try:
    llm = OpenAILLM()
    await llm.generate("test")
except Exception as e:
    if "api_key" in str(e).lower():
        print("❌ Invalid or missing API key")
        print("Set OPENAI_API_KEY environment variable")
```

### Rate Limiting

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5)
)
async def generate_with_retry(prompt: str, llm_client):
    """Generate with exponential backoff for rate limits."""
    try:
        return await llm_client.generate(prompt)
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate limited, backing off...")
            raise
        raise

# Usage
response = await generate_with_retry("What is AI?", llm)
```

### Context Length Errors

```python
def truncate_to_context_limit(
    text: str,
    max_tokens: int = 8000,
    model: str = "gpt-4"
) -> str:
    """Truncate text to fit context window."""
    import tiktoken
    
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Usage
long_prompt = "..." * 10000  # Very long prompt
safe_prompt = truncate_to_context_limit(long_prompt, max_tokens=7000)
response = await llm.generate(safe_prompt)
```

## Best Practices

1. **Choose the Right Model**
   - Production: `gpt-4-turbo` (best quality)
   - Cost-effective: `gpt-3.5-turbo`
   - Privacy/offline: `llama3` (Ollama)

2. **Prompt Engineering**
   - Be specific and clear
   - Provide examples (few-shot)
   - Use system messages for role/context
   - Structure with clear sections

3. **Error Handling**
   - Implement retries with exponential backoff
   - Handle rate limits gracefully
   - Validate response quality
   - Have fallback strategies

4. **Cost Optimization**
   - Cache common queries
   - Use cheaper models when appropriate
   - Estimate costs before making calls
   - Monitor usage regularly

5. **Performance**
   - Use streaming for better UX
   - Batch requests when possible
   - Set appropriate timeouts
   - Implement concurrency limits

6. **Quality Assurance**
   - Use temperature=0 for consistency
   - Validate responses
   - A/B test different models
   - Monitor response quality

## Next Steps

- [RAG Pipeline](rag_pipeline.md) - Build complete RAG systems
- [Embeddings Guide](embeddings.md) - Learn about embeddings
- [Advanced Pipeline Example](../examples/advanced_pipeline.md)
- [Production Setup](../examples/production_setup.md)

## See Also

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Documentation](https://ollama.com/docs)
- [LLM Protocol](protocols.md#llmclient)
