# Installation

## Requirements

- Python 3.11 or higher
- pip (recommended) or conda
- Git (for development installation)

## Basic Installation

Install the core library with minimal dependencies:

```bash
pip install rag-toolkit
```

This includes:
- Core protocols and types
- Pydantic models
- Basic infrastructure

## Optional Dependencies

rag-toolkit uses optional dependencies to keep the core library lightweight. Install only what you need:

### LLM Providers

**Ollama** (local LLM server):
```bash
pip install rag-toolkit[ollama]
```

**OpenAI** (GPT models):
```bash
pip install rag-toolkit[openai]
```

### Document Parsing

**PDF Support**:
```bash
pip install rag-toolkit[pdf]
```

**DOCX Support**:
```bash
pip install rag-toolkit[docx]
```

**OCR Support** (for scanned documents):
```bash
pip install rag-toolkit[ocr]
```

### Language Detection

**Language Detection**:
```bash
pip install rag-toolkit[lang]
```

### Combined Installations

**All LLM providers**:
```bash
pip install rag-toolkit[ollama,openai]
```

**All document parsers**:
```bash
pip install rag-toolkit[pdf,docx,ocr]
```

**Everything**:
```bash
pip install rag-toolkit[all]
```

## Development Installation

For contributing to the project or running from source:

```bash
# Clone the repository
git clone https://github.com/gmottola00/rag-toolkit.git
cd rag-toolkit

# Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"
```

This installs:
- All dev tools (pytest, mypy, ruff, black, isort)
- Documentation tools (Sphinx, Furo theme)
- All optional dependencies for testing

## Verification

Verify your installation works:

```python
# Test basic imports
from rag_toolkit import RagPipeline
from rag_toolkit.core import VectorStoreClient, EmbeddingClient, LLMClient

print("✅ Core installation verified!")

# Test optional imports (if installed)
try:
    from rag_toolkit import get_ollama_embedding
    print("✅ Ollama support available")
except ImportError:
    print("ℹ️  Ollama not installed (optional)")

try:
    from rag_toolkit import get_openai_embedding
    print("✅ OpenAI support available")
except ImportError:
    print("ℹ️  OpenAI not installed (optional)")
```

## External Services

### Milvus (Vector Store)

rag-toolkit uses Milvus as the default vector store. Run it with Docker:

```bash
# Using docker-compose (recommended)
docker-compose up -d milvus

# Or using the Makefile
make docker-milvus
```

Alternative: Use Milvus Lite (embedded):
```bash
pip install milvus
```

### Ollama (Optional)

For local LLM inference:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or with Docker
docker run -d -p 11434:11434 ollama/ollama

# Pull a model
ollama pull llama2
ollama pull nomic-embed-text
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, install the required optional dependency:

```python
# Error: No module named 'ollama'
# Solution:
pip install rag-toolkit[ollama]
```

### Type Checking Issues

If using mypy, install type stubs:
```bash
pip install types-requests
```

### Docker Issues

If Milvus won't start:
```bash
# Check if ports are available
lsof -i :19530

# Clean up and restart
docker-compose down -v
docker-compose up -d milvus
```

## Next Steps

- Follow the [Quick Start](quickstart.md) guide
- Learn about [Core Concepts](user_guide/core_concepts.md)
- Explore [Examples](examples/index.md)

## Platform-Specific Notes

### macOS

On Apple Silicon (M1/M2/M3), some dependencies may need Rosetta 2:
```bash
softwareupdate --install-rosetta
```

### Linux

Ensure you have Python 3.11+ development headers:
```bash
# Ubuntu/Debian
sudo apt-get install python3.11-dev

# CentOS/RHEL
sudo yum install python311-devel
```

### Windows

Use WSL2 for best compatibility:
```bash
wsl --install
```

## Update Instructions

To update to the latest version:

```bash
pip install --upgrade rag-toolkit
```

To update with all optional dependencies:
```bash
pip install --upgrade rag-toolkit[all]
```
