# Contributing to quaerium

Thank you for your interest in contributing to quaerium! This document provides guidelines and instructions for contributors.

## 🎯 Development Philosophy

- **Clean code**: Readable, maintainable, well-documented
- **Type safety**: Full type hints, mypy compliance
- **Test coverage**: Aim for >80% coverage
- **Protocol-based**: Favor composition over inheritance
- **Performance**: Optimize for production use cases

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/quaerium.git
cd quaerium
```

### 2. Setup Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/issue-123
```

## 📝 Code Standards

### Style Guide

We follow **PEP 8** with these tools:

```bash
# Format code
black src/quaerium tests

# Sort imports
isort src/quaerium tests

# Lint
ruff check src/quaerium tests

# Type check
mypy src/quaerium
```

### Type Hints

All public APIs must have complete type hints:

```python
# ✅ Good
def embed(self, text: str) -> list[float]:
    """Convert text to embedding vector."""
    return self._model.encode(text).tolist()

# ❌ Bad
def embed(self, text):
    return self._model.encode(text).tolist()
```

### Docstrings

Use **Google-style docstrings**:

```python
def search(
    self,
    query: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None
) -> list[SearchResult]:
    """
    Search for similar documents.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Optional metadata filters
        
    Returns:
        List of search results ordered by relevance
        
    Raises:
        ValueError: If top_k <= 0
        
    Example:
        >>> results = searcher.search("RAG tutorial", top_k=5)
        >>> for result in results:
        ...     print(result.text)
    """
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quaerium --cov-report=html

# Run specific test file
pytest tests/core/test_embedding.py

# Run specific test
pytest tests/core/test_embedding.py::test_embed_batch
```

### Writing Tests

```python
import pytest
from quaerium.core import EmbeddingClient

def test_embedding_protocol():
    """Test that custom embedding implements protocol."""
    
    class MockEmbedding:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 384
        
        @property
        def model_name(self) -> str:
            return "mock"
    
    # Should work with Protocol
    client: EmbeddingClient = MockEmbedding()
    vector = client.embed("test")
    
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)
```

### Mocking External Services

```python
@pytest.fixture
def mock_ollama():
    """Mock Ollama API responses."""
    with patch("quaerium.infra.embedding.ollama.requests.post") as mock:
        mock.return_value.json.return_value = {
            "embedding": [0.1] * 768
        }
        yield mock

def test_ollama_embedding(mock_ollama):
    """Test Ollama embedding with mocked API."""
    embedding = OllamaEmbedding(model="nomic-embed-text")
    vector = embedding.embed("test")
    
    assert len(vector) == 768
    mock_ollama.assert_called_once()
```

## 🔧 Adding New Features

### Adding a New Vector Store

1. **Create Protocol implementation** in `src/quaerium/infra/vectorstore/`:

```python
# src/quaerium/infra/vectorstore/pinecone/client.py
from quaerium.core import VectorStoreClient, SearchResult

class PineconeVectorStore:
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: str, environment: str):
        import pinecone
        pinecone.init(api_key=api_key, environment=environment)
    
    def create_collection(self, name: str, dimension: int, **kwargs):
        # Implementation
        pass
    
    # Implement all Protocol methods...
```

2. **Add tests** in `tests/infra/vectorstore/`:

```python
def test_pinecone_create_collection(pinecone_store):
    pinecone_store.create_collection("test", dimension=384)
    assert pinecone_store.collection_exists("test")
```

3. **Update documentation** in `docs/` and `README.md`

4. **Add to optional dependencies**:

```toml
[project.optional-dependencies]
pinecone = ["pinecone-client>=2.0.0"]
```

### Adding a New LLM Provider

Similar process in `src/quaerium/infra/llm/`:

```python
from quaerium.core import LLMClient

class AnthropicLLMClient:
    """Anthropic Claude LLM client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self._model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
    
    @property
    def model_name(self) -> str:
        return self._model
```

## 📚 Documentation

### Building Docs Locally

```bash
cd docs
make html
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
```

### Documentation Standards

- All public APIs must be documented
- Include usage examples
- Link to related concepts
- Keep examples up-to-date

## 🐛 Reporting Bugs

Use GitHub Issues with this template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Install quaerium with '...'
2. Run code '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- quaerium version: [e.g., 0.1.0]
- Relevant dependencies: [e.g., pymilvus 2.3.0]

**Additional context**
Any other context about the problem.
```

## 🎉 Pull Request Process

1. **Update tests**: Ensure all tests pass
2. **Update docs**: Add/update relevant documentation
3. **Update CHANGELOG**: Add entry under "Unreleased"
4. **Run checks**: `pytest && mypy src/quaerium && ruff check .`
5. **Create PR**: Use descriptive title and description

### PR Title Format

```
type(scope): brief description

Examples:
feat(vectorstore): add Pinecone support
fix(embedding): handle empty text input
docs(readme): update installation instructions
test(rag): add integration tests for pipeline
```

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Type checks pass (`mypy src/quaerium`)
- [ ] Linting passes (`ruff check .`)
- [ ] Code formatted (`black .` and `isort .`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] PR description explains changes

## 🔄 Development Workflow

1. Pick an issue or feature
2. Create branch
3. Make changes with tests
4. Run all checks
5. Push and create PR
6. Address review feedback
7. Merge!

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn

## 📞 Questions?

- **Discussions**: [GitHub Discussions](https://github.com/gmottola00/quaerium/discussions)
- **Issues**: [GitHub Issues](https://github.com/gmottola00/quaerium/issues)

---

**Thank you for contributing to quaerium!** 🚀
